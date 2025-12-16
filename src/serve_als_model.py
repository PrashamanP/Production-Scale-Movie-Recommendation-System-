import logging
import os

# Set OpenBLAS threading to prevent conflicts with gunicorn workers
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import random
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

# Set OpenBLAS threading to prevent conflicts with gunicorn workers
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
from flask import Flask, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from flask import Flask, Response, jsonify
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    CONTENT_TYPE_LATEST,
    generate_latest,
)

from src.experimentation.experiment_router import (
    ExperimentAssignment,
    HashExperimentRouter,
    load_router,
)
from src.models.als.model import ALSRecommender
from src.models.popgen_loader import PopGenreRecommender
from src.provenance import ProvenanceTracker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from monitoring.custom_metrics import init_metrics, load_metrics_from_artifacts

# --- Artifact paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")
ALS_MODELS_DIR = "/app/models"
POPULARITY_FILE = os.path.join(ARTIFACTS_DIR, "popgen_popularity.parquet")
EXPERIMENTS_DIR = os.path.join(ARTIFACTS_DIR, "experiments")
EXPERIMENT_CONFIG_FILE = os.environ.get(
    "EXPERIMENT_CONFIG_PATH",
    os.path.join(EXPERIMENTS_DIR, "als_vs_pop.json"),
)

LOGGER = logging.getLogger("recommendations")
if not LOGGER.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")

METRICS_REGISTRY = CollectorRegistry()
REQUEST_COUNTER = Counter(
    "experiment_reco_requests_total",
    "Total recommendation requests served per variant.",
    ["variant"],
    registry=METRICS_REGISTRY,
)
LATENCY_HISTOGRAM = Histogram(
    "experiment_reco_latency_ms",
    "Latency of recommendation requests in milliseconds.",
    ["variant"],
    registry=METRICS_REGISTRY,
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000, float("inf")),
)

PROVENANCE_TRACKER: Optional[ProvenanceTracker] = None


class RecommendationService:
    """
    A service layer to orchestrate recommendation logic.
    It uses a primary recommender (ALS) and provides a fallback strategy.
    """
    def __init__(self, recommender: ALSRecommender, fallback_pool: List[str]):
        self._recommender = recommender
        self._fallback_pool = fallback_pool
        print("✅ RecommendationService initialized.")
        if not self._fallback_pool:
            print("⚠️ Warning: Fallback pool is empty. Cold-start recommendations will be disabled.")

    def get_recommendations(self, user_id: int, k: int) -> List[str]:
        """
        Generates recommendations for a user.
        If the primary recommender returns no results (cold-start),
        it uses the fallback pool.
        """
        recs = self._recommender.recommend(user_id=user_id, k=k)

        if not recs and self._fallback_pool:
            print(f"User {user_id} is a cold-start user. Serving from popular fallback pool.")
            num_to_sample = min(k, len(self._fallback_pool))
            return random.sample(self._fallback_pool, num_to_sample)
        
        return recs
    
    def is_ready(self) -> bool:
        """Check if the service is ready to serve traffic."""
        return self._recommender is not None


class ExperimentGateway:
    """Routes traffic across model variants according to an experiment router."""

    def __init__(
        self,
        router: HashExperimentRouter,
        variant_handlers: Dict[str, Callable[[int, int], List[str]]],
        default_variant: str,
        fallback_pool: List[str],
    ):
        self._router = router
        self._handlers = variant_handlers
        self._default_variant = default_variant
        self._fallback_pool = list(fallback_pool or [])

    def is_ready(self) -> bool:
        return bool(self._router and self._handlers.get(self._default_variant))

    def recommend(self, user_id: int, k: int) -> Tuple[ExperimentAssignment, List[str]]:
        assignment = self._router.assign(user_id)
        handler = self._handlers.get(assignment.variant)
        if handler is None:
            handler = self._handlers[self._default_variant]
            assignment = ExperimentAssignment(
                experiment_id=assignment.experiment_id,
                variant=self._default_variant,
                bucket=assignment.bucket,
            )

        recs = handler(user_id=user_id, k=k) or []
        if not recs and self._fallback_pool:
            print(f"User {user_id} routed to {assignment.variant} but no recs. Serving fallback.")
            recs = random.sample(self._fallback_pool, min(k, len(self._fallback_pool)))
        return assignment, recs


def load_popular_movies_pool() -> List[str]:
    """Loads the popularity fallback pool used for cold-start users."""
    print("➡️ Loading fallback data for cold-start users...")
    try:
        popularity_df = pd.read_parquet(POPULARITY_FILE)
        popular_movies_pool = popularity_df.head(200).index.tolist()
        print(f"Loaded {len(popular_movies_pool)} popular movies for fallback.")
    except FileNotFoundError:
        popular_movies_pool = []

    # --- Initialize Models and Service ---
    print("➡️ Initializing recommender models...")
    return popular_movies_pool


def build_experiment_gateway(popular_movies_pool: List[str]) -> tuple[ExperimentGateway, 'ALSRecommender']:
    """Wire up recommenders and router for experiment-aware serving.
    
    Returns:
        tuple: (ExperimentGateway, ALSRecommender) - gateway and the ALS recommender instance
    """
    router = load_router(EXPERIMENT_CONFIG_FILE)
    als_recommender = ALSRecommender.load(artifacts_dir=ALS_MODELS_DIR)
    popgen_recommender = PopGenreRecommender()

    als_service = RecommendationService(
        recommender=als_recommender,
        fallback_pool=popular_movies_pool,
    )

    variant_handlers: Dict[str, Callable[[int, int], List[str]]] = {
        "als_model": lambda user_id, k: als_service.get_recommendations(user_id, k),
        "popgen_model": lambda user_id, k: popgen_recommender.recommend(user_id=user_id, k=k),
    }

    gateway = ExperimentGateway(
        router=router,
        variant_handlers=variant_handlers,
        default_variant="als_model",
        fallback_pool=popular_movies_pool,
    )
    
    return gateway, als_recommender


def log_recommendation_event(
    user_id: int,
    assignment: ExperimentAssignment,
    recommendations: List[str],
    latency_ms: float,
) -> None:
    """Emit structured log entries for downstream monitoring."""
    recs_str = ",".join(str(mid) for mid in recommendations)
    provenance_suffix = ""
    if PROVENANCE_TRACKER and PROVENANCE_TRACKER.available():
        prov = PROVENANCE_TRACKER.extra_log_suffix()
        if prov:
            provenance_suffix = f", {prov}"

    log_line = (
        f"{datetime.utcnow().isoformat()}Z,"
        f"{user_id},"
        f"recommendation request /recommend/{user_id}, status 200, "
        f"variant={assignment.variant}, "
        f"bucket={assignment.bucket:.6f}, "
        f"result: {recs_str}, {int(latency_ms)} ms{provenance_suffix}"
    )
    LOGGER.info(log_line)
    REQUEST_COUNTER.labels(assignment.variant).inc()
    LATENCY_HISTOGRAM.labels(assignment.variant).observe(latency_ms)


def create_app():
    """
    Factory function to create the Flask application and initialize services.
    """
    app = Flask(__name__)

    # --- Initialize Prometheus Metrics ---
    # This automatically creates the /metrics endpoint and tracks:
    # - flask_http_request_total
    # - flask_http_request_duration_seconds
    # - flask_http_request_exceptions_total
    # - Our custom metrics: experiment_reco_requests_total, experiment_reco_latency_ms
    metrics = PrometheusMetrics(app, registry=METRICS_REGISTRY)

    # --- Load Fallback Data ---
    popular_movies_pool = load_popular_movies_pool()

    # --- Initialize Experiment Gateway ---
    print("➡️ Initializing experiment router and model variants...")
    try:
        experiment_gateway, als_recommender = build_experiment_gateway(popular_movies_pool)
        print("🚀 Experiment-aware recommender service is ready.")
        global PROVENANCE_TRACKER
        try:
            PROVENANCE_TRACKER = ProvenanceTracker.load(ALS_MODELS_DIR)
            if PROVENANCE_TRACKER.available():
                print(
                    "ℹ️ Loaded provenance manifest "
                    f"({PROVENANCE_TRACKER.manifest_path()})."
                )
            else:
                print("⚠️ No provenance manifest found in model directory.")
        except Exception as prov_exc:
            PROVENANCE_TRACKER = None
            print(f"⚠️ Failed to load provenance manifest: {prov_exc}")
    except Exception as exc:
        print(f"❌ Failed to initialize experiment gateway: {exc}")
        experiment_gateway = None
        als_recommender = None

    # Initialize custom metrics with the same registry as Flask metrics
    init_metrics(registry=METRICS_REGISTRY)

    # Load metrics from model artifacts (once at startup)
    # Note: eval_results.json and drift_report.json are in /app/models, not /app/artifacts
    load_metrics_from_artifacts(model_dir=ALS_MODELS_DIR)

    # --- Define Health Check Endpoints ---
    @app.route("/health/live", methods=["GET"])
    def liveness():
        """Liveness probe - is the app process running?"""
        return jsonify({"status": "alive"}), 200

    @app.route("/health/ready", methods=["GET"])
    def readiness():
        """Readiness probe - is the app ready to serve traffic?"""
        if not experiment_gateway or not experiment_gateway.is_ready():
            return (
                jsonify({"status": "not ready", "reason": "experiment gateway unavailable"}),
                503,
            )
        
        # Verify artifacts directory exists
        if not os.path.exists(ARTIFACTS_DIR):
            return jsonify({
                "status": "not ready",
                "reason": "artifacts directory not accessible"
            }), 503
        
        return jsonify({"status": "ready"}), 200

    @app.route("/health", methods=["GET"])
    def health_endpoint():
        """Enhanced health check endpoint with system verification."""
        build_number = os.environ.get("BUILD_NUMBER", "dev")
        git_commit = os.environ.get("GIT_COMMIT", "unknown")
        
        health_status = {
            "status": "healthy",
            "version": "2",
            "git_commit": git_commit[:7] if git_commit != "unknown" else git_commit,
            "build_number": build_number,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Verify experiment routing is available
            if not experiment_gateway or not experiment_gateway.is_ready():
                health_status["status"] = "unhealthy"
                health_status["error"] = "Experiment gateway unavailable"
                return jsonify(health_status), 503
            
            # Verify artifacts directory is accessible
            if not os.path.exists(ARTIFACTS_DIR):
                health_status["status"] = "degraded"
                health_status["warning"] = "Artifacts directory not accessible"
            
            return jsonify(health_status), 200
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
            return jsonify(health_status), 503

    @app.route("/recommend/<int:user_id>", methods=["GET"])
    def recommend_endpoint(user_id):
        """Flask endpoint to get recommendations for a user."""
        print(f"Received recommendation request for user_id: {user_id}")

        if not experiment_gateway:
            return (
                jsonify({"error": "Experiment gateway not initialized"}),
                503,
            )

        start = time.perf_counter()
        assignment, recs = experiment_gateway.recommend(user_id, k=20)
        latency_ms = (time.perf_counter() - start) * 1000
        
        if not recs:
            # Respond with a plain text error and a 404 status code
            return f"Could not generate recommendations for User ID {user_id}.", 404, {"Content-Type": "text/plain"}

        # Format the response as a comma-separated string
        recs_str = ",".join(str(mid) for mid in recs)
        log_recommendation_event(user_id, assignment, recs, latency_ms)

        headers = {
            "Content-Type": "text/plain",
            "X-Experiment-Id": assignment.experiment_id,
            "X-Experiment-Variant": assignment.variant,
        }
        if PROVENANCE_TRACKER and PROVENANCE_TRACKER.available():
            headers.update(
                {
                    key: value
                    for key, value in PROVENANCE_TRACKER.response_headers().items()
                    if value
                }
            )
        return recs_str, 200, headers

    @app.route("/metrics", methods=["GET"])
    def metrics_endpoint():
        """Expose Prometheus metrics for experiment monitoring."""
        return Response(generate_latest(METRICS_REGISTRY), mimetype=CONTENT_TYPE_LATEST)

    return app

# --- Main Execution Block ---
if __name__ == "__main__":
    flask_app = create_app()
    # To run: `python -m src.serve_als_model` from the project root
    flask_app.run(host="0.0.0.0", port=8082)
