import os
import random
from typing import List

import pandas as pd
from flask import Flask
from src.models.als.model import ALSRecommender

# --- Artifact paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")
POPULARITY_FILE = os.path.join(ARTIFACTS_DIR, "popgen_popularity.parquet")


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


def create_app():
    """
    Factory function to create the Flask application and initialize services.
    """
    app = Flask(__name__)

    # --- Load Fallback Data ---
    print("➡️ Loading fallback data for cold-start users...")
    try:
        popularity_df = pd.read_parquet(POPULARITY_FILE)
        popular_movies_pool = popularity_df.head(200).index.tolist()
        print(f"Loaded {len(popular_movies_pool)} popular movies for fallback.")
    except FileNotFoundError:
        popular_movies_pool = []
    
    # --- Initialize Models and Service ---
    print("➡️ Initializing recommender models...")
    als_recommender = ALSRecommender.load(artifacts_dir=ARTIFACTS_DIR)
    reco_service = RecommendationService(
        recommender=als_recommender,
        fallback_pool=popular_movies_pool
    )
    print("🚀 Recommender service is ready.")

    # --- Define API Endpoints ---
    @app.route("/recommend/<int:user_id>", methods=["GET"])
    def recommend_endpoint(user_id):
        """Flask endpoint to get recommendations for a user."""
        print(f"Received recommendation request for user_id: {user_id}")
        
        recs = reco_service.get_recommendations(user_id, k=20)
        
        if not recs:
            # Respond with a plain text error and a 404 status code
            return f"Could not generate recommendations for User ID {user_id}.", 404, {"Content-Type": "text/plain"}

        # **CORRECTED LINE:** Format the response as a comma-separated string
        recs_str = ",".join(str(mid) for mid in recs)
        
        return recs_str, 200, {"Content-Type": "text/plain"}

    return app

# --- Main Execution Block ---
if __name__ == "__main__":
    flask_app = create_app()
    # To run: `python -m src.serve_als_model` from the project root
    flask_app.run(host="0.0.0.0", port=8082)