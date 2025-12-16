from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.experimentation.experiment_router import ExperimentAssignment
from src.models.als.model import ALSRecommender
from src.serve_als_model import POPULARITY_FILE, RecommendationService, create_app


@pytest.fixture
def client(mocker):
    """Configures the Flask test client and mocks dependencies."""
    mock_popularity_df = pd.DataFrame({"score": [10, 9, 8]}, index=["pop_m1", "pop_m2", "pop_m3"])
    mocker.patch("pandas.read_parquet", return_value=mock_popularity_df)

    mock_gateway = mocker.MagicMock()
    mock_assignment = ExperimentAssignment("als_vs_pop", "als_model", 0.12)
    mock_gateway.recommend.return_value = (mock_assignment, ["movie_a", "movie_b"])
    mock_gateway.is_ready.return_value = True
    
    # Mock ALS recommender for metrics collector
    mock_als_recommender = mocker.MagicMock()
    
    # Return tuple (gateway, als_recommender) to match updated function signature
    mocker.patch("src.serve_als_model.build_experiment_gateway", return_value=(mock_gateway, mock_als_recommender))

    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client, mock_gateway


def test_recommend_endpoint_success(client):
    """Tests that the /recommend endpoint returns successful recommendations."""
    test_client, mock_gateway = client
    assignment = ExperimentAssignment("als_vs_pop", "als_model", 0.42)
    mock_gateway.recommend.return_value = (assignment, ["movie_a", "movie_b"])

    response = test_client.get("/recommend/123")

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/plain"
    assert response.headers["X-Experiment-Variant"] == "als_model"
    assert response.data.decode("utf-8") == "movie_a,movie_b"
    mock_gateway.recommend.assert_called_once()


def test_recommend_endpoint_cold_start_fallback(client):
    """Tests that the /recommend endpoint handles cold-start users with fallback."""
    test_client, mock_gateway = client
    assignment = ExperimentAssignment("als_vs_pop", "popgen_model", 0.55)
    mock_gateway.recommend.return_value = (assignment, ["pop_m1", "pop_m2"])

    response = test_client.get("/recommend/999")

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/plain"
    assert response.headers["X-Experiment-Variant"] == "popgen_model"
    assert response.data.decode("utf-8") == "pop_m1,pop_m2"


def test_recommend_endpoint_no_recommendations(client):
    """Tests that the /recommend endpoint returns 404 if no recommendations are found."""
    test_client, mock_gateway = client
    assignment = ExperimentAssignment("als_vs_pop", "als_model", 0.33)
    mock_gateway.recommend.return_value = (assignment, [])

    response = test_client.get("/recommend/123")

    assert response.status_code == 404
    assert response.headers["Content-Type"] == "text/plain"
    assert "Could not generate recommendations" in response.data.decode("utf-8")
    mock_gateway.recommend.assert_called_once()


def test_create_app_loads_artifacts_and_fallback(mocker):
    """Tests that create_app correctly loads the fallback catalog."""
    mock_pd_read_parquet = mocker.patch("pandas.read_parquet")
    mock_gateway_builder = mocker.patch("src.serve_als_model.build_experiment_gateway")

    mock_popularity_df = pd.DataFrame({"score": [10, 9]}, index=["pop_m1", "pop_m2"])
    mock_pd_read_parquet.return_value = mock_popularity_df

    app = create_app()

    mock_pd_read_parquet.assert_called_once_with(POPULARITY_FILE)
    mock_gateway_builder.assert_called_once()
    args, _ = mock_gateway_builder.call_args
    assert args[0] == ["pop_m1", "pop_m2"]


def test_create_app_handles_missing_popularity_file(mocker):
    """Tests that create_app handles the case where the popularity file is missing."""
    mock_read_parquet = mocker.patch("pandas.read_parquet", side_effect=FileNotFoundError)
    mock_gateway_builder = mocker.patch("src.serve_als_model.build_experiment_gateway")

    app = create_app()

    mock_read_parquet.assert_called_once_with(POPULARITY_FILE)
    mock_gateway_builder.assert_called_once()
    args, _ = mock_gateway_builder.call_args
    assert args[0] == []


def test_recommend_endpoint_invalid_user_id_type(client):
    """Tests that the /recommend endpoint handles invalid user ID types gracefully."""
    test_client, _ = client
    response = test_client.get("/recommend/abc")
    assert response.status_code == 404
    assert "Not Found" in response.data.decode("utf-8")


def test_readiness_reflects_gateway_status(client):
    """Readiness endpoint should return 503 if the experiment gateway is not ready."""
    test_client, mock_gateway = client
    mock_gateway.is_ready.return_value = False

    response = test_client.get("/health/ready")

    assert response.status_code == 503
    assert response.json["status"] == "not ready"
    assert "experiment gateway" in response.json["reason"]


def test_metrics_endpoint_reports_counters(client):
    """Metrics endpoint should expose Prometheus-compatible metrics."""
    test_client, mock_gateway = client
    assignment = ExperimentAssignment("als_vs_pop", "als_model", 0.42)
    mock_gateway.recommend.return_value = (assignment, ["movie_a"])

    test_client.get("/recommend/42")
    response = test_client.get("/metrics")

    assert response.status_code == 200
    assert b"experiment_reco_requests_total" in response.data


def test_recommendation_service_fallback_pool_smaller_than_k():
    """Tests RecommendationService when fallback pool has fewer items than k."""
    mock_recommender = MagicMock(spec=ALSRecommender)
    mock_recommender.recommend.return_value = []

    small_fallback = ["m1", "m2", "m3"]
    service = RecommendationService(mock_recommender, small_fallback)

    recs = service.get_recommendations(user_id=999, k=20)

    assert len(recs) == 3
    assert set(recs) == set(small_fallback)


def test_recommendation_service_fallback_pool_empty():
    """Tests RecommendationService when fallback pool is empty."""
    mock_recommender = MagicMock(spec=ALSRecommender)
    mock_recommender.recommend.return_value = []

    empty_fallback = []
    service = RecommendationService(mock_recommender, empty_fallback)

    recs = service.get_recommendations(user_id=999, k=20)

    assert recs == []


def test_recommendation_service_uses_recommender_first():
    """Tests that RecommendationService uses the recommender before fallback."""
    mock_recommender = MagicMock(spec=ALSRecommender)
    mock_recommender.recommend.return_value = ["rec1", "rec2", "rec3"]

    fallback = ["fb1", "fb2"]
    service = RecommendationService(mock_recommender, fallback)

    recs = service.get_recommendations(user_id=123, k=3)

    assert recs == ["rec1", "rec2", "rec3"]
    mock_recommender.recommend.assert_called_once_with(user_id=123, k=3)


def test_recommendation_service_fallback_randomness(mocker):
    """Tests that fallback pool is sampled randomly."""
    mock_recommender = MagicMock(spec=ALSRecommender)
    mock_recommender.recommend.return_value = []  # Cold-start

    large_fallback = [f"m{i}" for i in range(100)]
    service = RecommendationService(mock_recommender, large_fallback)

    mock_sample = mocker.patch("random.sample", return_value=["m1", "m2", "m3"])

    recs = service.get_recommendations(user_id=999, k=3)

    mock_sample.assert_called_once_with(large_fallback, 3)
    assert recs == ["m1", "m2", "m3"]
