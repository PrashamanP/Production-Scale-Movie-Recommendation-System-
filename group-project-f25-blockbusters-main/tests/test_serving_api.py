import pytest
import json
from unittest.mock import MagicMock

import pandas as pd
from src.serve_als_model import create_app, RecommendationService, ARTIFACTS_DIR, POPULARITY_FILE
from src.models.als.model import ALSRecommender # Needed for mocking ALSRecommender.load

@pytest.fixture
def client(mocker):
    """Configures the Flask test client and mocks dependencies."""
    mock_als_recommender = mocker.MagicMock(spec=ALSRecommender)
    mocker.patch('src.models.als.model.ALSRecommender.load', return_value=mock_als_recommender)

    mock_popularity_df = pd.DataFrame({'score': [10, 9, 8]}, index=['pop_m1', 'pop_m2', 'pop_m3'])
    mocker.patch('pandas.read_parquet', return_value=mock_popularity_df)

    mock_reco_service = mocker.MagicMock(spec=RecommendationService)
    mocker.patch('src.serve_als_model.RecommendationService', return_value=mock_reco_service)

    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client, mock_reco_service

def test_recommend_endpoint_success(client):
    """Tests that the /recommend endpoint returns successful recommendations."""
    test_client, mock_reco_service = client
    mock_reco_service.get_recommendations.return_value = ["movie_a", "movie_b"]

    response = test_client.get("/recommend/123")

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/plain"
    assert response.data.decode("utf-8") == "movie_a,movie_b"
    mock_reco_service.get_recommendations.assert_called_once_with(123, k=20)

def test_recommend_endpoint_cold_start_fallback(client):
    """Tests that the /recommend endpoint handles cold-start users with fallback."""
    test_client, mock_reco_service = client
    mock_reco_service.get_recommendations.return_value = ["pop_m1", "pop_m2"]

    response = test_client.get("/recommend/999")

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/plain"
    assert response.data.decode("utf-8") == "pop_m1,pop_m2"
    mock_reco_service.get_recommendations.assert_called_once_with(999, k=20)

def test_recommend_endpoint_no_recommendations(client):
    """Tests that the /recommend endpoint returns 404 if no recommendations are found."""
    test_client, mock_reco_service = client
    mock_reco_service.get_recommendations.return_value = []

    response = test_client.get("/recommend/123")

    assert response.status_code == 404
    assert response.headers["Content-Type"] == "text/plain"
    assert "Could not generate recommendations" in response.data.decode("utf-8")
    mock_reco_service.get_recommendations.assert_called_once_with(123, k=20)

def test_create_app_loads_artifacts_and_fallback(mocker):
    """Tests that create_app correctly loads ALS recommender and fallback data."""
    mock_als_load = mocker.patch('src.models.als.model.ALSRecommender.load')
    mock_pd_read_parquet = mocker.patch('pandas.read_parquet')
    mock_reco_service_init = mocker.patch('src.serve_als_model.RecommendationService.__init__', return_value=None) # Mock __init__ to avoid actual init logic

    mock_popularity_df = pd.DataFrame({'score': [10, 9]}, index=['pop_m1', 'pop_m2'])
    mock_pd_read_parquet.return_value = mock_popularity_df

    app = create_app()

    mock_pd_read_parquet.assert_called_once_with(POPULARITY_FILE)
    mock_als_load.assert_called_once_with(artifacts_dir=ARTIFACTS_DIR)
    mock_reco_service_init.assert_called_once()
    args, kwargs = mock_reco_service_init.call_args
    assert kwargs['fallback_pool'] == ['pop_m1', 'pop_m2']

def test_create_app_handles_missing_popularity_file(mocker):
    """Tests that create_app handles the case where the popularity file is missing."""
    mocker.patch('src.models.als.model.ALSRecommender.load')
    mock_read_parquet = mocker.patch('pandas.read_parquet', side_effect=FileNotFoundError)
    mock_reco_service_init = mocker.patch('src.serve_als_model.RecommendationService.__init__', return_value=None)

    app = create_app()

    mock_read_parquet.assert_called_once_with(POPULARITY_FILE)
    mock_reco_service_init.assert_called_once()
    args, kwargs = mock_reco_service_init.call_args
    assert kwargs['fallback_pool'] == []

def test_recommend_endpoint_invalid_user_id_type(client):
    """Tests that the /recommend endpoint handles invalid user ID types gracefully."""
    test_client, _ = client
    response = test_client.get("/recommend/abc") 
    assert response.status_code == 404 
    assert "Not Found" in response.data.decode("utf-8")


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
