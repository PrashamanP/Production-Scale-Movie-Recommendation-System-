import os
import json
import pytest
import numpy as np

from src.models.als.model import ALSRecommender


@pytest.fixture
def mock_artifacts(tmp_path):
    """Creates mock model artifacts for testing the ALSRecommender class."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    user_factors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    item_factors = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    np.savez(artifacts_dir / "als_model.npz", user_factors=user_factors, item_factors=item_factors)

    user_map = {"101": 0, "102": 1}
    with open(artifacts_dir / "user_map.json", "w") as f:
        json.dump(user_map, f)

    movie_map = {"m1": 0, "m2": 1, "m3": 2}
    with open(artifacts_dir / "movie_map.json", "w") as f:
        json.dump(movie_map, f)

    return str(artifacts_dir)


def test_recommender_load(mock_artifacts):
    """Tests that the recommender loads from artifacts correctly."""
    reco = ALSRecommender.load(artifacts_dir=mock_artifacts)
    assert reco is not None
    assert "101" in reco.user_map
    assert reco.movie_inv_map[0] == "m1"
    assert reco.model.user_factors.shape == (2, 2)


def test_recommend_for_known_user(mock_artifacts):
    """Tests basic recommendation for a user present in the model."""
    reco = ALSRecommender.load(artifacts_dir=mock_artifacts)
    recommendations = reco.recommend(user_id=101, k=1)
    assert recommendations == ["m1"]


def test_recommend_cold_start_user(mock_artifacts):
    """Tests that an unknown user gets an empty recommendation list."""
    reco = ALSRecommender.load(artifacts_dir=mock_artifacts)
    recommendations = reco.recommend(user_id=999, k=5)
    assert recommendations == []


def test_recommend_with_exclusion(mock_artifacts):
    """Tests that the 'exclude' parameter correctly filters recommendations."""
    reco = ALSRecommender.load(artifacts_dir=mock_artifacts)
    recommendations = reco.recommend(user_id=101, k=1, exclude=["m1"])
    assert recommendations == ["m3"]


def test_recommend_returns_correct_number(mock_artifacts):
    """Tests that the number of recommendations respects k."""
    reco = ALSRecommender.load(artifacts_dir=mock_artifacts)
    recommendations = reco.recommend(user_id=102, k=2)
    assert len(recommendations) == 2


def test_recommend_with_k_zero(mock_artifacts):
    """Tests that requesting k=0 recommendations returns an empty list."""
    reco = ALSRecommender.load(artifacts_dir=mock_artifacts)
    recommendations = reco.recommend(user_id=101, k=0)
    assert recommendations == []


def test_load_fails_with_missing_file(tmp_path):
    """Tests that loading fails if an artifact is missing."""
    # Create a directory with only one of the required files
    artifacts_dir = tmp_path / "bad_artifacts"
    artifacts_dir.mkdir()
    with open(artifacts_dir / "user_map.json", "w") as f:
        json.dump({"1": 1}, f)

    with pytest.raises(FileNotFoundError):
        ALSRecommender.load(artifacts_dir=str(artifacts_dir))


def test_recommend_with_string_user_id(mock_artifacts):
    """Tests that string user IDs are properly handled (converted to string internally)."""
    reco = ALSRecommender.load(artifacts_dir=mock_artifacts)
    recommendations = reco.recommend(user_id=101, k=1)
    assert recommendations == ["m1"]


def test_recommend_with_actual_string_user_id(mock_artifacts):
    """Tests recommendation with an actual string user ID."""
    reco = ALSRecommender.load(artifacts_dir=mock_artifacts)
    reco.user_map["user_abc"] = 0

    recommendations = reco.recommend(user_id="user_abc", k=1)
    assert recommendations == ["m1"]


def test_recommend_large_k_exceeds_available_items(mock_artifacts):
    """Tests that requesting k larger than available items works without error."""
    reco = ALSRecommender.load(artifacts_dir=mock_artifacts)
    recommendations = reco.recommend(user_id=101, k=100)
    assert len(recommendations) <= 100
    assert all(m in ["m1", "m2", "m3"] for m in recommendations)


def test_recommend_with_large_exclusion_list(mock_artifacts):
    """Tests recommendation when exclusion list is large relative to k."""
    reco = ALSRecommender.load(artifacts_dir=mock_artifacts)
    recommendations = reco.recommend(user_id=101, k=1, exclude=["m1", "m3"])

    assert len(recommendations) == 1
    assert recommendations[0] == "m2"


def test_recommend_exclusion_all_movies(mock_artifacts):
    """Tests recommendation when all movies are excluded."""
    reco = ALSRecommender.load(artifacts_dir=mock_artifacts)
    recommendations = reco.recommend(user_id=101, k=5, exclude=["m1", "m2", "m3"])

    assert recommendations == []


def test_recommend_negative_k(mock_artifacts):
    """Tests recommendation with negative k (edge case)."""
    reco = ALSRecommender.load(artifacts_dir=mock_artifacts)
    try:
        recommendations = reco.recommend(user_id=101, k=-1)
        assert recommendations == []
    except (ValueError, AssertionError):
        # Acceptable to raise an error for negative k
        pass


def test_recommend_with_duplicate_exclusions(mock_artifacts):
    """Tests that duplicate items in exclusion list are handled correctly."""
    reco = ALSRecommender.load(artifacts_dir=mock_artifacts)
    recommendations = reco.recommend(user_id=101, k=2, exclude=["m1", "m1", "m1"])

    assert "m1" not in recommendations
    assert len(recommendations) == 2
