import pytest
import pandas as pd
from unittest.mock import MagicMock

from src.models.als.evaluate import (
    sample_holdouts,
    evaluate_hit_at_20,
    measure_latency_throughput,
    model_size_bytes,
)
from src.models.als.model import ALSRecommender


@pytest.fixture
def dummy_interactions_csv(tmp_path):
    """Creates a dummy interactions CSV for testing holdout sampling."""
    file_path = tmp_path / "interactions.csv"
    # User 1's last interaction is m3, User 2's is m1
    data = {
        "user_id": [1, 1, 2, 3, 4, 5, 6],
        "movie_id": ["m1", "m3", "m1", "m2", "m3", "m4", "m1"],
        "rating": [5, 4, 3, 2, 1, 2, 3],
    }
    pd.DataFrame(data).to_csv(file_path, index=False)
    return str(file_path)


def test_sample_holdouts(dummy_interactions_csv):
    """Tests that holdout sampling correctly identifies the last interaction for each user."""
    pairs = sample_holdouts(dummy_interactions_csv, n_users=2)

    assert len(pairs) == 2 
    df = pd.read_csv(dummy_interactions_csv)
    user_id, movie_id = pairs[0]
    user_movies = df[df['user_id'] == user_id]['movie_id']
    last_movie = user_movies.iloc[-1]  
    assert last_movie == movie_id


def test_evaluate_hit_at_20():
    """Tests the Hit@20 calculation logic."""
    mock_reco = MagicMock(spec=ALSRecommender)

    def mock_recommend(user_id, k):
        if user_id == 1:
            return [f"movie_{i}" for i in range(15)] + ["m3"]
        if user_id == 2:
            return [f"movie_{i}" for i in range(20)]
        return []

    mock_reco.recommend.side_effect = mock_recommend

    test_pairs = [(1, "m3"), (2, "m4")]

    hit_rate = evaluate_hit_at_20(mock_reco, test_pairs)
    assert hit_rate == 0.5  


def test_measure_latency_throughput(mocker):
    """Tests latency and throughput calculations using a mocked time.perf_counter."""
    # 1. Mock the recommender and time
    mock_reco = MagicMock(spec=ALSRecommender)
    # Mock perf_counter to return predictable, increasing timestamps.
    # The return value of patch() is not needed here.
    mocker.patch("time.perf_counter", side_effect=[
        0.0,      # Start of loop
        0.0,      # Start of user 1 rec
        0.01,     # End of user 1 rec (10ms)
        0.01,     # Start of user 2 rec
        0.04,     # End of user 2 rec (30ms)
        0.04      # End of loop
    ])

    users = [1, 2]
    avg_ms, p95_ms, qps = measure_latency_throughput(mock_reco, users)

    assert avg_ms == pytest.approx(20.0)
    assert p95_ms == pytest.approx(29.0)
    assert qps == pytest.approx(50.0)


def test_model_size_bytes(tmp_path):
    """Tests that artifact size calculation is correct."""
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()

    # Create dummy files with known content/size
    (art_dir / "als_model.npz").write_text("a" * 100)
    (art_dir / "user_map.json").write_text("b" * 50)
    # Leave movie_map.json missing to test robustness

    size = model_size_bytes(str(art_dir))

    assert size == 150  


def test_sample_holdouts_with_fewer_users_than_requested(tmp_path):
    """Tests holdout sampling when available users < n_users."""
    file_path = tmp_path / "small_interactions.csv"
    # Only 2 users
    data = {
        "user_id": [1, 2],
        "movie_id": ["m1", "m2"],
        "rating": [5, 4],
    }
    pd.DataFrame(data).to_csv(file_path, index=False)

    pairs = sample_holdouts(str(file_path), n_users=100)

    assert len(pairs) <= 2


def test_sample_holdouts_with_empty_file(tmp_path):
    """Tests holdout sampling with an empty interactions file."""
    file_path = tmp_path / "empty.csv"
    file_path.write_text("user_id,movie_id\n")

    pairs = sample_holdouts(str(file_path), n_users=10)

    assert pairs == []


def test_sample_holdouts_single_interaction_per_user(tmp_path):
    """Tests that users with only one interaction have that as their holdout."""
    file_path = tmp_path / "single_interactions.csv"
    # Each user has exactly one interaction
    data = {
        "user_id": [1, 2, 3],
        "movie_id": ["m1", "m2", "m3"],
        "rating": [5, 4, 3],
    }
    pd.DataFrame(data).to_csv(file_path, index=False)

    pairs = sample_holdouts(str(file_path), n_users=3)

    # Each user's only interaction should be their holdout
    assert len(pairs) == 3
    holdouts = dict(pairs)
    assert holdouts[1] == "m1"
    assert holdouts[2] == "m2"
    assert holdouts[3] == "m3"


def test_evaluate_hit_at_20_all_hits():
    """Tests Hit@20 when all targets are found."""
    mock_reco = MagicMock(spec=ALSRecommender)

    def mock_recommend(user_id, k):
        return [f"movie_{i}" for i in range(15)] + [f"target_{user_id}"]

    mock_reco.recommend.side_effect = mock_recommend

    test_pairs = [(1, "target_1"), (2, "target_2"), (3, "target_3")]

    hit_rate = evaluate_hit_at_20(mock_reco, test_pairs)
    assert hit_rate == 1.0  # All hits


def test_evaluate_hit_at_20_no_hits():
    """Tests Hit@20 when no targets are found."""
    mock_reco = MagicMock(spec=ALSRecommender)

    def mock_recommend(user_id, k):
        return [f"movie_{i}" for i in range(20)]

    mock_reco.recommend.side_effect = mock_recommend

    test_pairs = [(1, "target_1"), (2, "target_2")]

    hit_rate = evaluate_hit_at_20(mock_reco, test_pairs)
    assert hit_rate == 0.0 


def test_evaluate_hit_at_20_empty_pairs():
    """Tests Hit@20 with empty pairs list."""
    mock_reco = MagicMock(spec=ALSRecommender)

    hit_rate = evaluate_hit_at_20(mock_reco, [])

    assert hit_rate == 0.0


def test_measure_latency_throughput_empty_users(mocker):
    """Tests latency measurement with empty user list."""
    mock_reco = MagicMock(spec=ALSRecommender)

    avg_ms, p95_ms, qps = measure_latency_throughput(mock_reco, [])

    assert avg_ms == 0.0
    assert p95_ms == 0.0
    assert qps == 0.0


def test_measure_latency_throughput_with_large_user_list(mocker):
    """Tests that latency measurement samples down to TIMING_USERS."""
    from src.models.als.evaluate import TIMING_USERS

    mock_reco = MagicMock(spec=ALSRecommender)
    mock_reco.recommend.return_value = []

    # Create a list larger than TIMING_USERS
    large_user_list = list(range(TIMING_USERS + 500))

    # Mock time to avoid actual delays
    mocker.patch("time.perf_counter", side_effect=[0.0] + [0.01 * i for i in range(TIMING_USERS * 2 + 10)])

    avg_ms, p95_ms, qps = measure_latency_throughput(mock_reco, large_user_list)

    # Should only call recommend TIMING_USERS times, not more
    assert mock_reco.recommend.call_count == TIMING_USERS


def test_model_size_bytes_all_files_missing(tmp_path):
    """Tests model size calculation when all artifact files are missing."""
    art_dir = tmp_path / "empty_artifacts"
    art_dir.mkdir()

    size = model_size_bytes(str(art_dir))

    assert size == 0
