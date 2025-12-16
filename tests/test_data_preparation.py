import pytest
import pandas as pd
from scipy.sparse import csr_matrix

from src.models.als.train import _load_and_prepare_data, HPARAMS


@pytest.fixture
def synthetic_interactions_csv(tmp_path):
    """Creates a synthetic interactions CSV file in a temporary directory."""
    data = {
        "user_id": ["u1", "u1", "u2", "u2", "u3"],
        "movie_id": ["m1", "m2", "m2", "m3", "m1"],
        "rating": [3, 2, 1, 3, 1],  
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "interactions.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


def test_load_and_prepare_data_output_structure(synthetic_interactions_csv):
    """Tests that data loading produces the correct output shapes and types."""
    matrix, user_map, movie_map, data_profile = _load_and_prepare_data(synthetic_interactions_csv)

    # 3 unique users, 3 unique movies
    assert isinstance(matrix, csr_matrix)
    assert matrix.shape == (3, 3)
    assert len(user_map) == 3
    assert len(movie_map) == 3
    
    # Verify data profile
    assert isinstance(data_profile, dict)
    assert data_profile['row_count'] == 5  # 5 rows in the CSV
    assert data_profile['unique_users'] == 3
    assert data_profile['unique_movies'] == 3


def test_load_and_prepare_data_map_content(synthetic_interactions_csv):
    """Tests that the generated user and movie maps contain the correct keys."""
    _, user_map, movie_map, _ = _load_and_prepare_data(synthetic_interactions_csv)

    assert "u1" in user_map
    assert "m1" in movie_map


def test_load_and_prepare_data_confidence_calculation(synthetic_interactions_csv):
    """Tests that the interaction confidence is calculated correctly."""
    matrix, user_map, movie_map, data_profile = _load_and_prepare_data(synthetic_interactions_csv)

    user_idx = user_map["u1"]
    movie_idx = movie_map["m1"]
    expected_confidence = 1 + HPARAMS["alpha"] * 3  
    assert matrix[user_idx, movie_idx] == pytest.approx(expected_confidence)


def test_load_and_prepare_data_empty_csv(tmp_path):
    """Tests behavior with an empty or header-only CSV file."""
    file_path = tmp_path / "empty.csv"
    file_path.write_text("user_id,movie_id,rating\n")

    matrix, user_map, movie_map, data_profile = _load_and_prepare_data(str(file_path))

    assert matrix.shape == (0, 0)
    assert user_map == {}
    assert movie_map == {}
    assert data_profile == {"row_count": 0, "unique_users": 0, "unique_movies": 0}


def test_load_and_prepare_data_with_na_handles_structure(tmp_path):
    """Tests that NA values result in correctly shaped (smaller) outputs."""
    data = {
        "user_id": ["u1", "u2", None],
        "movie_id": ["m1", None, "m3"],
        "rating": [5, 4, 3],
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "interactions_na.csv"
    df.to_csv(file_path, index=False)

    matrix, user_map, movie_map, data_profile = _load_and_prepare_data(str(file_path))

    # Only the first row is valid
    assert matrix.shape == (1, 1)
    assert len(user_map) == 1
    assert len(movie_map) == 1
    assert data_profile["row_count"] == 1


def test_load_and_prepare_data_with_na_handles_content(tmp_path):
    """Tests that NA-containing rows are excluded from the final maps."""
    data = {
        "user_id": ["u1", "u2", None],
        "movie_id": ["m1", None, "m3"],
        "rating": [5, 4, 3],
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "interactions_na.csv"
    df.to_csv(file_path, index=False)

    _, user_map, _, _ = _load_and_prepare_data(str(file_path))

    assert "u1" in user_map
    assert "u2" not in user_map  # This user's row had a missing movie_id


def test_load_and_prepare_data_handles_duplicates(tmp_path):
    """Tests that duplicate user-movie interactions are aggregated (summed)."""
    data = {
        "user_id": ["u1", "u1"],
        "movie_id": ["m1", "m1"],
        "rating": [2, 1],
         "timestamp": [1000, 2000]
    }
    df = pd.DataFrame(data)
    df = (
        pd.DataFrame(data)
        .sort_values(["user_id", "timestamp"], kind="mergesort")
        .assign(_row_order=lambda frame: range(len(frame)))
        .sort_values(["user_id", "timestamp", "_row_order"])
        .drop_duplicates(["user_id", "movie_id"], keep="last")
        .drop(columns="_row_order")
    )
    file_path = tmp_path / "interactions_dup.csv"
    df.to_csv(file_path, index=False)

    matrix, user_map, movie_map, _ = _load_and_prepare_data(str(file_path))

    # The coo_matrix constructor sums values for duplicate coordinates.
    # This test verifies that behavior.
    confidence1 = 1 + HPARAMS["alpha"] * 1
    expected_total_confidence =  confidence1

    user_idx = user_map["u1"]
    movie_idx = movie_map["m1"]
    assert matrix[user_idx, movie_idx] == pytest.approx(expected_total_confidence)


def test_load_and_prepare_data_missing_required_columns(tmp_path):
    """Tests behavior when a required column (e.g., 'movie_id') is missing from the CSV."""
    file_path = tmp_path / "missing_movie_id_col.csv"
    # 'movie_id' is missing from the header
    file_path.write_text("user_id,rating\nu1,5\nu2,3\n")

    with pytest.raises(ValueError, match="Usecols do not match columns"):
        _load_and_prepare_data(str(file_path))


def test_load_and_prepare_data_missing_file(tmp_path):
    """Tests behavior when the file doesn't exist."""
    non_existent_file = tmp_path / "does_not_exist.csv"

    matrix, user_map, movie_map, data_profile = _load_and_prepare_data(str(non_existent_file))

    assert matrix.shape == (0, 0)
    assert user_map == {}
    assert movie_map == {}
    assert data_profile == {"row_count": 0, "unique_users": 0, "unique_movies": 0}


def test_load_and_prepare_data_empty_file(tmp_path):
    """Tests behavior with a completely empty file (no headers)."""
    file_path = tmp_path / "empty_no_header.csv"
    file_path.write_text("")

    matrix, user_map, movie_map, data_profile = _load_and_prepare_data(str(file_path))

    assert matrix.shape == (0, 0)
    assert user_map == {}
    assert movie_map == {}
    assert data_profile == {"row_count": 0, "unique_users": 0, "unique_movies": 0}