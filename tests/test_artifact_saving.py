import json
import numpy as np

from src.models.als.train import _save_artifacts


def test_save_artifacts_creates_files(tmp_path):
    """Tests that artifact saving creates the expected files."""
    class MockALSModel:
        def __init__(self):
            self.user_factors = np.random.rand(3, 2).astype(np.float32)
            self.item_factors = np.random.rand(4, 2).astype(np.float32)

    mock_model = MockALSModel()
    user_map = {"u1": 0, "u2": 1, "u3": 2}
    movie_map = {"m1": 0, "m2": 1, "m3": 2, "m4": 3}
    artifact_dir = tmp_path / "artifacts"

    # 2. Save artifacts
    _save_artifacts(mock_model, user_map, movie_map, str(artifact_dir))

    # 3. Assert files were created
    model_path = artifact_dir / "als_model.npz"
    user_map_path = artifact_dir / "user_map.json"
    movie_map_path = artifact_dir / "movie_map.json"

    assert model_path.exists()
    assert user_map_path.exists()
    assert movie_map_path.exists()


def test_save_artifacts_writes_correct_content(tmp_path):
    """Tests that saved artifacts contain the correct data."""
    class MockALSModel:
        def __init__(self):
            self.user_factors = np.random.rand(3, 2).astype(np.float32)
            self.item_factors = np.random.rand(4, 2).astype(np.float32)

    mock_model = MockALSModel()
    user_map = {"u1": 0, "u2": 1, "u3": 2}
    movie_map = {"m1": 0, "m2": 1, "m3": 2, "m4": 3}
    artifact_dir = tmp_path / "artifacts"

    _save_artifacts(mock_model, user_map, movie_map, str(artifact_dir))
    model_path = artifact_dir / "als_model.npz"
    user_map_path = artifact_dir / "user_map.json"
    movie_map_path = artifact_dir / "movie_map.json"

    loaded_model = np.load(model_path)
    np.testing.assert_array_almost_equal(
        loaded_model["user_factors"], mock_model.user_factors
    )
    np.testing.assert_array_almost_equal(
        loaded_model["item_factors"], mock_model.item_factors
    )

    with open(user_map_path, "r") as f:
        loaded_user_map = json.load(f)
    assert loaded_user_map == user_map

    with open(movie_map_path, "r") as f:
        loaded_movie_map = json.load(f)
    assert loaded_movie_map == movie_map


def test_save_artifacts_with_empty_data(tmp_path):
    """Tests saving artifacts when maps and factors are empty."""
    class MockALSModelEmpty:
        def __init__(self):
            self.user_factors = np.empty((0, 2), dtype=np.float32)
            self.item_factors = np.empty((0, 2), dtype=np.float32)

    mock_model_empty = MockALSModelEmpty()
    empty_user_map = {}
    empty_movie_map = {}
    artifact_dir = tmp_path / "empty_artifacts"

    _save_artifacts(mock_model_empty, empty_user_map, empty_movie_map, str(artifact_dir))

    model_path = artifact_dir / "als_model.npz"
    user_map_path = artifact_dir / "user_map.json"
    movie_map_path = artifact_dir / "movie_map.json"

    assert model_path.exists()
    assert user_map_path.exists()
    assert movie_map_path.exists()

    loaded_model = np.load(model_path)
    np.testing.assert_array_equal(loaded_model["user_factors"], mock_model_empty.user_factors)
    np.testing.assert_array_equal(loaded_model["item_factors"], mock_model_empty.item_factors)

    with open(user_map_path, "r") as f:
        loaded_user_map = json.load(f)
    assert loaded_user_map == empty_user_map

    with open(movie_map_path, "r") as f:
        loaded_movie_map = json.load(f)
    assert loaded_movie_map == empty_movie_map