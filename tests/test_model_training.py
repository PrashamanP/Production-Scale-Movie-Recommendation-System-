import pandas as pd
from scipy.sparse import csr_matrix

from src.models.als.train import _train_model, HPARAMS, main


def test_train_model_output_type_and_shape():
    """Tests that the model trains and returns a fitted ALS object with correct shapes."""
    interaction_matrix = csr_matrix(
        ([4, 5, 1], ([0, 1, 2], [0, 1, 2])), shape=(3, 3)
    )

    model = _train_model(interaction_matrix)

    assert model.__class__.__name__ == 'AlternatingLeastSquares'
    assert hasattr(model, 'user_factors')
    assert hasattr(model, 'item_factors')
    assert model.user_factors.shape == (3, HPARAMS["factors"])
    assert model.item_factors.shape == (3, HPARAMS["factors"])


def test_train_model_empty_matrix():
    """Tests model training with an empty interaction matrix."""
    interaction_matrix = csr_matrix((0, 0))

    try:
        model = _train_model(interaction_matrix)
        assert model is not None
    except (ValueError, IndexError):
        assert True


def test_train_model_respects_factors_hyperparameter(monkeypatch):
    """Tests that the model's factor shape is controlled by HPARAMS."""
    monkeypatch.setitem(HPARAMS, "factors", 10)

    interaction_matrix = csr_matrix(
        ([4, 5, 1], ([0, 1, 2], [0, 1, 2])), shape=(3, 3)
    )

    model = _train_model(interaction_matrix)

    assert model.user_factors.shape == (3, 10)
    assert model.item_factors.shape == (3, 10)


def test_training_pipeline_integration_end_to_end(tmp_path, monkeypatch):
    """Integration test: full training pipeline from CSV to saved artifacts."""
    data = {
        "user_id": ["u1", "u1", "u2", "u3"],
        "movie_id": ["m1", "m2", "m1", "m3"],
        "rating": [5, 4, 3, 5],
    }
    df = pd.DataFrame(data)
    data_path = tmp_path / "interactions.csv"
    df.to_csv(data_path, index=False)

    artifact_dir = tmp_path / "artifacts"

    monkeypatch.setattr("sys.argv", [
        "train.py",
        "--data_path", str(data_path),
        "--artifact_dir", str(artifact_dir)
    ])

    main()

    assert (artifact_dir / "als_model.npz").exists()
    assert (artifact_dir / "user_map.json").exists()
    assert (artifact_dir / "movie_map.json").exists()
    manifest_path = artifact_dir / "model_manifest.json"
    assert manifest_path.exists()

    import json
    import numpy as np

    npz = np.load(artifact_dir / "als_model.npz")
    assert "user_factors" in npz
    assert "item_factors" in npz
    assert npz["user_factors"].shape[0] == 3  # 3 unique users
    assert npz["item_factors"].shape[0] == 3  # 3 unique movies
    assert npz["user_factors"].shape[1] == HPARAMS["factors"]

    # Check user map
    with open(artifact_dir / "user_map.json", "r") as f:
        user_map = json.load(f)
    assert len(user_map) == 3
    assert "u1" in user_map
    assert "u2" in user_map
    assert "u3" in user_map

    with open(artifact_dir / "movie_map.json", "r") as f:
        movie_map = json.load(f)
    assert len(movie_map) == 3
    assert "m1" in movie_map
    assert "m2" in movie_map
    assert "m3" in movie_map

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    assert manifest["model_family"] == "als"
    assert manifest["data"]["row_count"] == len(df)
    assert manifest["artifacts"]["als_model"]["path"]


def test_training_pipeline_with_empty_data(tmp_path, monkeypatch):
    """Integration test: training pipeline with empty data file."""
    data_path = tmp_path / "empty.csv"
    data_path.write_text("user_id,movie_id,rating\n")

    artifact_dir = tmp_path / "artifacts"

    # Mock sys.argv
    monkeypatch.setattr("sys.argv", [
        "train.py",
        "--data_path", str(data_path),
        "--artifact_dir", str(artifact_dir)
    ])

    try:
        main()
        assert (artifact_dir / "user_map.json").exists()
    except (ValueError, IndexError):
        assert True
