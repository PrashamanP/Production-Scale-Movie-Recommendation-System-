import json
import pytest
import requests

from src import build_movie_catalog
@pytest.fixture
def setup_files(tmp_path):
    """A fixture to create a dummy interactions.csv and a partial catalog."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    interactions_file = data_dir / "interactions.csv"
    interactions_file.write_text("movie_id\nm1\nm2\nm3\n")

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    catalog_file = artifacts_dir / "movie_catalog.json"
    with open(catalog_file, "w") as f:
        json.dump({"m1": {"title": "Movie One", "genres": []}}, f)

    return str(interactions_file), str(catalog_file)

def test_fetch_movie_success(mocker):
    """Tests that fetch_movie returns JSON on a 200 OK response."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"title": "Toy Story", "genres": ["Animation"]}
    mocker.patch("requests.get", return_value=mock_response)

    result = build_movie_catalog.fetch_movie("toy+story+1995")

    assert result is not None
    assert result["title"] == "Toy Story"


def test_fetch_movie_api_error(mocker):
    """Tests that fetch_movie returns None on a non-200 response."""
    mock_response = mocker.Mock()
    mock_response.status_code = 404  
    mocker.patch("requests.get", return_value=mock_response)

    result = build_movie_catalog.fetch_movie("non_existent_movie")

    assert result is None


def test_fetch_movie_network_error_with_retry(mocker):
    """Tests that fetch_movie retries on a network exception and eventually fails."""
    mocker.patch("requests.get", side_effect=requests.RequestException("Connection failed"))

    mock_sleep = mocker.patch("time.sleep")

    result = build_movie_catalog.fetch_movie("a_movie")

    assert result is None
    assert requests.get.call_count == build_movie_catalog.RETRY + 1
    assert mock_sleep.call_count == build_movie_catalog.RETRY


def test_main_logic_fetches_only_missing_movies(mocker, setup_files):
    """
    Tests the main script logic to ensure it only fetches movies
    not already in the cache.
    """
    interactions_path, catalog_path = setup_files

    def mock_fetch(movie_id):
        if movie_id == "m2":
            return {"title": "Movie Two", "genres": ["Action"]}
        if movie_id == "m3":
            return {"title": "Movie Three", "genres": ["Comedy"]}
        return None

    mocker.patch("src.build_movie_catalog.fetch_movie", side_effect=mock_fetch)
    mocker.patch("time.sleep") 

    mocker.patch.object(build_movie_catalog, "DATA", interactions_path)
    mocker.patch.object(build_movie_catalog, "OUT", catalog_path)

    build_movie_catalog.main()

    assert build_movie_catalog.fetch_movie.call_count == 2
    build_movie_catalog.fetch_movie.assert_any_call("m2")
    build_movie_catalog.fetch_movie.assert_any_call("m3")

    with open(catalog_path, "r") as f:
        final_catalog = json.load(f)

    assert len(final_catalog) == 3
    assert "m1" in final_catalog
    assert "m2" in final_catalog
    assert "m3" in final_catalog
    assert final_catalog["m2"]["title"] == "Movie Two"


def test_fetch_movie_malformed_json_response(mocker):
    """Tests that fetch_movie handles malformed JSON responses."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mocker.patch("requests.get", return_value=mock_response)

    result = build_movie_catalog.fetch_movie("bad_json_movie")

    assert result is None


def test_fetch_movie_timeout(mocker):
    """Tests that fetch_movie handles timeouts with retry."""
    mocker.patch("requests.get", side_effect=requests.Timeout("Connection timeout"))
    mock_sleep = mocker.patch("time.sleep")

    result = build_movie_catalog.fetch_movie("slow_movie")

    assert result is None
    assert requests.get.call_count == build_movie_catalog.RETRY + 1
    assert mock_sleep.call_count == build_movie_catalog.RETRY


def test_main_with_empty_interactions_file(mocker, tmp_path):
    """Tests that main() handles an empty interactions file gracefully."""
    # Create empty interactions file
    interactions_path = tmp_path / "empty_interactions.csv"
    interactions_path.write_text("movie_id\n")  # Header only

    catalog_path = tmp_path / "catalog.json"

    mocker.patch("src.build_movie_catalog.fetch_movie")
    mocker.patch.object(build_movie_catalog, "DATA", str(interactions_path))
    mocker.patch.object(build_movie_catalog, "OUT", str(catalog_path))

    build_movie_catalog.main()

    # Should create catalog even if empty
    assert catalog_path.exists()
    with open(catalog_path, "r") as f:
        catalog = json.load(f)
    assert catalog == {}


def test_main_checkpoint_saving_at_200_intervals(mocker, tmp_path):
    """Tests that catalog is saved at checkpoint intervals (every 200 movies)."""
    interactions_path = tmp_path / "interactions.csv"
    # Create 250 movies to trigger checkpoint
    movie_ids = [f"m{i}" for i in range(250)]
    interactions_path.write_text("movie_id\n" + "\n".join(movie_ids))

    catalog_path = tmp_path / "catalog.json"

    def mock_fetch(movie_id):
        return {"title": f"Movie {movie_id}", "genres": []}

    mocker.patch("src.build_movie_catalog.fetch_movie", side_effect=mock_fetch)
    mocker.patch("time.sleep")

    # Track file writes to verify checkpoint saving
    original_open = open
    write_count = 0

    def counting_open(*args, **kwargs):
        nonlocal write_count
        if len(args) > 0 and args[0] == str(catalog_path) and len(args) > 1 and 'w' in args[1]:
            write_count += 1
        return original_open(*args, **kwargs)

    mocker.patch("builtins.open", side_effect=counting_open)

    mocker.patch.object(build_movie_catalog, "DATA", str(interactions_path))
    mocker.patch.object(build_movie_catalog, "OUT", str(catalog_path))

    build_movie_catalog.main()

    # Should write at i=200 checkpoint + final write = at least 2 writes
    assert write_count >= 2


def test_fetch_movie_partial_data(mocker):
    """Tests that fetch_movie handles API responses with missing fields."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    # Missing "genres" field
    mock_response.json.return_value = {"title": "Incomplete Movie"}
    mocker.patch("requests.get", return_value=mock_response)

    result = build_movie_catalog.fetch_movie("incomplete_movie")

    assert result is not None
    assert result["title"] == "Incomplete Movie"


def test_main_preserves_existing_catalog_on_failure(mocker, tmp_path):
    """Tests that existing catalog entries are preserved even if new fetches fail."""
    interactions_path = tmp_path / "interactions.csv"
    interactions_path.write_text("movie_id\nm1\nm2\nm3\n")

    catalog_path = tmp_path / "catalog.json"
    # Pre-populate catalog with m1
    with open(catalog_path, "w") as f:
        json.dump({"m1": {"title": "Existing Movie", "genres": []}}, f)

    def mock_fetch(movie_id):
        if movie_id == "m2":
            return {"title": "Movie Two", "genres": []}
        # m3 fails to fetch
        return None

    mocker.patch("src.build_movie_catalog.fetch_movie", side_effect=mock_fetch)
    mocker.patch("time.sleep")

    mocker.patch.object(build_movie_catalog, "DATA", str(interactions_path))
    mocker.patch.object(build_movie_catalog, "OUT", str(catalog_path))

    build_movie_catalog.main()

    with open(catalog_path, "r") as f:
        final_catalog = json.load(f)

    # m1 should still be there, m2 should be added, m3 should not be in catalog
    assert "m1" in final_catalog
    assert "m2" in final_catalog
    assert "m3" not in final_catalog
    assert final_catalog["m1"]["title"] == "Existing Movie"

