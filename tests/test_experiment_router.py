import json
from pathlib import Path

import pytest

from src.experimentation.experiment_router import (
    ExperimentConfig,
    HashExperimentRouter,
    VariantConfig,
    load_router,
)


def test_router_deterministic_assignment():
    config = ExperimentConfig(
        experiment_id="als_vs_pop",
        salt="v1",
        variants=[VariantConfig("als", 1), VariantConfig("pop", 1)],
    )
    router = HashExperimentRouter(config)

    assignment_1 = router.assign(12345)
    assignment_2 = router.assign(12345)

    assert assignment_1.variant == assignment_2.variant
    assert assignment_1.bucket == assignment_2.bucket


def test_router_respects_weights():
    config = ExperimentConfig(
        experiment_id="small_bias",
        salt="bias",
        variants=[
            VariantConfig("majority", 3),
            VariantConfig("minority", 1),
        ],
    )
    router = HashExperimentRouter(config)
    allocations = dict(router.allocation())

    assert pytest.approx(allocations["majority"], rel=1e-3) == 0.75
    assert pytest.approx(allocations["minority"], rel=1e-3) == 0.25


def test_router_from_json_file(tmp_path: Path):
    config_path = tmp_path / "experiment.json"
    config_payload = {
        "experiment_id": "json_ab",
        "salt": "salty",
        "variants": [
            {"name": "A", "weight": 1},
            {"name": "B", "weight": 1},
        ],
    }
    config_path.write_text(json.dumps(config_payload))

    router = load_router(config_path)
    assignment = router.assign("user-1")

    assert assignment.experiment_id == "json_ab"
    assert assignment.variant in {"A", "B"}


def test_router_raises_for_invalid_weights():
    config = ExperimentConfig(
        experiment_id="bad",
        salt="bad",
        variants=[VariantConfig("A", -1)],
    )
    with pytest.raises(ValueError):
        HashExperimentRouter(config)
