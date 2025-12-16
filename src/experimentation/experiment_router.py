import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class VariantConfig:
    """Defines the label and traffic weight of a single variant."""

    name: str
    weight: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Strongly typed wrapper for experiment configuration."""

    experiment_id: str
    variants: Sequence[VariantConfig]
    salt: str = ""


@dataclass(frozen=True)
class ExperimentAssignment:
    """Result of routing a user into an experiment bucket."""

    experiment_id: str
    variant: str
    bucket: float


class HashExperimentRouter:
    """
    Deterministic experiment router that uses SHA-256 hashing of
    (salt + user_id) to place every user into a traffic bucket [0, 1).
    """

    def __init__(self, config: ExperimentConfig):
        if not config.variants:
            raise ValueError("Experiment must define at least one variant.")

        weights = [v.weight for v in config.variants]
        if any(w < 0 for w in weights):
            raise ValueError("Variant weights must be non-negative.")

        total = sum(weights)
        if total <= 0:
            raise ValueError("Sum of variant weights must be positive.")

        # Pre-compute cumulative ranges (0-1) for fast bucket lookup.
        cumulative = []
        cursor = 0.0
        for variant in config.variants:
            share = variant.weight / total
            upper = cursor + share
            cumulative.append((variant.name, cursor, upper))
            cursor = upper
        cumulative[-1] = (cumulative[-1][0], cumulative[-1][1], 1.0)

        self._config = config
        self._ranges: List[Tuple[str, float, float]] = cumulative

    @classmethod
    def from_dict(cls, payload: Dict) -> "HashExperimentRouter":
        """Convenience loader for JSON/YAML configs."""
        experiment_id = payload["experiment_id"]
        salt = payload.get("salt", "")
        variants = [
            VariantConfig(name=v["name"], weight=float(v["weight"]))
            for v in payload.get("variants", [])
        ]
        return cls(ExperimentConfig(experiment_id=experiment_id, variants=variants, salt=salt))

    @classmethod
    def from_json_file(cls, path: Path | str) -> "HashExperimentRouter":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)

    @staticmethod
    def _hash_to_bucket(user_id: str, salt: str) -> float:
        digest = hashlib.sha256(f"{salt}:{user_id}".encode("utf-8")).hexdigest()
        # Use first 15 hex chars (~60 bits) to stay within float precision comfortably.
        bucket_int = int(digest[:15], 16)
        max_int = 16 ** 15
        return bucket_int / max_int

    def assign(self, user_id: int | str) -> ExperimentAssignment:
        user_str = str(user_id)
        bucket = self._hash_to_bucket(user_str, self._config.salt)
        for variant_name, lower, upper in self._ranges:
            if lower <= bucket < upper:
                return ExperimentAssignment(
                    experiment_id=self._config.experiment_id,
                    variant=variant_name,
                    bucket=bucket,
                )
        # Numerically we should never reach this because last bucket is inclusive of 1.0.
        last_variant = self._ranges[-1][0]
        return ExperimentAssignment(
            experiment_id=self._config.experiment_id,
            variant=last_variant,
            bucket=1.0,
        )

    def allocation(self) -> List[Tuple[str, float]]:
        """Returns the normalized allocation for each variant."""
        return [(name, upper - lower) for name, lower, upper in self._ranges]


def load_router(config_path: str | Path) -> HashExperimentRouter:
    """
    Helper to align with dependency injection frameworks.
    Defaults to a simple 50/50 split if the path is missing.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        default = {
            "experiment_id": "default_ab",
            "salt": "default-salt",
            "variants": [
                {"name": "variant_a", "weight": 1},
                {"name": "variant_b", "weight": 1},
            ],
        }
        return HashExperimentRouter.from_dict(default)
    return HashExperimentRouter.from_json_file(config_file)
