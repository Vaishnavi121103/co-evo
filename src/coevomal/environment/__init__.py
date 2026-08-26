"""Environment package: datasets and the evasion environment."""

from coevomal.environment.dataset import (
    BENIGN,
    MALICIOUS,
    Dataset,
    FeatureSpace,
    load_ember,
    make_synthetic,
)
from coevomal.environment.malware_env import (
    MalwareEvasionEnv,
    build_mimicry_directions,
)

__all__ = [
    "BENIGN",
    "MALICIOUS",
    "Dataset",
    "FeatureSpace",
    "MalwareEvasionEnv",
    "build_mimicry_directions",
    "load_ember",
    "make_synthetic",
]
