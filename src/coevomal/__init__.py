"""Co-Evolutionary Malware Detection benchmark harness.

A reusable harness for studying how a periodically-retrained malware
classifier (the *defender*) behaves under sustained pressure from an
RL-driven evasion agent (the *attacker*), with pluggable retraining
policies (cadence x data-selection x mode).

The scientific contribution this package is built to support is *not* a
new attacker or a new defense mechanism, but a controlled, factorial
comparison of retraining *policies* under a shared evaluation harness --
see ``docs/roadmap.md``.
"""

__version__ = "0.1.0"

from coevomal.config import ExperimentConfig  # noqa: E402

__all__ = ["ExperimentConfig", "__version__"]
