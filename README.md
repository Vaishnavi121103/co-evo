# Co-Evolutionary Malware Detection — Retraining-Policy Benchmark Harness

A reusable harness for studying **which defensive retraining policy wins the iterated
attacker–defender game** in ML-based malware detection.

An RL **attacker** (DQN / PPO) learns functionality-preserving mutations that evade a **defender**
(a LightGBM-style GBDT, or a torch MLP). The defender periodically retrains under a **pluggable
policy** — *cadence* × *data-selection* × *mode* — and the **evaluation harness** scores every
policy combination with identical metrics, including a novel **oscillation index** that measures
whether the defender converges to a stable boundary or perpetually chases the attacker.

The scientific contribution this supports is **not** a new attacker or defense mechanism, but the
**controlled, factorial comparison of retraining policies** the surrounding 2022–2026 literature
does not provide. See [`docs/roadmap.md`](docs/roadmap.md) for the full framework, novelty
statement, and how each roadmap element maps to the code.

> **Scope note.** By default the harness runs on a **synthetic, EMBER-like feature space** — fully
> reproducible, no downloads, no live malware. This exercises the entire co-evolution loop and the
> policy study today. Real EMBER features and a real gym-malware PE-mutation environment are
> first-class extension points (`environment/dataset.py::load_ember`, `environment/malware_env.py`)
> and slot in without touching the orchestrator, policies, or metrics. Any work with real malware
> binaries must go through institutional/legal sign-off and run VM-isolated, as noted in the
> roadmap's risk register.

## Install

```bash
python -m venv .venv && . .venv/Scripts/activate    # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .            # or: pip install -r requirements.txt
```

LightGBM is optional; without it the defender uses scikit-learn's `HistGradientBoostingClassifier`,
which is algorithmically close to the EMBER reference model.

## Quickstart

```bash
# 1. Fast end-to-end smoke run (well under a minute on CPU)
python -m coevomal run --config configs/smoke.yaml

# 2. Reference single experiment
python -m coevomal run --config configs/default.yaml

# 3. Lower-bound baseline: frozen classifier that never retrains
python -m coevomal run --config configs/baseline_frozen.yaml

# 4. The core factorial policy sweep (cadence × data-selection)
python -m coevomal factorial --config configs/factorial.yaml --out results/factorial
```

Each run writes to `results/<name>/`: `result.json` (per-round logs + summary), `config.yaml`
(exact reproducible config), and `dynamics.png` (evasion rate, clean accuracy, attacker query
complexity). The factorial sweep also writes `factorial_summary.csv` and a `frontier.png`
robustness-vs-cost scatter.

## The independent variable: retraining policy

Configured under `retrain:` in any config (`src/coevomal/policies/retraining.py`):

| Axis | Options | Meaning |
|---|---|---|
| `cadence` | `every_round`, `every_n`, `threshold` | **When** to retrain |
| `data_selection` | `full_replay`, `hard_mining`, `bounded_buffer` | **What data** to retrain on |
| `mode` | `scratch`, `finetune` | Fresh fit vs. warm-start |

The factorial runner crosses `cadence × data_selection` (× attacker) so any difference in
convergence / oscillation / cost is attributable to a single policy variable.

## Key metrics (`src/coevomal/evaluation/metrics.py`)

- **evasion rate** per round — primary outcome
- **oscillation index** — trailing-window std of evasion rate (converged vs. chasing)
- **rounds-to-convergence** — or `None` for divergence
- **attacker query complexity** — mean defender queries per evasion (difficulty)
- **retraining cost** — wall-clock seconds + samples per round
- **robustness-per-cost** — the frontier EDR-vendor guidance is expressed on

## Project layout

```
src/coevomal/
  config.py                 # dataclass configs + YAML (de)serialization
  orchestrator.py           # the co-evolution round loop
  utils.py                  # seeding, device resolution
  environment/
    dataset.py              # synthetic EMBER-like data + load_ember() hook
    malware_env.py          # feature-space PE-mutation evasion environment
  defenders/                # base, GBDT (LightGBM stand-in), MLP (torch)
  attackers/                # base, DQN, PPO, random baseline
  policies/retraining.py    # cadence × data-selection × mode
  evaluation/               # metrics + plots
experiments/run_factorial.py
configs/                    # smoke / default / baseline_frozen / factorial
tests/                      # pytest suite
docs/roadmap.md             # full project framework & novelty
```

## Tests

```bash
pip install pytest
pytest            # unit tests for metrics, policies, environment + an end-to-end smoke run
```

## Extending to real data

1. **Real EMBER defender:** implement `environment/dataset.py::load_ember` to return a `Dataset` +
   held-out malicious pool from the EMBER vectorized features; set `dataset.name: ember` in a config.
2. **Real PE-mutation attacker:** re-implement `MalwareEvasionEnv.reset/step` around gym-malware /
   MAB-malware PE rewriting, keeping the same interface; the orchestrator, policies, and metrics are
   unchanged.
3. **Exact EMBER model:** swap `GBDTDefender`'s backend for LightGBM (optional dependency).

## License

MIT.
