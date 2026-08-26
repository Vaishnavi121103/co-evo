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

- **evasion rate** per round — fraction of the malicious pool the defender ends up calling benign
- **attack success rate** — of the samples the defender *initially caught*, the fraction the attacker
  flipped. Reported separately from **pre-evasive rate** (the defender's baseline false negatives),
  because conflating the two overstates the attacker: a sample the classifier already misses is not
  an attack success. This is the cleaner signal when comparing retraining policies.
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

## Running on the real EMBER-2018 dataset

The real-data path is **implemented**, not just a stub.

```bash
# 1. Download EMBER-2018 (v2), ~1.6 GB, into data/ember/
curl -L -C - -o data/ember/ember2018.tar.bz2 https://ember.elastic.co/ember_dataset_2018_2.tar.bz2

# 2. Extract the raw-feature JSONL
tar -xjf data/ember/ember2018.tar.bz2 -C data/ember/

# 3. Run — the first run vectorizes + caches a balanced subsample, then co-evolves
python -m coevomal run --config configs/ember.yaml

# 4. The real-data factorial policy study, multi-seed (mean ± std for publication)
python experiments/run_multiseed.py --config configs/ember_factorial.yaml --seeds 5 \
    --out results/ember_multiseed
```

How it works (`environment/ember_loader.py`):

- EMBER ships **already-extracted** raw features as JSON. We vectorize them with EMBER's *own*
  official vectorizer (vendored in `environment/ember_features.py`, `feature_version=2` → the
  canonical **2381-dim** vector), so no `lief` and no `ember` package install is required — it runs
  clean on Python 3.13.
- We draw a **balanced subsample** (configurable) and **z-score** it; a full 600k-sample factorial ×
  multi-seed study is intractable, and every policy sees the identical subsample so the *comparison*
  is unbiased. Tree ensembles are scale-invariant, so standardization only makes the attack
  well-posed, it doesn't change the defender.
- The attacker may additively mutate only the feature blocks an append/add-style,
  functionality-preserving mutation can plausibly grow (byte histogram, byte-entropy, strings,
  sections, imports); header/optional-header/exports/data-directories are structural and immutable.

**Limitation (stated for the thesis):** this is the *feature-space* study — the attacker perturbs
EMBER feature vectors, not raw PE bytes. A raw-binary attacker (gym-malware / MAB-malware around
`MalwareEvasionEnv.reset/step`) needs actual malware binaries under institutional/legal sign-off and
VM isolation, per the roadmap's risk register. The feature-space formulation is standard and
publishable in this line of work.

**Exact EMBER model:** swap `GBDTDefender`'s backend for LightGBM (optional dependency) if you need
the exact reference classifier rather than the scikit-learn histogram GBDT.

## License

MIT.
