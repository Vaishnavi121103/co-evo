"""Load the real EMBER-2018 dataset into the harness.

The EMBER tarball ships *raw* (already-extracted) features as JSON lines. We
vectorize a balanced subsample with EMBER's own official vectorizer
(``ember_features.PEFeatureExtractor``, feature_version=2 -> the canonical
2381-dim vector), standardize it, and hand back a :class:`Dataset` +
held-out malicious pool with exactly the same interface as the synthetic
generator -- so the orchestrator, attackers, defenders, policies and metrics
all run unchanged.

Two modelling choices worth stating for the writeup:

* **Subsampling.** EMBER-2018 has 600k labelled training samples; running a
  full factorial x multi-seed co-evolution study over all of them is
  intractable. We draw a balanced random subsample (default configurable),
  which is standard practice and does not bias a policy *comparison* since
  every policy sees the identical subsample.

* **Standardization.** Raw EMBER features span wildly different scales (file
  size in bytes vs. a normalized byte histogram). We z-score with a
  ``StandardScaler`` fit on the training subsample and run the whole pipeline
  in standardized units, so a mutation of a given magnitude is comparable
  across features. Tree ensembles are scale-invariant, so this does not
  change the defender's power; it only makes the attack well-posed.

* **Mutable features.** Only feature blocks an append/add-style,
  functionality-preserving mutation can plausibly *increase* are marked
  mutable and add-only: the byte histogram, byte-entropy histogram, string
  features, section-info, and imports. Header, optional-header, exports and
  data-directory features are structural and left immutable. This mirrors
  the real gym-malware action space at the feature level.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from coevomal.environment.dataset import BENIGN, MALICIOUS, Dataset, FeatureSpace
from coevomal.environment.ember_features import PEFeatureExtractor

# Feature blocks an append/add attacker can plausibly grow.
ADDITIVE_BLOCKS = {"histogram", "byteentropy", "strings", "section", "imports"}


def _block_offsets(extractor: PEFeatureExtractor) -> list[tuple[str, int, int]]:
    """Return [(block_name, start, end), ...] over the concatenated vector."""
    offsets = []
    start = 0
    for fe in extractor.features:
        offsets.append((fe.name, start, start + fe.dim))
        start += fe.dim
    return offsets


def _find_jsonl(data_dir: Path) -> tuple[list[Path], Path | None]:
    """Locate EMBER train_features_*.jsonl and test_features.jsonl."""
    # The tarball extracts to an 'ember2018' (or similar) subdirectory.
    candidates = list(data_dir.rglob("train_features_*.jsonl"))
    train_files = sorted(candidates)
    test_matches = list(data_dir.rglob("test_features.jsonl"))
    test_file = test_matches[0] if test_matches else None
    return train_files, test_file


def _vectorize_stream(
    files: list[Path],
    extractor: PEFeatureExtractor,
    n_per_class: int,
    rng: np.random.Generator,
    want_labels=(BENIGN, MALICIOUS),
) -> tuple[np.ndarray, np.ndarray]:
    """Stream JSONL, vectorize labelled rows until n_per_class of each is met."""
    need = {int(l): n_per_class for l in want_labels}
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    for path in files:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if all(v <= 0 for v in need.values()):
                    break
                obj = json.loads(line)
                label = int(obj.get("label", -1))
                if label not in need or need[label] <= 0:
                    continue
                vec = extractor.process_raw_features(obj)
                X_rows.append(vec)
                y_rows.append(label)
                need[label] -= 1
        if all(v <= 0 for v in need.values()):
            break
    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int64)
    # Shuffle so classes are interleaved.
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


def load_ember_dataset(
    data_dir: str | Path,
    n_train: int = 20000,
    n_test_malicious: int = 1000,
    mutable_fraction: float = 0.5,
    seed: int = 0,
    cache: bool = True,
) -> tuple[Dataset, np.ndarray]:
    """Load real EMBER into a (Dataset, test_malicious) pair.

    Parameters mirror :func:`coevomal.environment.dataset.make_synthetic` so
    the two datasets are drop-in interchangeable via the ``dataset`` config.
    Vectorized subsamples are cached as ``.npz`` under ``data_dir`` because
    vectorization is the slow step; delete the cache to re-sample.
    """
    data_dir = Path(data_dir)
    rng = np.random.default_rng(seed)
    extractor = PEFeatureExtractor(feature_version=2, print_feature_warning=False)

    cache_path = data_dir / f"cache_ember_{n_train}_{n_test_malicious}_{seed}.npz"
    if cache and cache_path.exists():
        data = np.load(cache_path)
        Xtr, ytr, Xtest_mal = data["Xtr"], data["ytr"], data["Xtest_mal"]
    else:
        train_files, test_file = _find_jsonl(data_dir)
        if not train_files:
            raise FileNotFoundError(
                f"No EMBER train_features_*.jsonl found under {data_dir}. "
                "Extract ember2018.tar.bz2 there first."
            )
        Xtr, ytr = _vectorize_stream(
            train_files, extractor, n_per_class=n_train // 2, rng=rng
        )
        # Held-out malicious pool for evaluation, drawn from the test split
        # (or a second pass of train if the test file is absent).
        src = [test_file] if test_file else train_files
        Xtest_mal, _ = _vectorize_stream(
            src, extractor, n_per_class=n_test_malicious, rng=rng,
            want_labels=(MALICIOUS,),
        )
        Xtest_mal = Xtest_mal[:n_test_malicious]
        if cache:
            np.savez_compressed(
                cache_path, Xtr=Xtr, ytr=ytr, Xtest_mal=Xtest_mal
            )

    # ---- standardize (fit on train subsample) -------------------------------
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr).astype(np.float32)
    Xtest_mal_s = scaler.transform(Xtest_mal).astype(np.float32)
    # Guard against non-finite values from constant columns.
    Xtr_s = np.nan_to_num(Xtr_s, nan=0.0, posinf=8.0, neginf=-8.0)
    Xtest_mal_s = np.nan_to_num(Xtest_mal_s, nan=0.0, posinf=8.0, neginf=-8.0)

    # ---- feature space: which dims an add-only attacker may perturb ---------
    n_features = Xtr_s.shape[1]
    offsets = _block_offsets(extractor)
    eligible: list[int] = []
    for name, start, end in offsets:
        if name in ADDITIVE_BLOCKS:
            eligible.extend(range(start, end))
    eligible = np.array(sorted(eligible), dtype=np.int64)
    n_mutable = max(1, int(round(mutable_fraction * eligible.size)))
    mutable_idx = np.sort(rng.choice(eligible, size=n_mutable, replace=False))
    additive_only = np.ones(mutable_idx.size, dtype=bool)  # append semantics

    lo = np.full(n_features, -8.0, dtype=np.float32)
    hi = np.full(n_features, 8.0, dtype=np.float32)
    fs = FeatureSpace(
        n_features=n_features,
        mutable_idx=mutable_idx,
        additive_only=additive_only,
        low=lo,
        high=hi,
    )
    dataset = Dataset(X=Xtr_s, y=ytr, feature_space=fs)
    return dataset, Xtest_mal_s
