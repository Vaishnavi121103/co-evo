"""Tests for the real-EMBER loader, using a tiny synthetic JSONL that mimics
EMBER's raw-feature schema. Verifies vectorization, label filtering,
standardization, and the additive-only feature-space construction without
needing the multi-GB dataset.
"""

import json

import numpy as np
import pytest

from coevomal.environment.ember_loader import (
    ADDITIVE_BLOCKS,
    FULLY_MUTABLE_BLOCKS,
    HEADER_FROZEN_OFFSETS,
    MONOTONE_UP,
    load_ember_dataset,
)


def _record(label: int, rng: np.random.Generator) -> dict:
    """A schema-complete EMBER raw-feature record with random-ish values."""
    return {
        "sha256": "".join(rng.choice(list("0123456789abcdef"), size=8).tolist()),
        "label": label,
        "histogram": rng.integers(0, 50, size=256).tolist(),
        "byteentropy": rng.integers(0, 50, size=256).tolist(),
        "strings": {
            "numstrings": int(rng.integers(1, 100)), "avlength": float(rng.uniform(3, 12)),
            "printabledist": rng.integers(0, 20, size=96).tolist(),
            "printables": int(rng.integers(50, 500)), "entropy": float(rng.uniform(2, 6)),
            "paths": int(rng.integers(0, 5)), "urls": int(rng.integers(0, 5)),
            "registry": int(rng.integers(0, 5)), "MZ": int(rng.integers(0, 3)),
        },
        "general": {
            "size": int(rng.integers(1000, 500000)), "vsize": int(rng.integers(1000, 600000)),
            "has_debug": int(rng.integers(0, 2)), "exports": int(rng.integers(0, 20)),
            "imports": int(rng.integers(0, 80)), "has_relocations": int(rng.integers(0, 2)),
            "has_resources": int(rng.integers(0, 2)), "has_signature": int(rng.integers(0, 2)),
            "has_tls": int(rng.integers(0, 2)), "symbols": int(rng.integers(0, 30)),
        },
        "header": {
            "coff": {"timestamp": int(rng.integers(0, 10**9)), "machine": "AMD64",
                     "characteristics": ["EXECUTABLE_IMAGE"]},
            "optional": {"subsystem": "WINDOWS_GUI", "dll_characteristics": ["DYNAMIC_BASE"],
                         "magic": "PE32_PLUS", "major_image_version": 1, "minor_image_version": 0,
                         "major_linker_version": 14, "minor_linker_version": 0,
                         "major_operating_system_version": 6, "minor_operating_system_version": 0,
                         "major_subsystem_version": 6, "minor_subsystem_version": 0,
                         "sizeof_code": 4096, "sizeof_headers": 1024, "sizeof_heap_commit": 4096},
        },
        "section": {"entry": ".text", "sections": [
            {"name": ".text", "size": 4096, "entropy": 6.1, "vsize": 4096,
             "props": ["MEM_READ", "MEM_EXECUTE"]}]},
        "imports": {"kernel32.dll": ["CreateFileA", "ReadFile"], "user32.dll": ["MessageBoxA"]},
        "exports": ["funcA", "funcB"],
        "datadirectories": [{"name": "IMPORT_TABLE", "size": 40, "virtual_address": 8192}],
    }


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def test_load_ember_from_fake_jsonl(tmp_path):
    rng = np.random.default_rng(0)
    # Train file: benign, malicious and unlabeled (-1) rows interleaved.
    train = []
    for _ in range(30):
        train.append(_record(0, rng))
        train.append(_record(1, rng))
        train.append(_record(-1, rng))  # must be filtered out
    _write_jsonl(tmp_path / "train_features_0.jsonl", train)
    _write_jsonl(tmp_path / "test_features.jsonl", [_record(1, rng) for _ in range(20)])

    ds, test_mal = load_ember_dataset(
        data_dir=tmp_path, n_train=20, n_test_malicious=10,
        mutable_fraction=0.5, seed=0, cache=False,
    )

    # canonical EMBER dim
    assert ds.X.shape[1] == 2381
    assert ds.X.shape[0] == 20
    assert test_mal.shape == (10, 2381)
    # only labelled 0/1 kept, roughly balanced
    assert set(np.unique(ds.y).tolist()).issubset({0, 1})
    assert ds.y.sum() == 10  # n_train//2 malicious
    # standardized + finite
    assert np.isfinite(ds.X).all()
    assert np.isfinite(test_mal).all()
    # Feature space: mutable dims come from the additive blocks. Only the few
    # genuine monotone counters are add-only; the normalized histograms and the
    # signed FeatureHasher outputs must be bidirectional, otherwise the modelled
    # attack cannot represent what a real append-style mutation does.
    fs = ds.feature_space
    assert fs.n_features == 2381
    assert fs.n_mutable >= 1
    assert not fs.additive_only.all(), "hashed/normalized features must be bidirectional"
    assert int(fs.additive_only.sum()) <= len(MONOTONE_UP["strings"]) + len(
        MONOTONE_UP["section"]
    )


def test_missing_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ember_dataset(data_dir=tmp_path, n_train=4, n_test_malicious=2, cache=False)


def test_mutable_blocks_cover_everything_an_append_can_change():
    """`general` and `datadirectories` must be attacker-controllable.

    Appending bytes provably changes file size / virtual size, and adding a
    section or import changes the data-directory sizes and RVAs. Freezing them
    lets a classifier trained on the frozen remainder alone reach 0.97 on real
    EMBER, so adversarial retraining just learns those features, evasion goes
    to zero, and every retraining policy ties -- the study becomes vacuous.
    """
    assert FULLY_MUTABLE_BLOCKS == {
        "histogram", "byteentropy", "strings", "section",
        "imports", "general", "exports", "datadirectories",
    }
    assert ADDITIVE_BLOCKS == FULLY_MUTABLE_BLOCKS   # back-compat alias


def test_only_structural_header_fields_are_frozen():
    """Header metadata is editable; machine/characteristics/subsystem/magic are not."""
    assert 0 not in HEADER_FROZEN_OFFSETS            # timestamp: editable
    for off in range(31, 41):                        # dll_characteristics
        assert off not in HEADER_FROZEN_OFFSETS
    for off in range(51, 62):                        # version / sizeof fields
        assert off not in HEADER_FROZEN_OFFSETS
    for off in list(range(1, 31)) + list(range(41, 51)):
        assert off in HEADER_FROZEN_OFFSETS          # structural
    assert len(HEADER_FROZEN_OFFSETS) == 40


def test_general_counters_are_add_only(tmp_path):
    """size/vsize/imports/exports/symbols only grow when content is added."""
    from coevomal.environment.ember_loader import MONOTONE_UP

    assert MONOTONE_UP["general"] == (0, 1, 3, 4, 9)


def test_eval_pool_disjoint_from_train_when_no_test_file(tmp_path):
    """Without EMBER's test split, the eval pool must exclude training shas.

    Otherwise the attacker would be scored on samples the defender trained
    on, which would invalidate every evasion number in the study.
    """
    rng = np.random.default_rng(3)
    recs = []
    for i in range(60):
        r = _record(1, rng)
        r["sha256"] = f"mal{i:04d}"   # deterministic, checkable ids
        recs.append(r)
        b = _record(0, rng)
        b["sha256"] = f"ben{i:04d}"
        recs.append(b)
    _write_jsonl(tmp_path / "train_features_0.jsonl", recs)
    # deliberately NO test_features.jsonl

    ds, test_mal = load_ember_dataset(
        data_dir=tmp_path, n_train=20, n_test_malicious=10,
        mutable_fraction=0.5, seed=0, cache=False,
    )
    assert ds.X.shape[0] == 20
    assert test_mal.shape[0] == 10
    # The pool rows must not duplicate any training row.
    train_rows = {tuple(np.round(r, 5)) for r in ds.X}
    pool_rows = {tuple(np.round(r, 5)) for r in test_mal}
    assert train_rows.isdisjoint(pool_rows)


def test_truncated_final_line_is_skipped(tmp_path):
    """A partially-written last line (mid-extraction) must not crash loading."""
    rng = np.random.default_rng(4)
    recs = [_record(i % 2, rng) for i in range(40)]
    path = tmp_path / "train_features_0.jsonl"
    _write_jsonl(path, recs)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"sha256": "trunc", "label": 1, "histo')  # truncated
    _write_jsonl(tmp_path / "test_features.jsonl", [_record(1, rng) for _ in range(10)])

    ds, test_mal = load_ember_dataset(
        data_dir=tmp_path, n_train=10, n_test_malicious=5, seed=0, cache=False,
    )
    assert ds.X.shape[0] == 10
    assert np.isfinite(ds.X).all()
