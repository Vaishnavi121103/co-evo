"""Tests for the evaluation metrics -- especially the oscillation index and
convergence detection, which are the study's novel measurements.
"""

import numpy as np
import pytest

from coevomal.evaluation.metrics import (
    oscillation_index,
    rounds_to_convergence,
)


def test_oscillation_index_zero_for_flat_series():
    assert oscillation_index([0.2, 0.2, 0.2, 0.2], window=3) == pytest.approx(0.0, abs=1e-12)


def test_oscillation_index_high_for_swinging_series():
    flat = oscillation_index([0.2, 0.2, 0.2, 0.2], window=4)
    swing = oscillation_index([0.0, 1.0, 0.0, 1.0], window=4)
    assert swing > flat
    assert np.isclose(swing, 0.5)


def test_convergence_detected_when_settled():
    rates = [0.9, 0.6, 0.4, 0.31, 0.30, 0.31, 0.30]
    conv = rounds_to_convergence(rates, window=3, tol=0.02)
    assert conv is not None
    assert conv >= 3


def test_convergence_none_when_divergent():
    rates = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    assert rounds_to_convergence(rates, window=3, tol=0.02) is None
