"""Tests for collapse_fp_duplicates (pSA/SDI period FP-noise de-duplication).

Self-contained: imports only the pure-numpy period_utils helper, so it runs in the
local venv (which cannot import im_calculation: seis2txt / uncompiled rspectra).
"""
import numpy as np

from IM_calculation.IM.period_utils import collapse_fp_duplicates


def test_merges_fp_adjacent_double_onto_canonical():
    # np.logspace's 10**-1 can land on the double just below 0.1; it must merge into 0.1.
    periods = np.array([0.1, 0.09999999999999999, 0.10722672220103231])
    out = collapse_fp_duplicates(periods, canonical=[0.1])
    labels = [str(p) for p in out]
    assert "0.09999999999999999" not in labels
    assert "0.1" in labels
    # the genuinely distinct logspace neighbour is preserved byte-for-byte
    assert "0.10722672220103231" in labels
    assert len(out) == 2


def test_collapses_all_power_of_ten_duplicates():
    periods = np.array([
        0.09999999999999999, 0.1,
        0.9999999999999999, 1.0,
        9.999999999999998, 10.0,
    ])
    out = collapse_fp_duplicates(periods, canonical=[0.1, 1.0, 10.0])
    assert sorted(out) == [0.1, 1.0, 10.0]


def test_leaves_distinct_periods_untouched():
    # 0.04977 / 0.20092 are ~0.46% from the canonical 0.05 / 0.2 -> must NOT merge.
    periods = np.array([0.05, 0.04977023564332111, 0.2, 0.20092330025650466])
    out = collapse_fp_duplicates(periods, canonical=[0.05, 0.2])
    assert len(out) == 4
    assert 0.04977023564332111 in out and 0.20092330025650466 in out
