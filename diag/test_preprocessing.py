"""
Unit tests for core/preprocessing.py — synthetic channel scenarios.

Run with:  python -m pytest tests/test_preprocessing.py -v
           python tests/test_preprocessing.py   (standalone)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from core.preprocessing import (
    preprocess, detect_bad_channels, _line_noise_snr,
    _channel_amplitude, _build_detect_copy, _mad_z, _B_NOTCH, _A_NOTCH,
)
from config import FS, FLAT_AMP_THRESH

RNG   = np.random.default_rng(42)
T     = 10 * FS   # 10 s of data


def _make_data(**overrides):
    """
    Return (x_raw, x_notch) for a 5-channel array with named column slots.
    Columns (0-indexed): 0=normal, 1=flat, 2=high_amp, 3=high_lnr, 4=strong_lmp
    All overrides replace a whole column: key = col index, value = (T,) array.
    """
    normal_amp = 200.0   # typical int16 amplitude

    x = np.zeros((T, 5), dtype=float)

    # 0: normal clean channel
    x[:, 0] = RNG.normal(0, normal_amp, T)

    # 1: flat / dead — std << FLAT_AMP_THRESH
    x[:, 1] = RNG.normal(0, 0.01, T)

    # 2: high amplitude outlier — IQR >> other channels
    x[:, 2] = RNG.normal(0, normal_amp * 15, T)

    # 3: strong 60 Hz line noise
    t_ax = np.arange(T) / FS
    x[:, 3] = RNG.normal(0, normal_amp, T) + normal_amp * 30 * np.sin(2 * np.pi * 60 * t_ax)

    # 4: strong LMP (slow, large amplitude) — should be preserved with HP-detect copy
    # Low-frequency component big enough to trigger amp z-score WITHOUT the HP copy
    t_ax = np.arange(T) / FS
    lmp  = normal_amp * 12 * np.sin(2 * np.pi * 0.3 * t_ax)  # 0.3 Hz
    x[:, 4] = RNG.normal(0, normal_amp, T) + lmp

    for col, arr in overrides.items():
        x[:, col] = arr

    import scipy.signal as sig
    x_notch = sig.filtfilt(_B_NOTCH, _A_NOTCH, x, axis=0)
    return x.astype(float), x_notch


# ─────────────────────────────────────────────────────────────────────────────


def test_flat_channel_detected():
    """Flat/dead channel (col 1) must be flagged by the flat criterion."""
    x_raw, x_notch = _make_data()
    lnr = _line_noise_snr(x_raw)
    bad, reasons = detect_bad_channels(x_raw, x_notch, line_noise_snr=lnr)
    assert reasons['flat'][1], "Flat channel was not caught by flat criterion"
    assert bad[1]



def test_high_amplitude_channel_detected():
    """Channel with 15× normal amplitude (col 2) must fire amplitude criterion."""
    x_raw, x_notch = _make_data()
    lnr = _line_noise_snr(x_raw)
    bad, reasons = detect_bad_channels(x_raw, x_notch, line_noise_snr=lnr)
    assert reasons['amplitude'][2], "High-amplitude channel was not caught"
    assert bad[2]


def test_high_line_noise_channel_detected():
    """Channel with 30× 60 Hz sine (col 3) must fire line-noise criterion."""
    x_raw, x_notch = _make_data()
    lnr = _line_noise_snr(x_raw)
    bad, reasons = detect_bad_channels(x_raw, x_notch, line_noise_snr=lnr)
    assert reasons['line_noise'][3], "Line-noise channel was not caught"
    assert bad[3]


def test_strong_lmp_channel_NOT_flagged_by_amplitude():
    """
    Strong LMP channel (col 4) must NOT be flagged on amplitude.
    The HP detect copy removes the slow LMP before computing IQR×0.7413,
    so the channel's HP amplitude is similar to normal channels.
    """
    x_raw, x_notch = _make_data()
    lnr = _line_noise_snr(x_raw)
    bad, reasons = detect_bad_channels(x_raw, x_notch, line_noise_snr=lnr,
                                       detect_hp_hz=2.0)
    # LMP channel may be borderline but should not fire amplitude with HP copy
    assert not reasons['amplitude'][4], (
        "LMP channel incorrectly flagged by amplitude — HP copy not removing slow component"
    )


def test_lmp_preservation_in_preprocess():
    """preprocess() must not modify V_notch in-place (checked internally via checksum)."""
    x_raw, _ = _make_data()
    # Just verify it runs without AssertionError (the assertion is inside preprocess)
    x_car, good_idx, bad_mask, reasons = preprocess(x_raw.astype(int))
    assert x_car.shape[0] == T
    assert len(good_idx) == (~bad_mask).sum()


def test_preprocess_returns_four_tuple():
    """preprocess() returns exactly 4 values."""
    x_raw, _ = _make_data()
    result = preprocess(x_raw.astype(int))
    assert len(result) == 4, f"Expected 4-tuple, got {len(result)}-tuple"


if __name__ == '__main__':
    import traceback
    tests = [
        test_flat_channel_detected,
        test_high_amplitude_channel_detected,
        test_high_line_noise_channel_detected,
        test_strong_lmp_channel_NOT_flagged_by_amplitude,
        test_lmp_preservation_in_preprocess,
        test_preprocess_returns_four_tuple,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f'  PASS  {t.__name__}')
            passed += 1
        except Exception as e:
            print(f'  FAIL  {t.__name__}: {e}')
            traceback.print_exc()
            failed += 1
    print(f'\n{passed} passed, {failed} failed')
