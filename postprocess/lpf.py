"""
Temporal LPF screen: test zero-phase LPFs on ridge predictions to find the
per-axis optimal smoothing cutoff.
"""

import numpy as np
import scipy.signal as sig
from config import FS_FEAT
from core.metrics import pearson_r_1d

CUTOFFS = [0.25, 0.5, 1.0, 2.0]   # Hz to evaluate


def smooth_pred(pred_1d, cutoff_hz, fs=FS_FEAT, order=4):
    """Apply zero-phase Butterworth LPF to a 1D prediction array."""
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        return pred_1d.copy()
    b, a    = sig.butter(order, cutoff_hz / nyq, btype='low')
    min_len = 3 * (max(len(b), len(a)) - 1) + 1
    return sig.filtfilt(b, a, pred_1d) if len(pred_1d) >= min_len else pred_1d.copy()


def temporal_screen(pos_pred, pos_true, cutoffs=CUTOFFS):
    """
    Oracle diagnostic: grid over LPF cutoffs; pick the one maximising per-axis r on pos_true.

    Both the cutoff choice and the per-axis independence are test-set tuned —
    this result should not be reported as a primary outcome.

    Returns
    -------
    rows   : list of (label, r_cx, r_cy) — 'base' then one per cutoff
    opt_cx : float  best cutoff Hz for cx (0.0 = no smoothing wins)
    opt_cy : float  best cutoff Hz for cy (0.0 = no smoothing wins)
    """
    base_cx = pearson_r_1d(pos_true[:, 0], pos_pred[:, 0])
    base_cy = pearson_r_1d(pos_true[:, 1], pos_pred[:, 1])
    rows    = [('base', base_cx, base_cy)]

    best_cx = base_cx; best_cy = base_cy
    opt_cx  = 0.0;     opt_cy  = 0.0

    for c in cutoffs:
        sm_cx = smooth_pred(pos_pred[:, 0], c)
        sm_cy = smooth_pred(pos_pred[:, 1], c)
        r_cx  = pearson_r_1d(pos_true[:, 0], sm_cx)
        r_cy  = pearson_r_1d(pos_true[:, 1], sm_cy)
        rows.append((f'{c}Hz', r_cx, r_cy))
        if r_cx > best_cx: best_cx = r_cx; opt_cx = c
        if r_cy > best_cy: best_cy = r_cy; opt_cy = c

    return rows, opt_cx, opt_cy
