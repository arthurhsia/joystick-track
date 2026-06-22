"""
Scalar accuracy metrics: Pearson r (1D) and R².
"""

import numpy as np


def pearson_r_1d(y_true, y_pred):
    """Signed Pearson r between two 1D arrays. Returns float in [-1, 1]."""
    tc    = y_true - y_true.mean()
    pc    = y_pred - y_pred.mean()
    num   = (tc * pc).sum()
    denom = np.sqrt((tc ** 2).sum() * (pc ** 2).sum() + 1e-12)
    return float(np.clip(num / denom, -1.0, 1.0))


def r2(y_true, y_pred):
    """Coefficient of determination. Can be negative."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))
