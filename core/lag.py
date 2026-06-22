"""
Temporal lag matrix construction with per-trial boundary guards.
"""

import numpy as np
from config import LAGS


def build_lag_matrix(X, lags=LAGS):
    """
    Stack time-lagged copies of feature matrix.
    Lag k: column t uses feature value from time t-k (first k rows padded).

    X    : (T, F)
    Returns: (T, F * len(lags))
    """
    T, F  = X.shape
    parts = []
    for lag in lags:
        if lag == 0:
            parts.append(X.copy())
        else:
            shifted        = np.empty_like(X)
            shifted[lag:]  = X[:-lag]
            shifted[:lag]  = X[0]
            parts.append(shifted)
    return np.hstack(parts)


def build_lag_matrix_by_trial(X, trial_ids, lags=LAGS):
    """
    Build lag matrix without cross-trial contamination.
    Each trial is lagged independently using that trial's own first sample as padding.

    X         : (T, F)    feature matrix (concatenated trials)
    trial_ids : (T,) int  trial index per sample
    Returns   : (T, F * len(lags))
    """
    T, F  = X.shape
    X_lag = np.empty((T, F * len(lags)), dtype=X.dtype)
    for tid in np.unique(trial_ids):
        idx        = np.where(trial_ids == tid)[0]
        X_lag[idx] = build_lag_matrix(X[idx], lags)
    return X_lag
