"""
Per-channel Pearson r evaluation: vectorized (T,C,B)→(C,B) scoring and k-fold CV.
"""

import numpy as np
from config import N_FOLDS, BAND_NAMES
from core.splits import contiguous_kfold_splits


def pearson_r_matrix(feats, target_1d):
    """
    Vectorized Pearson r: every (channel, band) pair vs a 1D target.

    feats     : (T, C, B)
    target_1d : (T,)
    Returns   : (C, B) in [-1, 1]
    """
    fc    = feats - feats.mean(axis=0, keepdims=True)
    tc    = target_1d - target_1d.mean()
    num   = (fc * tc[:, None, None]).sum(axis=0)
    denom = np.sqrt((fc ** 2).sum(axis=0) * (tc ** 2).sum() + 1e-12)
    return np.clip(num / denom, -1.0, 1.0)


def pearson_r(feats, target):
    """
    Mean signed Pearson r across all target columns for channel ranking.

    feats  : (T, C, B)
    target : (T, K)  K = 2 (pos) or 4 (joint pos+vel)
    Returns: (C, B)  mean r across K columns
    """
    K = target.shape[1]
    return sum(pearson_r_matrix(feats, target[:, k]) for k in range(K)) / K


def cv_channel_selection(feats, target, n_folds=N_FOLDS, valid_mask=None,
                         return_per_axis=False):
    """
    Evaluate per-channel, per-band Pearson r with contiguous k-fold CV.

    Pearson r is invariant to affine scaling, so no z-scoring is applied here
    (pearson_r_matrix demeans internally).

    Parameters
    ----------
    return_per_axis : bool
        When True, also return a list of K (C, B) arrays — one per target
        column — holding the mean cross-validated test r for that axis alone.
        Useful for diagnosing signed-mean cancellation.

    Returns
    -------
    test_r_cv   : (C, B)           mean test r across folds (signed mean over axes)
    train_r_cv  : (C, B)           mean train r
    fold_test_r : (n_folds, C, B)  per-fold test r
    [per_axis]  : list of K (C, B) — only when return_per_axis=True
    """
    K = target.shape[1]
    fold_train, fold_test = [], []
    fold_test_ax = [[] for _ in range(K)]   # per-axis folds

    for tr_idx, te_idx in contiguous_kfold_splits(len(feats), k=n_folds):
        if valid_mask is not None:
            tr_idx = tr_idx[valid_mask[tr_idx]]
            te_idx = te_idx[valid_mask[te_idx]]
        if len(tr_idx) < 10 or len(te_idx) < 5:
            continue

        fold_train.append(pearson_r(feats[tr_idx], target[tr_idx]))
        fold_test.append(pearson_r(feats[te_idx],  target[te_idx]))
        if return_per_axis:
            for k in range(K):
                fold_test_ax[k].append(
                    pearson_r_matrix(feats[te_idx], target[te_idx, k]))

    fold_test  = np.stack(fold_test)
    fold_train = np.stack(fold_train)
    if return_per_axis:
        per_axis = [np.stack(folds).mean(0) for folds in fold_test_ax]
        return fold_test.mean(0), fold_train.mean(0), fold_test, per_axis
    return fold_test.mean(0), fold_train.mean(0), fold_test


def print_r_summary(test_r, train_r, good_idx):
    """Print per-band best-channel r and R² to stdout."""
    for bi, bname in enumerate(BAND_NAMES):
        best  = np.abs(test_r[:, bi]).argmax()
        r_val = test_r[best, bi]
        print(f'    {bname:8s}  ch {good_idx[best]+1:3d}  '
              f'test r={r_val:+.4f}  R²={r_val**2:.4f}  '
              f'train r={train_r[best,bi]:+.4f}')
