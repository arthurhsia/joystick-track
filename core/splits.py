"""
Cross-validation split generators.
"""

import numpy as np
from config import N_FOLDS, EMBARGO


def contiguous_kfold_splits(n_samples, k=N_FOLDS, embargo=EMBARGO):
    """
    Yield (tr_idx, te_idx) for k contiguous time-block folds.
    Last fold absorbs any remainder so every sample appears in exactly one test fold.

    embargo: training samples within this many samples of each test block boundary
    are excluded on both sides to prevent leakage from temporal feature memory
    (envelope filter settling + lag embedding).  Test indices are never modified.
    """
    fold_size = n_samples // k
    for fold in range(k):
        te_start = fold * fold_size
        te_end   = te_start + fold_size if fold < k - 1 else n_samples
        te_idx   = np.arange(te_start, te_end)
        lo = max(0, te_start - embargo)
        hi = min(n_samples, te_end + embargo)
        tr_idx = np.concatenate([np.arange(0, lo), np.arange(hi, n_samples)])

        pre  = tr_idx[tr_idx < te_start]
        post = tr_idx[tr_idx >= te_end]
        assert (not len(pre)  or te_start - pre.max()  >= embargo), \
            f"fold {fold}: pre-boundary training sample within embargo of test block"
        assert (not len(post) or post.min() - te_end    >= embargo), \
            f"fold {fold}: post-boundary training sample within embargo of test block"

        yield tr_idx, te_idx


def trial_loto_splits(trial_ids):
    """Leave-one-trial-out splits."""
    for tid in np.unique(trial_ids):
        te_idx = np.where(trial_ids == tid)[0]
        tr_idx = np.where(trial_ids != tid)[0]
        yield tr_idx, te_idx
