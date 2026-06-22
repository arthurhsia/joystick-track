"""
Channel selection: pick top-N (channel, band) pairs globally by |r|.
"""

import numpy as np
from config import TOP_N, BAND_NAMES, SELECTION_METRIC


def _ranking_score(per_axis_rs, metric=SELECTION_METRIC):
    """
    Collapse a list of K (C, B) per-axis r matrices into a single non-negative
    (C, B) ranking score.

    'signed_mean' : |(r0 + r1 + ...) / K|  — default; can cancel opposite-sign pairs
    'mean_abs'    : (|r0| + |r1| + ...) / K — axis-symmetric
    'l2'          : sqrt(r0² + r1² + ...)   — axis-symmetric, rewards strong single-axis r
    """
    if metric == 'signed_mean':
        return np.abs(sum(per_axis_rs) / len(per_axis_rs))
    elif metric == 'mean_abs':
        return sum(np.abs(r) for r in per_axis_rs) / len(per_axis_rs)
    elif metric == 'l2':
        return np.sqrt(sum(r ** 2 for r in per_axis_rs))
    else:
        raise ValueError(f'Unknown SELECTION_METRIC: {metric!r}. '
                         f"Use 'signed_mean', 'mean_abs', or 'l2'.")


def select_features(feats, test_r, top_n=TOP_N):
    """
    Select top_n (channel, band) pairs ranked by |r|.

    feats  : (T, C, B)
    test_r : (C, B)  cross-validated Pearson r

    Returns
    -------
    X    : (T, top_n)  selected features ordered by |r| descending
    meta : list of (ch_idx, band_idx)
    """
    C, B      = test_r.shape
    r_flat    = np.abs(test_r).flatten()
    top_flat  = np.argsort(r_flat)[-top_n:][::-1]
    ch_idxs   = top_flat // B
    band_idxs = top_flat % B

    cols, meta = [], []
    for ch, bi in zip(ch_idxs, band_idxs):
        cols.append(feats[:, ch, bi])
        meta.append((int(ch), int(bi)))

    return np.column_stack(cols), meta


def summarise_selection(meta, good_idx, test_r):
    """Print ranked table of selected (channel, band) pairs."""
    print(f'\n  {"Rank":>4}  {"Orig ch":>7}  {"Band":>8}  {"r":>7}')
    print(f'  {"─"*4}  {"─"*7}  {"─"*8}  {"─"*7}')
    for rank, (ch, bi) in enumerate(meta):
        r_val = test_r[ch, bi]
        print(f'  {rank+1:>4}  {good_idx[ch]+1:>7}  {BAND_NAMES[bi]:>8}  {r_val:+.4f}')
