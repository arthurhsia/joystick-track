"""
Diagnostic: shared vs per-axis channel selection.

Current pipeline: top-10 by |mean(r_cx, r_cy)| → same feature set for cx and cy decoders.
Alternative:      top-10 by |r_cx| for cx decoder, top-10 by |r_cy| for cy decoder.

If per-axis cy jumps while cx holds → shared selection was bottlenecking cy.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from sklearn.linear_model import RidgeCV

from config import LAGS, RIDGE_ALPHAS, TOP_N, N_FOLDS, SUBJECTS
from pipeline.loader      import load_subject
from core.splits          import contiguous_kfold_splits
from core.channel_eval    import pearson_r, pearson_r_matrix
from core.channel_select  import select_features
from core.lag             import build_lag_matrix_by_trial
from core.metrics         import pearson_r_1d


def decode_fold(z_full, tr_idx, te_idx, pos, trial_ids, tr_r_cx, tr_r_cy,
                mode='shared'):
    """
    mode='shared'   : top-N by |mean(r_cx, r_cy)|, same X for both axes
    mode='per_axis' : top-N by |r_cx| for cx, top-N by |r_cy| for cy
    """
    if mode == 'shared':
        tr_r_mean  = (tr_r_cx + tr_r_cy) / 2        # signed mean
        X, _       = select_features(z_full, tr_r_mean, TOP_N)
        X_lag      = build_lag_matrix_by_trial(X, trial_ids, LAGS)
        X_tr, X_te = X_lag[tr_idx], X_lag[te_idx]

        tr_pos = pos[tr_idx]; te_pos = pos[te_idx]
        mu_p = tr_pos.mean(0); sd_p = tr_pos.std(0) + 1e-9
        tr_pz = (tr_pos - mu_p) / sd_p

        reg_x = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_tr, tr_pz[:, 0])
        reg_y = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_tr, tr_pz[:, 1])
        cx_pred = reg_x.predict(X_te) * sd_p[0] + mu_p[0]
        cy_pred = reg_y.predict(X_te) * sd_p[1] + mu_p[1]

    else:  # per_axis
        tr_pos = pos[tr_idx]; te_pos = pos[te_idx]
        mu_p = tr_pos.mean(0); sd_p = tr_pos.std(0) + 1e-9
        tr_pz = (tr_pos - mu_p) / sd_p

        # cx: select on |r_cx|, build its own lag matrix
        X_cx, _ = select_features(z_full, tr_r_cx, TOP_N)
        Xl_cx   = build_lag_matrix_by_trial(X_cx, trial_ids, LAGS)
        reg_x   = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xl_cx[tr_idx], tr_pz[:, 0])
        cx_pred = reg_x.predict(Xl_cx[te_idx]) * sd_p[0] + mu_p[0]

        # cy: select on |r_cy|, build its own lag matrix
        X_cy, _ = select_features(z_full, tr_r_cy, TOP_N)
        Xl_cy   = build_lag_matrix_by_trial(X_cy, trial_ids, LAGS)
        reg_y   = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xl_cy[tr_idx], tr_pz[:, 1])
        cy_pred = reg_y.predict(Xl_cy[te_idx]) * sd_p[1] + mu_p[1]

    r_cx = pearson_r_1d(te_pos[:, 0], cx_pred)
    r_cy = pearson_r_1d(te_pos[:, 1], cy_pred)
    return r_cx, r_cy


# ── Run ───────────────────────────────────────────────────────────────────────
print(f'\n{"─"*68}')
print(f'  {"Subj":4}  {"Mode":10}  '
      f'{"cx r":>7} {"cy r":>7}  '
      f'{"Δcx":>7} {"Δcy":>7}')
print(f'{"─"*68}')

for s in SUBJECTS:
    d  = load_subject(s)
    feats, pos, trial_ids = d['feats'], d['pos'], d['trial_ids']
    T  = feats.shape[0]

    shared_cx, shared_cy = [], []
    perax_cx,  perax_cy  = [], []

    for tr_idx, te_idx in contiguous_kfold_splits(T, k=N_FOLDS):
        mu = feats[tr_idx].mean(0, keepdims=True)
        sd = feats[tr_idx].std(0,  keepdims=True) + 1e-9
        z  = (feats - mu) / sd

        # Per-axis r matrices (on train only)
        tr_r_cx = pearson_r_matrix(z[tr_idx], pos[tr_idx, 0])  # (C, B)
        tr_r_cy = pearson_r_matrix(z[tr_idx], pos[tr_idx, 1])  # (C, B)

        rx, ry = decode_fold(z, tr_idx, te_idx, pos, trial_ids,
                             tr_r_cx, tr_r_cy, mode='shared')
        shared_cx.append(rx); shared_cy.append(ry)

        rx, ry = decode_fold(z, tr_idx, te_idx, pos, trial_ids,
                             tr_r_cx, tr_r_cy, mode='per_axis')
        perax_cx.append(rx); perax_cy.append(ry)

    s_cx = np.mean(shared_cx); s_cy = np.mean(shared_cy)
    p_cx = np.mean(perax_cx);  p_cy = np.mean(perax_cy)

    print(f'  {s.upper():4}  {"shared":10}  {s_cx:+7.4f} {s_cy:+7.4f}')
    print(f'  {s.upper():4}  {"per-axis":10}  {p_cx:+7.4f} {p_cy:+7.4f}  '
          f'{p_cx-s_cx:+7.4f} {p_cy-s_cy:+7.4f}')
    print(f'  {"":4}  {"":10}  {"":7} {"":7}')

print(f'{"─"*68}')
