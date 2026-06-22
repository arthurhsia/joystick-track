"""
Ridge alpha sweep diagnostic.
For each subject: fit Ridge at every alpha across all LOTO folds and plot
mean test Pearson r vs alpha.  Also prints which alpha RidgeCV selected per fold.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV

from config import DATA_DIR, OUT_DIR, SUBJECTS, COLORS, LAGS, TOP_N, RIDGE_ALPHAS
from core.preprocessing  import preprocess
from core.features       import extract_features
from core.targets        import cursor_position
from core.segmentation   import find_trial_boundaries
from core.splits         import trial_loto_splits
from core.channel_eval   import pearson_r
from core.channel_select import select_features
from core.lag            import build_lag_matrix_by_trial
from core.metrics        import pearson_r_1d

ALPHAS = np.logspace(1, 14, 40)   # finer grid: 1e1 → 1e14


fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Ridge alpha sweep — mean LOTO test Pearson r vs regularisation strength\n'
             'mean(cx, cy)  |  vertical line = alpha chosen by RidgeCV',
             fontsize=12, fontweight='bold')

for ax, s in zip(axes.flat, SUBJECTS):
    print(f'\n{s.upper()}')
    d     = sio.loadmat(f'{DATA_DIR}/{s}_joystick.mat')
    x_raw = d['data'].astype(float)
    cx    = d['CursorPosX'].flatten()
    cy    = d['CursorPosY'].flatten()

    x_car, good_idx, _ = preprocess(x_raw)
    trials    = find_trial_boundaries(cx, cy)
    feats_l, pos_l, tid_l = [], [], []
    for tid, (ts, te) in enumerate(trials):
        f = extract_features(x_car[ts:te])
        p = cursor_position(cx[ts:te], cy[ts:te])
        T_ = min(f.shape[0], p.shape[0])
        feats_l.append(f[:T_]); pos_l.append(p[:T_])
        tid_l.append(np.full(T_, tid, dtype=int))

    feats     = np.concatenate(feats_l)
    pos       = np.concatenate(pos_l)
    trial_ids = np.concatenate(tid_l)

    fold_r_mat    = np.zeros((len(trials), len(ALPHAS), 2))
    chosen_alphas = []

    for fold_i, (tr_idx, te_idx) in enumerate(trial_loto_splits(trial_ids)):
        tr_pos = pos[tr_idx]; te_pos = pos[te_idx]

        mu = feats[tr_idx].mean(0, keepdims=True)
        sd = feats[tr_idx].std(0,  keepdims=True) + 1e-9
        z  = (feats - mu) / sd

        tr_r   = pearson_r(z[tr_idx], tr_pos)
        X_, _  = select_features(z, tr_r, TOP_N)
        X_lag  = build_lag_matrix_by_trial(X_, trial_ids, LAGS)
        X_tr   = X_lag[tr_idx]; X_te = X_lag[te_idx]

        mu_p = tr_pos.mean(0); sd_p = tr_pos.std(0) + 1e-9
        tr_z = (tr_pos - mu_p) / sd_p

        for ai, alpha in enumerate(ALPHAS):
            for ci in range(2):
                reg  = Ridge(alpha=alpha, solver='svd').fit(X_tr, tr_z[:, ci])
                pred = reg.predict(X_te) * sd_p[ci] + mu_p[ci]
                fold_r_mat[fold_i, ai, ci] = pearson_r_1d(te_pos[:, ci], pred)

        rcv = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_tr, tr_z[:, 0])
        chosen_alphas.append(rcv.alpha_)
        print(f'  fold {fold_i+1}/{len(trials)}  RidgeCV chose alpha={rcv.alpha_:.1e}')

    mean_r_curve = fold_r_mat.mean(axis=(0, 2))

    ax.semilogx(ALPHAS, mean_r_curve, color=COLORS[s], lw=2)
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)

    best_ai  = mean_r_curve.argmax()
    best_a   = ALPHAS[best_ai]
    best_r   = mean_r_curve[best_ai]
    ax.axvline(best_a, color=COLORS[s], lw=1.5, ls=':',
               label=f'best α={best_a:.1e}  r={best_r:.4f}')

    med_chosen = np.median(chosen_alphas)
    ax.axvline(med_chosen, color='gray', lw=1, ls='--',
               label=f'RidgeCV median={med_chosen:.1e}')

    ax.set_title(s.upper())
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Mean test r  (cx+cy, all folds)')
    ax.legend(fontsize=8)
    print(f'  best alpha={best_a:.2e}  mean r={best_r:.4f}  '
          f'RidgeCV picks (median)={med_chosen:.2e}')

plt.tight_layout()
out = f'{OUT_DIR}/figS1_alpha_sweep.png'
plt.savefig(out, dpi=150)
plt.close()
print(f'\nSaved {out}')
