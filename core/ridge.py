"""
Ridge regression decoder: single fold, k-fold CV, joint pos+vel.
"""

import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV
from config import LAGS, RIDGE_ALPHAS, TOP_N, N_FOLDS, SELECTION_METRIC
from core.metrics        import pearson_r_1d, r2 as _r2
from core.splits         import contiguous_kfold_splits
from core.channel_eval   import pearson_r_matrix
from core.channel_select import select_features, _ranking_score
from core.lag            import build_lag_matrix_by_trial


def _fit_predict(feats, tr_idx, te_idx, pos, trial_ids, model, top_n, lags,
                 metric=SELECTION_METRIC):
    """
    One CV fold: normalise → select → lag (per-trial) → regress → score.

    Returns
    -------
    pos_pred : (T_te, 2)  predicted cx, cy
    r_x, r_y : float  Pearson r per axis
    r2_x, r2_y : float  R² per axis
    coef     : (n_lag_features, 2)
    """
    tr_pos = pos[tr_idx]
    te_pos = pos[te_idx]

    mu     = feats[tr_idx].mean(axis=0, keepdims=True)
    sd     = feats[tr_idx].std(axis=0,  keepdims=True) + 1e-9
    z_full = (feats - mu) / sd

    per_axis  = [pearson_r_matrix(z_full[tr_idx], tr_pos[:, k])
                 for k in range(tr_pos.shape[1])]
    X_full, _ = select_features(z_full, _ranking_score(per_axis, metric), top_n)

    X_full_lag = build_lag_matrix_by_trial(X_full, trial_ids, lags)
    X_tr_lag   = X_full_lag[tr_idx]
    X_te_lag   = X_full_lag[te_idx]

    pos_mu   = tr_pos.mean(axis=0)
    pos_sd   = tr_pos.std(axis=0) + 1e-9
    tr_pos_z = (tr_pos - pos_mu) / pos_sd

    if model == 'ridge':
        reg_x = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_tr_lag, tr_pos_z[:, 0])
        reg_y = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_tr_lag, tr_pos_z[:, 1])
    elif model == 'lasso':
        reg_x = LassoCV(cv=3, max_iter=3000, n_alphas=30).fit(X_tr_lag, tr_pos_z[:, 0])
        reg_y = LassoCV(cv=3, max_iter=3000, n_alphas=30).fit(X_tr_lag, tr_pos_z[:, 1])
    else:
        raise ValueError(f'Unknown model: {model!r}')

    cx_pred  = reg_x.predict(X_te_lag) * pos_sd[0] + pos_mu[0]
    cy_pred  = reg_y.predict(X_te_lag) * pos_sd[1] + pos_mu[1]
    pos_pred = np.column_stack([cx_pred, cy_pred])
    coef     = np.column_stack([reg_x.coef_, reg_y.coef_])

    return (pos_pred,
            pearson_r_1d(te_pos[:, 0], cx_pred), pearson_r_1d(te_pos[:, 1], cy_pred),
            _r2(te_pos[:, 0], cx_pred),           _r2(te_pos[:, 1], cy_pred),
            coef)


def decode_cv(feats, target, trial_ids, model='ridge', top_n=TOP_N, lags=LAGS,
              n_folds=N_FOLDS, valid_mask=None, label='pos', metric=SELECTION_METRIC):
    """
    Contiguous k-fold decoder.

    trial_ids is used only for the lag matrix (lags never cross trial gaps).
    valid_mask (bool T-array) restricts each fold to valid samples when given.

    Returns
    -------
    pred      : (N, 2)
    true      : (N, 2)
    test_idx  : (N,)
    fold_r    : (n_folds, 2)  per-fold [r_x, r_y]
    fold_r2   : (n_folds, 2)  per-fold [R²_x, R²_y]
    fold_coef : list of (n_lag_features, 2)
    """
    T              = len(feats)
    target_buf     = np.full_like(target, np.nan)
    fold_r, fold_r2, fold_coef, all_te_idx = [], [], [], []

    for fold_i, (tr_idx, te_idx) in enumerate(contiguous_kfold_splits(T, k=n_folds)):
        if valid_mask is not None:
            tr_idx = tr_idx[valid_mask[tr_idx]]
            te_idx = te_idx[valid_mask[te_idx]]

        if len(tr_idx) < 50 or len(te_idx) < 10:
            print(f'    fold {fold_i+1}/{n_folds}  [{label}] skipped '
                  f'(tr={len(tr_idx)} te={len(te_idx)})')
            continue

        pp, r_x, r_y, r2_x, r2_y, coef = _fit_predict(
            feats, tr_idx, te_idx, target, trial_ids, model, top_n, lags, metric)
        target_buf[te_idx] = pp
        fold_r.append([r_x, r_y]); fold_r2.append([r2_x, r2_y])
        fold_coef.append(coef); all_te_idx.append(te_idx)
        n_str = f'  ({len(te_idx)} valid)' if valid_mask is not None else ''
        print(f'    fold {fold_i+1}/{n_folds}  [{label}]  '
              f'r=({r_x:+.3f},{r_y:+.3f})  R²=({r2_x:+.3f},{r2_y:+.3f}){n_str}')

    test_idx = np.concatenate(all_te_idx)
    return (target_buf[test_idx], target[test_idx], test_idx,
            np.array(fold_r), np.array(fold_r2), fold_coef)


def decode_cv_joint(feats, pos, vel, valid_mask, trial_ids,
                    top_n=TOP_N, lags=LAGS, n_folds=N_FOLDS, metric=SELECTION_METRIC):
    """
    Joint multi-output decoder: target = [cx, cy, vx, vy].
    One shared multi-output RidgeCV per fold; only valid (non-railed) samples used.

    Returns
    -------
    pos_pred, vel_pred, pos_true, vel_true : (N, 2) each
    test_idx   : (N,)
    fold_r_pos, fold_r_vel   : (n_folds, 2)
    fold_r2_pos, fold_r2_vel : (n_folds, 2)
    fold_coef  : list of (n_lag_features, 4)
    """
    T            = len(feats)
    joint_target = np.column_stack([pos, vel])
    pos_buf      = np.full_like(pos, np.nan)
    vel_buf      = np.full_like(vel, np.nan)
    fold_r_pos, fold_r_vel = [], []
    fold_r2_pos, fold_r2_vel = [], []
    fold_coef, all_te_idx = [], []

    for fold_i, (tr_idx, te_idx) in enumerate(contiguous_kfold_splits(T, k=n_folds)):
        tr_valid = tr_idx[valid_mask[tr_idx]]
        te_valid = te_idx[valid_mask[te_idx]]

        if len(tr_valid) < 50 or len(te_valid) < 10:
            print(f'    fold {fold_i+1}/{n_folds}  [joint] skipped'); continue

        mu     = feats[tr_valid].mean(axis=0, keepdims=True)
        sd     = feats[tr_valid].std(axis=0,  keepdims=True) + 1e-9
        z_full = (feats - mu) / sd

        per_axis  = [pearson_r_matrix(z_full[tr_valid], joint_target[tr_valid, k])
                     for k in range(joint_target.shape[1])]
        X_full, _ = select_features(z_full, _ranking_score(per_axis, metric), top_n)
        X_lag     = build_lag_matrix_by_trial(X_full, trial_ids, lags)
        X_tr      = X_lag[tr_valid]; X_te = X_lag[te_valid]

        tr_joint  = joint_target[tr_valid]
        target_mu = tr_joint.mean(axis=0); target_sd = tr_joint.std(axis=0) + 1e-9
        reg       = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_tr, (tr_joint - target_mu) / target_sd)
        pred      = reg.predict(X_te) * target_sd + target_mu

        pos_pred_fold = pred[:, :2]; vel_pred_fold = pred[:, 2:]
        te_pos        = joint_target[te_valid, :2]; te_vel = joint_target[te_valid, 2:]

        pos_buf[te_valid] = pos_pred_fold; vel_buf[te_valid] = vel_pred_fold
        all_te_idx.append(te_valid)

        r_cx  = pearson_r_1d(te_pos[:, 0], pos_pred_fold[:, 0])
        r_cy  = pearson_r_1d(te_pos[:, 1], pos_pred_fold[:, 1])
        r_vx  = pearson_r_1d(te_vel[:, 0], vel_pred_fold[:, 0])
        r_vy  = pearson_r_1d(te_vel[:, 1], vel_pred_fold[:, 1])
        r2_cx = _r2(te_pos[:, 0], pos_pred_fold[:, 0])
        r2_cy = _r2(te_pos[:, 1], pos_pred_fold[:, 1])
        r2_vx = _r2(te_vel[:, 0], vel_pred_fold[:, 0])
        r2_vy = _r2(te_vel[:, 1], vel_pred_fold[:, 1])

        fold_r_pos.append([r_cx, r_cy]); fold_r2_pos.append([r2_cx, r2_cy])
        fold_r_vel.append([r_vx, r_vy]); fold_r2_vel.append([r2_vx, r2_vy])
        fold_coef.append(reg.coef_.T)

        print(f'    fold {fold_i+1}/{n_folds}  [joint]  '
              f'pos r=({r_cx:+.3f},{r_cy:+.3f}) R²=({r2_cx:+.3f},{r2_cy:+.3f})  '
              f'vel r=({r_vx:+.3f},{r_vy:+.3f}) R²=({r2_vx:+.3f},{r2_vy:+.3f})  '
              f'α={reg.alpha_:.0e}  ({len(te_valid)} valid)')

    test_idx = np.concatenate(all_te_idx)
    return (pos_buf[test_idx], vel_buf[test_idx],
            joint_target[test_idx, :2], joint_target[test_idx, 2:],
            test_idx,
            np.array(fold_r_pos), np.array(fold_r_vel),
            np.array(fold_r2_pos), np.array(fold_r2_vel),
            fold_coef)
