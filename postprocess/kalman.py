"""
Per-axis constant-velocity Kalman filter with σ_a grid search and velocity gating.

The filter is forward-only (causal). This is appropriate for real-time BCI analysis.
For strictly better offline reconstruction an RTS backward smoothing pass could be
added after the forward pass — left as a future extension.
"""

import numpy as np
from config import FS_FEAT
from core.metrics import pearson_r_1d

DT         = 1.0 / FS_FEAT          # 0.1 s
SIGMA_GRID = np.geomspace(1e4, 1e8, 20)


def build_Q_1d(sigma_a, dt=DT):
    """Process noise covariance for constant-velocity model."""
    return sigma_a ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2],
                                     [dt ** 3 / 2, dt ** 2]])


def r_cov(pred, true):
    """
    Diagonal of empirical residual variance for two-column arrays (R_pos_x, R_pos_y).

    NOTE: Uses test-set residuals (pred and true are both held-out data). Train-fold
    R estimation would require per-fold in-sample predictions not currently returned
    by decode_cv. This is a known limitation; the resulting R values are used only as
    fixed noise-model parameters, not for hyperparameter selection, so the bias is small.
    """
    resid = pred - true
    return float(np.var(resid[:, 0])), float(np.var(resid[:, 1]))


def kalman_1d(pos_pred, vel_pred, R_pos, R_vel, sigma_a, dt=DT, trial_ids=None):
    """
    2-state (position, velocity) constant-velocity Kalman for one axis.
    Forward-only (causal).

    Always fuses the position observation.
    Fuses the velocity observation only when vel_pred[t] is not NaN.

    State is reset to [pos_pred[t], 0] at the first sample of each new trial
    (trial_ids[t] != trial_ids[t-1]) to prevent inter-trial velocity transients.

    Parameters
    ----------
    pos_pred  : (T,)           ridge position prediction
    vel_pred  : (T,)           ridge velocity prediction; NaN where railed
    R_pos     : float          position observation noise variance
    R_vel     : float          velocity observation noise variance
    sigma_a   : float          acceleration process noise std dev
    dt        : float          sample period (default 0.1 s)
    trial_ids : (T,) int|None  trial identifier per sample; resets filter at boundaries
    """
    T  = len(pos_pred)
    F  = np.array([[1.0, dt ], [0.0, 1.0]])
    Hp = np.array([[1.0, 0.0]])
    Hf = np.array([[1.0, 0.0], [0.0, 1.0]])
    Q  = build_Q_1d(sigma_a, dt)
    Rp = np.array([[R_pos]])
    Rf = np.array([[R_pos, 0.0], [0.0, R_vel]])
    P0 = np.diag([R_pos * 10, R_vel * 100])

    x   = np.array([pos_pred[0], 0.0])
    P   = P0.copy()
    out = np.empty(T)

    for t in range(T):
        if trial_ids is not None and t > 0 and trial_ids[t] != trial_ids[t - 1]:
            # Trial boundary: re-initialise state from current observation
            x = np.array([pos_pred[t], 0.0])
            P = P0.copy()
        else:
            x = F @ x
            P = F @ P @ F.T + Q

        if not np.isnan(vel_pred[t]):
            z = np.array([pos_pred[t], vel_pred[t]]); H, R = Hf, Rf
        else:
            z = np.array([pos_pred[t]]); H, R = Hp, Rp

        S   = H @ P @ H.T + R
        K   = P @ H.T @ np.linalg.solve(S, np.eye(len(z)))
        x   = x + K @ (z - H @ x)
        P   = (np.eye(2) - K @ H) @ P
        out[t] = x[0]

    return out


def sigma_from_bandwidth(f_target, R_pos, dt=DT):
    """
    Compute σ_a so the Kalman effective bandwidth ≈ f_target Hz.

    Inverts: f_eff = σ_a · dt / (4π√R_pos)
    → σ_a = f_target · 4π√R_pos / dt
    """
    return f_target * 4.0 * np.pi * np.sqrt(R_pos) / dt


def grid_search_sigma(pos_pred, vel_pred, pos_true, R_pos, R_vel,
                      sigma_grid=SIGMA_GRID, trial_ids=None):
    """
    Oracle diagnostic: find σ_a maximising test-set Pearson r for one axis.

    Returns (best_sigma, best_r, sigma_grid, all_rs)
    """
    rs = np.array([
        pearson_r_1d(pos_true,
                     kalman_1d(pos_pred, vel_pred, R_pos, R_vel, s,
                               trial_ids=trial_ids))
        for s in sigma_grid
    ])
    i = rs.argmax()
    return sigma_grid[i], rs[i], sigma_grid, rs


def best_kalman_axis(pos_pred, vel_pred, pos_true, R_pos, R_vel,
                     sigma_grid=SIGMA_GRID, trial_ids=None):
    """
    Oracle diagnostic: grid-search σ_a with and without velocity; return test-set winner.

    Both σ_a selection and use_vel decision are made by maximising pos_true correlation —
    this is test-set tuning and should not be used as a reported result.

    Returns (best_sigma, best_r, use_vel, sigma_grid, rs_pos_vel, rs_pos_only)
    """
    no_vel = np.full_like(vel_pred, np.nan)
    s_pv, r_pv, g, rs_pv = grid_search_sigma(
        pos_pred, vel_pred, pos_true, R_pos, R_vel, sigma_grid, trial_ids=trial_ids)
    s_po, r_po, _, rs_po = grid_search_sigma(
        pos_pred, no_vel,   pos_true, R_pos, R_vel, sigma_grid, trial_ids=trial_ids)
    if r_pv >= r_po:
        return s_pv, r_pv, True,  g, rs_pv, rs_po
    else:
        return s_po, r_po, False, g, rs_pv, rs_po
