"""
Behavioral targets: cursor position decimation and velocity with rail masking.
"""

import numpy as np
import scipy.signal as sig
from scipy.ndimage import binary_dilation
from config import FS, DECIM, RAIL_TOL, RAIL_PAD_MS
from core.filters import B_LP, A_LP


def cursor_position(cx, cy):
    """
    Decimate cursor position to FS_FEAT Hz.

    Returns
    -------
    pos : (T_down, 2)  columns [cx, cy] at FS_FEAT Hz
    """
    T_down = len(cx) // DECIM
    return np.stack([
        cx[:T_down * DECIM:DECIM].astype(np.float64),
        cy[:T_down * DECIM:DECIM].astype(np.float64),
    ], axis=1)


def cursor_velocity_masked(cx, cy, rail_tol=RAIL_TOL, pad_ms=RAIL_PAD_MS):
    """
    Per-segment velocity at FS_FEAT Hz with rail masking.

    1. Flag samples within rail_tol of int16 boundary [0, 32767].
    2. Dilate railed epochs by pad_ms.
    3. Within each valid run: diff → 1.5 Hz LPF (never across gaps).
    4. Decimate by DECIM.

    Returns
    -------
    vel_down   : (T_down, 2)  float64, NaN where masked  (AU/s)
    valid_down : (T_down,)    bool — True where velocity is defined
    """
    cx = cx.astype(np.float64)
    cy = cy.astype(np.float64)
    T  = len(cx)

    railed = ((cx <= rail_tol) | (cx >= 32767 - rail_tol) |
              (cy <= rail_tol) | (cy >= 32767 - rail_tol))

    pad_samp = int(pad_ms * FS / 1000)
    if pad_samp > 0:
        railed = binary_dilation(railed, structure=np.ones(2 * pad_samp + 1, bool))
    valid = ~railed

    vx = np.full(T, np.nan)
    vy = np.full(T, np.nan)
    edges  = np.where(np.diff(valid.astype(np.int8), prepend=0, append=0))[0]
    starts = edges[0::2]
    ends   = edges[1::2]

    min_filt = 3 * (max(len(B_LP), len(A_LP)) - 1) + 1

    for s, e in zip(starts, ends):
        seg_len = e - s
        if seg_len < 2:
            continue
        dvx = np.diff(cx[s:e]) * FS
        dvy = np.diff(cy[s:e]) * FS
        if seg_len >= min_filt:
            dvx = sig.filtfilt(B_LP, A_LP, dvx)
            dvy = sig.filtfilt(B_LP, A_LP, dvy)
        vx[s]     = dvx[0]; vx[s + 1:e] = dvx
        vy[s]     = dvy[0]; vy[s + 1:e] = dvy

    T_down     = T // DECIM
    idx        = np.arange(T_down) * DECIM
    vel_down   = np.stack([vx[idx], vy[idx]], axis=1)
    valid_down = valid[idx]

    return vel_down, valid_down
