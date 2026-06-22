"""
Load and assemble per-subject trial data into a single dict.
"""

import numpy as np
import scipy.io as sio
from config import DATA_DIR, DECIM
from core.preprocessing import preprocess
from core.segmentation  import find_trial_boundaries
from core.features      import extract_features
from core.targets       import cursor_position, cursor_velocity_masked


def load_subject(s, method=None):
    """
    Full data pipeline for one subject up to feature/target arrays.

    Returns dict with keys:
      feats, pos, vel, vel_mask, trial_ids  — concatenated across trials
      good_idx, bad_mask, n_ch, n_trials
      elec, cx_raw, cy_raw

    method: line-noise removal method passed to preprocess(); None uses LINE_NOISE_METHOD from config.
    """
    d     = sio.loadmat(f'{DATA_DIR}/{s}_joystick.mat')
    x_raw = d['data'].astype(float)
    cx    = d['CursorPosX'].flatten()
    cy    = d['CursorPosY'].flatten()

    preprocess_kwargs = {} if method is None else {'method': method}
    x_car, good_idx, bad_mask, _ = preprocess(x_raw, **preprocess_kwargs)
    trials = find_trial_boundaries(cx, cy)

    # Filter the full continuous signal before segmenting into trials.
    # filtfilt applied to per-trial segments causes ringing at every trial
    # boundary; applying it to the full recording avoids those edge artifacts.
    feats_full = extract_features(x_car)   # (T_full // DECIM, C, 8)

    feats_l, pos_l, vel_l, mask_l, tid_l = [], [], [], [], []
    for tid, (ts, te) in enumerate(trials):
        f     = feats_full[ts // DECIM : te // DECIM]
        p     = cursor_position(cx[ts:te], cy[ts:te])
        v, vm = cursor_velocity_masked(cx[ts:te], cy[ts:te])
        T_    = min(f.shape[0], p.shape[0], v.shape[0])
        feats_l.append(f[:T_]); pos_l.append(p[:T_])
        vel_l.append(v[:T_]);   mask_l.append(vm[:T_])
        tid_l.append(np.full(T_, tid, dtype=int))

    return dict(
        feats     = np.concatenate(feats_l),
        pos       = np.concatenate(pos_l),
        vel       = np.concatenate(vel_l),
        vel_mask  = np.concatenate(mask_l),
        trial_ids = np.concatenate(tid_l),
        good_idx  = good_idx,
        bad_mask  = bad_mask,
        n_ch      = x_raw.shape[1],
        n_trials  = len(trials),
        elec      = d.get('electrodes', None),
        cx_raw    = cx,
        cy_raw    = cy,
    )
