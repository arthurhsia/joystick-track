"""
Neural feature extraction: 7 power envelopes + LMP → (T_down, C, 8).
"""

import numpy as np
import scipy.signal as sig
from config import DECIM, BANDS
from core.filters import BP_FILTERS, B_LP, A_LP, B_LMP, A_LMP

N_FEATS = len(BANDS) + 1   # 7 envelopes + 1 LMP


def extract_features(x_car):
    """
    (T, C) ECoG at FS Hz → (T//DECIM, C, 8) float32 at FS_FEAT Hz.

    Bands 0–6: bandpass → |·| → 1.5 Hz LPF → decimate.
    Band 7 (LMP): 1.5 Hz LPF only, signed, no rectification.
    """
    T, C   = x_car.shape
    T_down = T // DECIM
    feats  = np.empty((T_down, C, N_FEATS), dtype=np.float32)

    for bi, (b_bp, a_bp) in enumerate(BP_FILTERS):
        for ch in range(C):
            bp       = sig.filtfilt(b_bp, a_bp, x_car[:, ch])
            envelope = sig.filtfilt(B_LP, A_LP, np.abs(bp))
            feats[:, ch, bi] = envelope[:T_down * DECIM:DECIM]

    lmp_bi = len(BANDS)
    for ch in range(C):
        lmp = sig.filtfilt(B_LMP, A_LMP, x_car[:, ch])
        feats[:, ch, lmp_bi] = lmp[:T_down * DECIM:DECIM]

    return feats
