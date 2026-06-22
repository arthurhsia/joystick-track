"""
Filter bank constants: bandpass filters for 7 bands + LPF for envelopes and LMP.
Built once at import time and shared by feature extraction.
"""

import scipy.signal as sig
from config import FS, SMOOTH_HZ, BANDS, LMP_HZ


def _bandpass(lo, hi, order):
    nyq = FS / 2
    return sig.butter(order, [max(lo / nyq, 1e-3), min(hi / nyq, 0.98)], btype='band')


def _lowpass(cutoff, order=4):
    return sig.butter(order, cutoff / (FS / 2), btype='low')


BP_FILTERS    = [_bandpass(lo, hi, order) for _, lo, hi, order in BANDS]
B_LP, A_LP    = _lowpass(SMOOTH_HZ)   # envelope smoother
B_LMP, A_LMP  = _lowpass(LMP_HZ)     # LMP smoother (signed, no rectification)
