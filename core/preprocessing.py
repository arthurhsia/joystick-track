"""
Preprocessing: line-noise removal → iterative bad-channel removal + CAR.

Bad-channel detection follows PREP's statistical core — robust per-channel
amplitude estimator, reference-invariant V_raw criteria, collapsed robust
reference loop — without PREP's EEG-specific spatial machinery (no
interpolation, no RANSAC, no correlation criteria).

Data lineage per criterion
--------------------------
V_raw    = x_raw.astype(float)               pre-removal
V_notch  = remove_line_noise(V_raw, method)  line-noise-removed master — never modified in-place
V_detect = HP-copy(V_notch)                  throwaway; amplitude only; LMP stays in V_notch

Line-noise removal methods (LINE_NOISE_METHOD in config):
  'notch'        — IIR notch + filtfilt; zero-phase but creates spectral holes at 60/120/180 Hz
  'spectrum_fit' — PREP/CleanLine multi-taper spectral fit via mne; subtracts only the sinusoidal
                   component so no spectral hole; inherently zero-phase (no filtfilt wrapper).

The line-noise SNR bad-channel criterion always reads V_raw (pre-removal) regardless of method.
"""

import numpy as np
import scipy.signal as sig
from config import (FS, FLAT_AMP_THRESH, AMP_Z_THRESH, LINE_NOISE_Z_THRESH, LINE_NOISE_METHOD)


# ── Notch filter ──────────────────────────────────────────────────────────────

def _build_notch(freqs=(60, 120, 180), Q=35):
    b, a = np.array([1.0]), np.array([1.0])
    for f0 in freqs:
        bn, an = sig.iirnotch(f0, Q, FS)
        b, a = np.convolve(b, bn), np.convolve(a, an)
    return b, a

_B_NOTCH, _A_NOTCH = _build_notch()


def remove_line_noise(x, fs=FS, freqs=(60, 120, 180), method='notch', q=35):
    """
    Remove line noise from a (T, n_ch) array.

    method='notch'        : IIR notch + filtfilt — creates spectral holes at each harmonic.
    method='spectrum_fit' : PREP/CleanLine multi-taper spectral fit (mne); subtracts only the
                            sinusoidal component, so no spectral hole.  Inherently zero-phase —
                            do NOT wrap in filtfilt.

    The line-noise SNR bad-channel criterion must read the original pre-removal signal;
    never pass the return value of this function to _line_noise_snr.
    """
    if method == 'notch':
        b, a = _build_notch(freqs, q)
        return sig.filtfilt(b, a, x, axis=0)
    elif method == 'spectrum_fit':
        import mne
        xt = mne.filter.notch_filter(
            x.T.astype(float), fs, np.asarray(freqs),
            method='spectrum_fit', p_value=0.05,
            filter_length='10s', verbose=False)
        return xt.T
    else:
        raise ValueError(f"Unknown line-noise method: {method!r}. Use 'notch' or 'spectrum_fit'.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _line_noise_snr(x_raw, harmonics=(60, 120, 180), bw_hz=2, max_s=120):
    """Mean harmonic-peak SNR per channel, computed on pre-notch data."""
    T = min(x_raw.shape[0], int(max_s * FS))
    f, pxx = sig.welch(x_raw[:T], fs=FS, nperseg=FS * 4, axis=0)  # (n_freqs, C)
    snr = np.zeros(x_raw.shape[1])
    for h in harmonics:
        idx_h     = np.argmin(np.abs(f - h))
        base_mask = (f >= h - 10) & (f <= h + 10) & ~((f >= h - bw_hz) & (f <= h + bw_hz))
        base      = np.median(pxx[base_mask], axis=0)
        snr      += pxx[idx_h] / (base + 1e-30)
    return snr / len(harmonics)


def _mad_z(x):
    """Modified z-score (median/MAD) — robust to masking by outliers."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 0.6745 * (x - med) / (mad + 1e-9)


def _channel_amplitude(x):
    """IQR × 0.7413 per channel — PREP find_bad_by_deviation estimator."""
    return 0.7413 * (np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0))



def _build_detect_copy(x_notch, detect_hp_hz=2.0):
    """
    V_detect: a throwaway copy of x_notch for amplitude detection only.

    High-passes at detect_hp_hz (Butterworth order 4, filtfilt) to remove
    DC, slow drift, and the LMP band so strong-LMP channels are not
    false-positived on amplitude. If detect_hp_hz is None, falls back to
    scipy.signal.detrend(type='linear').

    Returns a new array; x_notch is never modified.
    """
    x = x_notch.copy()
    if detect_hp_hz is not None:
        b, a = sig.butter(4, detect_hp_hz / (FS / 2.0), btype='high')
        return sig.filtfilt(b, a, x, axis=0)
    return sig.detrend(x, type='linear', axis=0)


# ── Single-pass detection ─────────────────────────────────────────────────────

def detect_bad_channels(
    x_raw,
    x_notch,
    line_noise_snr=None,
    z_thresh=AMP_Z_THRESH,
    flat_floor=FLAT_AMP_THRESH,
    detect_hp_hz=2.0,
):
    """
    Single-pass bad-channel detection. Returns (bad_mask, reasons).

    Criteria and data versions:
        flat       ← V_raw   NaN/inf present or std < flat_floor
        amplitude  ← V_detect = HP copy of x_notch; |z| > z_thresh (two-sided)
        line_noise ← V_raw   via pre-computed line_noise_snr

    Parameters
    ----------
    x_raw         : (T, n_ch) pre-notch float data
    x_notch       : (T, n_ch) post-notch float data  — NOT modified in-place
    line_noise_snr: (n_ch,) pre-computed per-channel SNR (from _line_noise_snr)

    Returns
    -------
    bad_mask : (n_ch,) bool
    reasons  : dict[str → (n_ch,) bool]  per-criterion breakdown
    """
    n_ch = x_raw.shape[1]

    # V_detect: throwaway high-passed copy for amplitude only
    x_det = _build_detect_copy(x_notch, detect_hp_hz)

    # 0. flat / dead (V_raw) — run first; near-zero channels must not corrupt amp z-scores
    flat = (
        np.any(~np.isfinite(x_raw), axis=0) |
        (x_raw.std(axis=0) < flat_floor)
    )

    # 1. amplitude deviation, two-sided (V_detect)
    bad_amp = np.abs(_mad_z(_channel_amplitude(x_det))) > z_thresh

    # 2. line-noise SNR (V_raw, pre-computed before notch)
    if line_noise_snr is not None and LINE_NOISE_Z_THRESH is not None:
        bad_snr = _mad_z(np.log1p(line_noise_snr)) > z_thresh
    else:
        bad_snr = np.zeros(n_ch, bool)

    reasons = dict(flat=flat, amplitude=bad_amp, line_noise=bad_snr)
    return flat | bad_amp | bad_snr, reasons


# ── Diagnostic ────────────────────────────────────────────────────────────────

def diagnose_channels(
    x_raw,
    z_thresh=AMP_Z_THRESH,
    flat_floor=FLAT_AMP_THRESH,
    detect_hp_hz=2.0,
    method=LINE_NOISE_METHOD,
):
    """
    Per-channel diagnostic report. Compares amplitude z-score with and without
    the detection HP to show which channels are rescued (strong LMP, would have
    been false-positived without HP) and which are only caught with HP.

    Returns dict of (n_ch,) arrays:
        amp_z_detrended   : z-score used in actual detection (HP copy)
        amp_z_undetrended : z-score without HP (linear detrend only)
        lnr_z             : line-noise SNR modified z-score
        flat              : flat/dead flag
        fired_*           : bool per criterion
        rescued_by_hp     : flagged only without HP (strong-LMP channels saved)
        newly_caught_by_hp: flagged only with HP
    """
    x_float = x_raw.astype(float)
    lnr     = _line_noise_snr(x_float)
    x_notch = remove_line_noise(x_float, method=method)

    x_det       = _build_detect_copy(x_notch, detect_hp_hz)
    x_nodetrend = _build_detect_copy(x_notch, None)   # linear detrend, no HP

    amp_z_hp  = _mad_z(_channel_amplitude(x_det))
    amp_z_raw = _mad_z(_channel_amplitude(x_nodetrend))
    lnr_z     = _mad_z(np.log1p(lnr))
    flat_flag = (np.any(~np.isfinite(x_float), axis=0) | (x_float.std(0) < flat_floor))

    flag_hp  = np.abs(amp_z_hp)  > z_thresh
    flag_raw = np.abs(amp_z_raw) > z_thresh

    return {
        'amp_z_detrended':     amp_z_hp,
        'amp_z_undetrended':   amp_z_raw,
        'lnr_z':               lnr_z,
        'flat':                flat_flag,
        'fired_flat':          flat_flag,
        'fired_amplitude':     flag_hp,
        'fired_line_noise':    lnr_z > z_thresh,
        'rescued_by_hp':       flag_raw & ~flag_hp,
        'newly_caught_by_hp':  flag_hp  & ~flag_raw,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def preprocess(
    x_raw,
    z_thresh=AMP_Z_THRESH,
    flat_floor=FLAT_AMP_THRESH,
    detect_hp_hz=2.0,
    max_passes=4,
    method=LINE_NOISE_METHOD,
):
    """
    Line-noise removal → iterative bad-channel removal + CAR.

    V_raw criteria (flat, line-noise) are reference-invariant and run once.
    The amplitude criterion iterates: after each CAR the distribution shifts
    and may expose new bad channels previously masked by a contaminated
    reference (PREP's collapsed robust reference). Capped at max_passes.

    Parameters
    ----------
    x_raw        : (T, n_ch) raw int16-valued ECoG
    z_thresh     : modified-z threshold; Iglewicz-Hoaglin default 3.5,
                   PREP uses 5.0 for its amplitude/HF criteria
    flat_floor   : channel std floor for flat/dead criterion (int16 units)
    detect_hp_hz : HP cutoff for V_detect; None = linear detrend fallback
    max_passes   : max CAR + re-detection iterations
    method       : line-noise removal — 'notch' (default) or 'spectrum_fit'

    Returns
    -------
    x_car    : (T, n_good) CAR-referenced float64  [LMP fully intact]
    good_idx : (n_good,) original indices of kept channels
    bad_mask : (n_ch,) True = removed
    reasons  : dict[str → (n_ch,) bool] per-criterion breakdown
    """
    x_float = x_raw.astype(float)
    n_ch    = x_float.shape[1]

    # Line-noise SNR on V_raw (computed once; reference-invariant)
    lnr = _line_noise_snr(x_float)

    # V_notch: master signal — never modified in-place
    x_notch         = remove_line_noise(x_float, method=method)
    _notch_checksum = float(x_notch.sum())   # for LMP-preservation assertion

    # V_raw criteria: run once (flat, line-noise do not shift with reference)
    flat    = np.any(~np.isfinite(x_float), axis=0) | (x_float.std(0) < flat_floor)
    bad_snr = (_mad_z(np.log1p(lnr)) > z_thresh
               if LINE_NOISE_Z_THRESH is not None else np.zeros(n_ch, bool))

    reasons = dict(
        flat       = flat.copy(),
        amplitude  = np.zeros(n_ch, bool),
        line_noise = bad_snr.copy(),
    )
    all_bad = flat | bad_snr

    # Iterative amplitude detection + collapsed robust re-referencing
    for _ in range(max_passes):
        active = ~all_bad
        if not active.any():
            break
        x_active = x_notch[:, active]
        x_car    = x_active - x_active.mean(axis=1, keepdims=True)

        x_det   = _build_detect_copy(x_car, detect_hp_hz)
        bad_amp = np.abs(_mad_z(_channel_amplitude(x_det))) > z_thresh

        if not bad_amp.any():
            break

        active_idx               = np.where(active)[0]
        new_bad                  = active_idx[bad_amp]
        all_bad[new_bad]         = True
        reasons['amplitude'][new_bad] = True

    # LMP-preservation assertion: V_notch must not have been touched in-place
    assert float(x_notch.sum()) == _notch_checksum, (
        "LMP preservation violated: V_notch was modified in-place during detection"
    )

    good_idx = np.where(~all_bad)[0]
    x_clean  = x_notch[:, good_idx]
    x_out    = x_clean - x_clean.mean(axis=1, keepdims=True)

    return x_out, good_idx, all_bad, reasons
