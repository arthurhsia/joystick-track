"""
Task 3 — threshold-justification analysis.

For each subject, sweeps the modified-z threshold (amplitude + line-noise
criteria jointly) from 3.0 to 5.0 in 0.25-step increments, and records:

  • the flagged set at each threshold
  • stable channels (always or never flagged across the range)
  • marginal channels (flip somewhere inside the range)
  • PSD-shape classification for every marginal / stably-bad channel
  • decoder impact: mean Pearson r with vs without each marginal channel
    (requires one full preprocessing + feature extraction per subject; use
    --no-decode to skip)

FP ch18 (amp_z ≈ −3.8) is called out explicitly as a known marginal case.

Output:
  stdout    — per-subject tables
  figs/thresh_justification_{s}.csv  — short summary table per subject

Usage:
    python diag/threshold_justification.py             # full run with decoder
    python diag/threshold_justification.py --no-decode # fast, skip decoder
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import scipy.io as sio
import scipy.signal as sig
import scipy.stats as stats_mod

from config import (FS, DATA_DIR, OUT_DIR, SUBJECTS, DECIM,
                    FLAT_AMP_THRESH, LINE_NOISE_Z_THRESH)
from core.preprocessing import (_B_NOTCH, _A_NOTCH, _line_noise_snr, _mad_z,
                                 _channel_amplitude, _build_detect_copy, preprocess)
from core.segmentation  import find_trial_boundaries
from core.features      import extract_features
from core.targets       import cursor_position
from core.ridge         import decode_cv

THRESH_SWEEP = np.round(np.arange(3.0, 5.01, 0.25), 2)


# ── PSD classifier ────────────────────────────────────────────────────────────

def _psd_classify(col, fs=FS):
    """
    Classify a channel's raw-data PSD shape.

    Returns one of:
      'normal 1/f + neural structure'
      'normal 1/f'
      'flat/structureless'
      'elevated-HF (noise-dominated)'
      'weak 1/f'
    """
    f, pxx = sig.welch(col, fs=fs, nperseg=fs * 4)

    # Log-log slope over 5–200 Hz (avoids DC and above recording BPF rolloff)
    mask = (f >= 5) & (f <= 200)
    lf   = np.log(f[mask] + 1e-30)
    lp   = np.log(pxx[mask] + 1e-30)
    slope, _, _, _, _ = stats_mod.linregress(lf, lp)

    # HF elevation: mean power 100–300 Hz vs 5–50 Hz
    mask_hf = (f >= 100) & (f <= min(300, fs / 2 - 1))
    mask_lf = (f >= 5) & (f <= 50)
    hf_ratio = (pxx[mask_hf].mean() / (pxx[mask_lf].mean() + 1e-30)
                if mask_hf.any() else 0.0)

    # Oscillatory peaks relative to local baseline
    def _peak_ratio(lo, hi, expand=10):
        b_mask = (f >= lo) & (f <= hi)
        w_mask = (f >= max(1, lo - expand)) & (f <= hi + expand)
        if not b_mask.any():
            return 0.0
        return float(pxx[b_mask].max() / (np.median(pxx[w_mask]) + 1e-30))

    alpha_pr = _peak_ratio(8,  14)
    beta_pr  = _peak_ratio(14, 35)
    has_neural = (alpha_pr > 2.0) or (beta_pr > 1.5)

    if hf_ratio > 3.0:
        return 'elevated-HF (noise-dominated)'
    elif slope < -0.5:
        return 'normal 1/f + neural structure' if has_neural else 'normal 1/f'
    elif slope < -0.2:
        return 'weak 1/f'
    else:
        return 'flat/structureless'


# ── Decoder impact ────────────────────────────────────────────────────────────

def _assemble_feats(x_raw, cx, cy, z_thresh):
    """
    Run preprocessing at z_thresh, extract features, segment trials.
    Returns (feats, pos, trial_ids, good_idx).
    """
    x_car, good_idx, _, _ = preprocess(x_raw, z_thresh=z_thresh)
    feats_full = extract_features(x_car)

    trials = find_trial_boundaries(cx, cy)
    feats_l, pos_l, tid_l = [], [], []
    for tid_i, (ts, te) in enumerate(trials):
        f = feats_full[ts // DECIM : te // DECIM]
        p = cursor_position(cx[ts:te], cy[ts:te])
        T_ = min(f.shape[0], p.shape[0])
        feats_l.append(f[:T_])
        pos_l.append(p[:T_])
        tid_l.append(np.full(T_, tid_i, dtype=int))

    feats    = np.concatenate(feats_l)
    pos_arr  = np.concatenate(pos_l)
    trial_ids = np.concatenate(tid_l)
    return feats, pos_arr, trial_ids, good_idx


def _decode_mean_r(feats, pos, trial_ids):
    """Return mean Pearson r across folds and axes (scalar)."""
    _, _, _, fold_r, _, _ = decode_cv(feats, pos, trial_ids)
    return float(fold_r.mean())


def compute_decoder_impacts(s, marginal_chs_0idx, x_raw, cx, cy):
    """
    For each marginal channel (0-indexed), return delta_r = r_with - r_without.

    Uses preprocessing at z_thresh=6.0 so all marginal channels are included.
    Positive delta_r means including the channel helps; negative means it hurts.
    """
    if not marginal_chs_0idx:
        return {}

    print(f'  [decoder] preprocessing at z_thresh=6.0 to include all marginal channels …')
    feats, pos, trial_ids, good_idx = _assemble_feats(x_raw, cx, cy, z_thresh=6.0)

    print(f'  [decoder] baseline ({feats.shape[1]} channels):')
    r_base = _decode_mean_r(feats, pos, trial_ids)
    print(f'  [decoder] baseline mean r = {r_base:.4f}')

    impacts = {}
    for ch in marginal_chs_0idx:
        positions = np.where(good_idx == ch)[0]
        if not len(positions):
            print(f'  [decoder] ch{ch+1}: not in good channels at z_thresh=6.0, skip')
            impacts[ch] = None
            continue
        feat_col = int(positions[0])
        feats_wo = np.delete(feats, feat_col, axis=1)
        print(f'  [decoder] ch{ch+1} (without):')
        r_wo = _decode_mean_r(feats_wo, pos, trial_ids)
        impacts[ch] = r_base - r_wo   # positive = channel contributes
        print(f'  [decoder] ch{ch+1}: Δr = {impacts[ch]:+.4f}  '
              f'(r_with={r_base:.4f}  r_without={r_wo:.4f})')

    return impacts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-decode', action='store_true',
                        help='Skip decoder impact computation (much faster)')
    args = parser.parse_args()
    run_decode = not args.no_decode

    os.makedirs(OUT_DIR, exist_ok=True)

    for s in SUBJECTS:
        print(f'\n{"=" * 72}')
        print(f'  {s.upper()} — threshold justification')
        print(f'{"=" * 72}')

        d     = sio.loadmat(f'{DATA_DIR}/{s}_joystick.mat')
        x_raw = d['data'].astype(float)
        cx    = d['CursorPosX'].flatten()
        cy    = d['CursorPosY'].flatten()
        n_ch  = x_raw.shape[1]

        # ── Compute z-scores once (single-pass, no iterative CAR) ─────────
        x_notch = sig.filtfilt(_B_NOTCH, _A_NOTCH, x_raw, axis=0)
        x_det   = _build_detect_copy(x_notch, detect_hp_hz=2.0)
        amp     = _channel_amplitude(x_det)
        amp_z   = _mad_z(amp)

        lnr   = _line_noise_snr(x_raw)
        snr_z = _mad_z(np.log1p(lnr))

        flat_flag = x_raw.std(0) < FLAT_AMP_THRESH

        # ── Threshold sweep ───────────────────────────────────────────────
        # amp_flagged_thrs[ch] = set of thresholds where |amp_z[ch]| > thresh
        amp_flagged_thrs = {ch: set() for ch in range(n_ch)}
        lnr_flagged_thrs = {ch: set() for ch in range(n_ch)}

        for thr in THRESH_SWEEP:
            flag_amp = np.abs(amp_z) > thr
            flag_lnr = snr_z > thr
            for ch in range(n_ch):
                if flag_amp[ch]: amp_flagged_thrs[ch].add(thr)
                if flag_lnr[ch]: lnr_flagged_thrs[ch].add(thr)

        def _stability(flagged_set):
            n = len(THRESH_SWEEP)
            k = len(flagged_set)
            if k == n:   return 'stable-bad'
            if k == 0:   return 'stable-good'
            return 'marginal'

        # ── Build report list ─────────────────────────────────────────────
        # Include channels that are stable-bad or marginal for either criterion.
        report_rows = []
        for ch in range(n_ch):
            if flat_flag[ch]:
                continue
            for crit, flagged_thrs, z_val in [
                ('amplitude',  amp_flagged_thrs[ch], float(amp_z[ch])),
                ('line_noise', lnr_flagged_thrs[ch], float(snr_z[ch])),
            ]:
                stab = _stability(flagged_thrs)
                if stab in ('stable-bad', 'marginal'):
                    # Determine transition range for marginal channels
                    if stab == 'marginal':
                        lo = min(flagged_thrs)
                        hi = max(flagged_thrs)
                        stab_label = f'marginal({lo:.2f}–{hi:.2f})'
                    else:
                        stab_label = 'stable-bad'
                    report_rows.append((ch, crit, z_val, stab, stab_label, flagged_thrs))

        # Sort: stable-bad first (by |z| desc), then marginal (by |z| desc)
        report_rows.sort(key=lambda r: (r[3] == 'marginal', -abs(r[2])))

        if not report_rows:
            print('  No flagged or marginal channels.')
            continue

        # ── PSD classification ────────────────────────────────────────────
        unique_chs = list({r[0] for r in report_rows})
        psd_class  = {ch: _psd_classify(x_raw[:, ch]) for ch in unique_chs}

        # ── Decoder impact ────────────────────────────────────────────────
        decode_delta = {}
        if run_decode:
            marginal_chs = list({r[0] for r in report_rows if r[3] == 'marginal'})
            decode_delta = compute_decoder_impacts(s, marginal_chs, x_raw, cx, cy)

        # ── Print summary table ───────────────────────────────────────────
        print(f'\n  {"ch":>4}  {"criterion":>12}  {"z":>7}  {"stability":>24}  '
              f'{"PSD class":<38}', end='')
        if run_decode:
            print(f'  {"decode_Δr":>10}', end='')
        print()
        print('  ' + '─' * (4 + 2 + 12 + 2 + 7 + 2 + 24 + 2 + 38 + (13 if run_decode else 0)))

        for ch, crit, z_val, stab, stab_label, flagged_thrs in report_rows:
            pc  = psd_class.get(ch, '—')
            row = (f'  {ch+1:>4}  {crit:>12}  {z_val:>7.2f}  '
                   f'{stab_label:>24}  {pc:<38}')
            if run_decode:
                delta = decode_delta.get(ch)
                row  += f'  {(f"{delta:+.4f}" if delta is not None else "n/a"):>10}'
            print(row)

        # ── FP ch18 call-out ──────────────────────────────────────────────
        if s == 'fp':
            fp18 = [(ch, crit, z_val, stab, stab_label, ft)
                    for ch, crit, z_val, stab, stab_label, ft in report_rows
                    if ch == 17 and crit == 'amplitude']
            if fp18:
                ch, crit, z_val, stab, stab_label, ft = fp18[0]
                flagged_at = sorted(ft)
                not_flagged = sorted(set(THRESH_SWEEP.tolist()) - ft)
                print(f'\n  ► FP ch18: amp_z = {z_val:.2f}  stability = {stab_label}')
                print(f'     flagged  at thresholds : {flagged_at}')
                print(f'     not flagged at thresholds: {not_flagged}')
                print(f'     PSD class : {psd_class.get(17, "—")}')
                if run_decode and 17 in decode_delta:
                    d18 = decode_delta[17]
                    print(f'     decode Δr : {d18:+.4f}  '
                          f'({"channel contributes — borderline call is load-bearing" if d18 > 0.01 else "channel hurts — exclusion is correct" if d18 < -0.01 else "immaterial — threshold choice does not affect decoding"})')
            else:
                print(f'\n  ► FP ch18: not in marginal/flagged set for amplitude criterion')

        # ── Save CSV ──────────────────────────────────────────────────────
        csv_path = os.path.join(OUT_DIR, f'thresh_justification_{s}.csv')
        with open(csv_path, 'w') as fh:
            header = 'ch,criterion,z_score,stability,psd_class'
            if run_decode:
                header += ',decode_delta_r'
            fh.write(header + '\n')
            for ch, crit, z_val, stab, stab_label, _ in report_rows:
                pc  = psd_class.get(ch, '')
                row = f'{ch+1},{crit},{z_val:.4f},{stab_label},"{pc}"'
                if run_decode:
                    delta = decode_delta.get(ch)
                    row  += f',{delta:.4f}' if delta is not None else ',n/a'
                fh.write(row + '\n')

        print(f'\n  Saved → {csv_path}')


if __name__ == '__main__':
    main()
