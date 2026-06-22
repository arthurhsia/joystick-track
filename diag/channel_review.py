"""
Per-subject bad-channel review.

For each subject produces one figure:
  Row 1: rank-ordered amplitude (IQR×0.7413) | log line-noise SNR
  Row 2: PSD overlay (raw, log-log) — gray=clean  orange=borderline  red=flagged
  Row 3: stacked raw traces — flagged (red) + borderline survivors (orange)

Usage:
    python diag/channel_review.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from config import (FS, FLAT_AMP_THRESH, AMP_Z_THRESH, LINE_NOISE_Z_THRESH,
                    DATA_DIR, OUT_DIR, SUBJECTS, COLORS)
from core.preprocessing import (_B_NOTCH, _A_NOTCH, _line_noise_snr, _mad_z,
                                _channel_amplitude, _build_detect_copy)

N_BORDER = 3   # top-N unflagged channels per metric to treat as borderline survivors
T_SHOW   = 30  # seconds of raw trace to display


def _load(s):
    d = sio.loadmat(f'{DATA_DIR}/{s}_joystick.mat')
    return d['data'].astype(float)


def _mad_thresh(x, z_thresh):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return med + (z_thresh / 0.6745) * mad


def _top_n_unflagged(score, flag, n):
    unflagged = np.where(~flag)[0]
    if len(unflagged) == 0:
        return np.array([], dtype=int)
    return unflagged[np.argsort(score[unflagged])[::-1][:n]]


def _rank_plot(ax, metric, flag_bad, flag_border, thresh, ylabel, color_good,
               title, log_y=False):
    n = len(metric)
    order = np.argsort(metric)[::-1]
    sv = metric[order]
    bar_colors = []
    for ch in order:
        if flag_bad[ch]:      bar_colors.append('tomato')
        elif flag_border[ch]: bar_colors.append('darkorange')
        else:                 bar_colors.append(color_good)
    ax.bar(np.arange(1, n + 1), sv, color=bar_colors, width=1.0, alpha=0.85)
    if thresh is not None:
        n_flag = int(flag_bad.sum())
        ax.axhline(thresh, color='red', ls='--', lw=1.5,
                   label=f'thresh {thresh:.4g}  ({n_flag} flagged)')
        ax.legend(fontsize=7, loc='upper right')
    ax.set_xlabel('Channel rank')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    if log_y:
        ax.set_yscale('log')


def main():
    for s in SUBJECTS:
        print(f'\n── {s.upper()} ──')
        x_raw   = _load(s)
        lnr     = _line_noise_snr(x_raw)
        x_notch = sig.filtfilt(_B_NOTCH, _A_NOTCH, x_raw, axis=0)
        x_det   = _build_detect_copy(x_notch, detect_hp_hz=2.0)

        amp       = _channel_amplitude(x_det)              # IQR×0.7413 on HP-copy
        log_snr   = np.log1p(lnr)
        flat_flag = x_raw.std(0) < FLAT_AMP_THRESH
        n_ch      = amp.shape[0]

        amp_z = _mad_z(amp)
        snr_z = _mad_z(log_snr)

        flag_amp  = np.abs(amp_z) > AMP_Z_THRESH   # two-sided
        flag_lnr  = (snr_z > LINE_NOISE_Z_THRESH
                     if LINE_NOISE_Z_THRESH is not None
                     else np.zeros(n_ch, bool))
        bad_mask = flat_flag | flag_amp | flag_lnr

        thresh_amp  = _mad_thresh(amp,     AMP_Z_THRESH)
        thresh_snr  = (_mad_thresh(log_snr, LINE_NOISE_Z_THRESH)
                       if LINE_NOISE_Z_THRESH is not None else None)

        # RMS for PSD/trace visualization only (not used in detection)
        rms = np.sqrt((x_notch ** 2).mean(0))

        # Borderline: top N unflagged by each metric's z-score
        border_set = (set(_top_n_unflagged(amp_z,  flag_amp, N_BORDER).tolist()) |
                      set(_top_n_unflagged(snr_z,  flag_lnr, N_BORDER).tolist()))
        border_idx  = np.array(sorted(border_set - set(np.where(bad_mask)[0])), dtype=int)
        flag_border = np.zeros(n_ch, bool)
        if len(border_idx):
            flag_border[border_idx] = True

        bad_idx   = np.where(bad_mask)[0]
        clean_idx = np.where(~bad_mask & ~flag_border)[0]
        n_bad, n_border = len(bad_idx), len(border_idx)

        def reason(ch):
            r = []
            if bad_mask[ch]:
                if flat_flag[ch]: r.append('flat')
                if flag_amp[ch]:  r.append(f'amp z={amp_z[ch]:.1f}')
                if flag_lnr[ch]:  r.append(f'SNR z={snr_z[ch]:.1f}')
            else:
                r.append(f'border — amp z={amp_z[ch]:.1f}  SNR z={snr_z[ch]:.1f}')
            return '  '.join(r)

        print(f'  flagged: {n_bad}   borderline: {n_border}   clean: {len(clean_idx)}')
        for ch in bad_idx:
            print(f'    ch{ch+1:3d}  {reason(ch)}')

        # ── Figure ──────────────────────────────────────────────────
        n_review = n_bad + n_border
        fig = plt.figure(figsize=(17, max(10, 4.5 + n_review * 0.55)))
        gs  = gridspec.GridSpec(
            3, 2, figure=fig,
            height_ratios=[2, 2.5, max(2.5, n_review * 0.55)],
            hspace=0.45, wspace=0.32
        )
        ax_rms_p = fig.add_subplot(gs[0, 0])
        ax_snr_p = fig.add_subplot(gs[0, 1])
        ax_psd   = fig.add_subplot(gs[1, :])
        ax_tr    = fig.add_subplot(gs[2, :])

        fig.suptitle(
            f'{s.upper()} — channel review   '
            f'{n_bad} flagged  ·  {n_border} borderline  ·  {len(clean_idx)} clean  '
            f'(total {n_ch})',
            fontsize=12, fontweight='bold'
        )

        # Row 1 — rank plots
        _rank_plot(ax_rms_p, amp,     bad_mask, flag_border,
                   thresh_amp, 'IQR×0.7413 (AU)', COLORS[s], 'Sorted amplitude (HP-copy)')
        _rank_plot(ax_snr_p, log_snr, bad_mask, flag_border,
                   thresh_snr, 'log(1 + SNR)',     COLORS[s], 'Sorted log line-noise SNR')

        # Row 2 — PSD overlay (raw, log-log)
        f_psd, all_psd = sig.welch(x_raw, fs=FS, nperseg=FS * 4, axis=0)
        all_psd = all_psd.T  # (C, F)
        mask1 = f_psd > 0
        for ch in clean_idx:
            ax_psd.loglog(f_psd[mask1], all_psd[ch, mask1],
                          color='#cccccc', lw=0.35, alpha=0.6)
        for ch in border_idx:
            ax_psd.loglog(f_psd[mask1], all_psd[ch, mask1],
                          color='darkorange', lw=0.9, alpha=0.85)
        for ch in bad_idx:
            ax_psd.loglog(f_psd[mask1], all_psd[ch, mask1],
                          color='tomato', lw=1.3, alpha=0.9)
        for h in [60, 120, 180]:
            ax_psd.axvline(h, color='red', ls=':', lw=0.8, alpha=0.35)
        ax_psd.set_xlim(1, FS // 2)
        ax_psd.set_xlabel('Frequency (Hz)')
        ax_psd.set_ylabel('PSD (AU²/Hz)')
        ax_psd.set_title('PSD overlay (raw, log-log) — 1/f breaks, 60 Hz spikes, flat lines all stand out')
        ax_psd.legend(handles=[
            Line2D([0], [0], color='#cccccc',    lw=1.5, label=f'clean ({len(clean_idx)})'),
            Line2D([0], [0], color='darkorange', lw=1.5, label=f'borderline ({n_border})'),
            Line2D([0], [0], color='tomato',     lw=1.5, label=f'flagged ({n_bad})'),
        ], fontsize=8, loc='upper right')

        # Row 3 — stacked raw traces
        T_samp = min(int(T_SHOW * FS), x_raw.shape[0])
        t_ax   = np.arange(T_samp) / FS
        scale  = np.median(rms[clean_idx]) if len(clean_idx) else np.median(rms)
        slot   = 12 * scale

        review = list(bad_idx) + list(border_idx)
        for i, ch in enumerate(review):
            trace  = x_raw[:T_samp, ch].astype(float)
            offset = (len(review) - 1 - i) * slot
            color  = 'tomato' if bad_mask[ch] else 'darkorange'
            ax_tr.plot(t_ax, trace + offset, color=color, lw=0.5, alpha=0.85)
            ax_tr.text(-0.5, offset, f'ch{ch+1}',
                       va='center', ha='right', fontsize=7.5, color=color,
                       fontweight='bold', clip_on=False)
            ax_tr.text(T_SHOW + 0.3, offset, reason(ch),
                       va='center', ha='left', fontsize=6.5, color=color, clip_on=False)

        if n_review == 0:
            ax_tr.text(0.5, 0.5, 'No flagged or borderline channels',
                       ha='center', va='center', transform=ax_tr.transAxes, fontsize=11)
        ax_tr.set_xlabel('Time (s)')
        ax_tr.set_title('Raw traces — red=flagged  orange=borderline survivors')
        ax_tr.set_xlim(0, T_SHOW)
        ax_tr.set_yticks([])

        path = f'{OUT_DIR}/figN9_{s}_channel_review.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  → {path}')


if __name__ == '__main__':
    main()
