"""
Raw amplitude trace figure — all channels per subject.

For each subject produces one figure with two panels:
  Top    : IQR×0.7413 per channel (HP-copy; detection metric)
  Bottom : stacked raw traces (first T_SHOW seconds)

  red    = flagged by production preprocess()
  orange = borderline (|amp_z| between 2.5 and AMP_Z_THRESH, not flagged)
  gray   = clean

Usage:
    python diag/raw_amp_plot.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import scipy.io as sio
import scipy.signal as sig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from config import (FS, DATA_DIR, OUT_DIR, SUBJECTS,
                    AMP_Z_THRESH, LINE_NOISE_Z_THRESH, FLAT_AMP_THRESH)
from core.preprocessing import (preprocess, _B_NOTCH, _A_NOTCH, _line_noise_snr,
                                 _mad_z, _channel_amplitude, _build_detect_copy)

T_SHOW       = 30    # seconds of raw trace to display
BORDER_Z_LO  = 2.5  # lower bound for "borderline" z-score (not flagged but notable)


def _channel_colors(bad_mask, amp_z, snr_z, n_ch):
    """Return per-channel color and linewidth arrays."""
    colors = []
    lws    = []
    for ch in range(n_ch):
        if bad_mask[ch]:
            colors.append('tomato')
            lws.append(0.9)
        elif (abs(amp_z[ch]) >= BORDER_Z_LO or
              snr_z[ch] >= BORDER_Z_LO):
            colors.append('darkorange')
            lws.append(0.6)
        else:
            colors.append('#999999')
            lws.append(0.35)
    return colors, lws


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for s in SUBJECTS:
        print(f'── {s.upper()} ──')
        d     = sio.loadmat(f'{DATA_DIR}/{s}_joystick.mat')
        x_raw = d['data'].astype(float)
        n_ch  = x_raw.shape[1]

        # ── Production detection ───────────────────────────────────────────
        _, good_idx, bad_mask, reasons = preprocess(x_raw)
        n_bad   = int(bad_mask.sum())
        bad_idx = np.where(bad_mask)[0]

        # ── Single-pass metrics ────────────────────────────────────────────
        x_notch    = sig.filtfilt(_B_NOTCH, _A_NOTCH, x_raw, axis=0)
        x_det      = _build_detect_copy(x_notch, detect_hp_hz=2.0)
        amp        = _channel_amplitude(x_det)   # IQR×0.7413 on HP-copy
        amp_z      = _mad_z(amp)
        lnr        = _line_noise_snr(x_raw)
        snr_z      = _mad_z(np.log1p(lnr))
        colors, lws = _channel_colors(bad_mask, amp_z, snr_z, n_ch)

        n_border = int(sum(
            1 for ch in range(n_ch)
            if not bad_mask[ch] and (abs(amp_z[ch]) >= BORDER_Z_LO
                                     or snr_z[ch] >= BORDER_Z_LO)
        ))
        n_clean = n_ch - n_bad - n_border

        # ── Figure layout: 2 panels ────────────────────────────────────────
        fig_h = max(16, n_ch * 0.22 + 6)
        fig   = plt.figure(figsize=(17, fig_h))
        gs    = gridspec.GridSpec(
            2, 1, figure=fig,
            height_ratios=[1.8, max(2, n_ch * 0.22)],
            hspace=0.30,
        )
        ax_iqr = fig.add_subplot(gs[0])
        ax_tr  = fig.add_subplot(gs[1])

        fig.suptitle(
            f'{s.upper()} — all {n_ch} channels   '
            f'red=flagged ({n_bad})  '
            f'orange=borderline |z|≥{BORDER_Z_LO} ({n_border})  '
            f'gray=clean ({n_clean})',
            fontsize=11, fontweight='bold', y=0.998,
        )

        ch_idx = np.arange(n_ch)

        # ── Top-left: IQR×0.7413 bar chart ────────────────────────────────
        ax_iqr.bar(ch_idx + 1, amp, color=colors, width=0.85, alpha=0.85)
        med_amp = np.median(amp)
        mad_amp = np.median(np.abs(amp - med_amp))
        for sigma in (AMP_Z_THRESH, -AMP_Z_THRESH):
            thr = med_amp + (sigma / 0.6745) * mad_amp
            if thr > 0:
                ax_iqr.axhline(thr, color='red', ls='--', lw=1.1,
                               label=f'±{AMP_Z_THRESH}σ ({thr:.0f})' if sigma > 0 else None)
        ax_iqr.axhline(med_amp, color='black', ls=':', lw=0.8, alpha=0.5,
                       label=f'median ({med_amp:.0f})')
        for ch in bad_idx:
            ax_iqr.text(ch + 1, amp[ch], f'{ch+1}',
                        ha='center', va='bottom', fontsize=6.5,
                        color='tomato', fontweight='bold')
        ax_iqr.set_xlim(0.5, n_ch + 0.5)
        ax_iqr.set_xticks(np.arange(1, n_ch + 1, 4))
        ax_iqr.set_xlabel('Channel', fontsize=9)
        ax_iqr.set_ylabel('IQR×0.7413  (AU)', fontsize=9)
        ax_iqr.set_title('Amplitude — IQR×0.7413 (HP-copy, detection metric)', fontsize=9)
        ax_iqr.legend(fontsize=7, loc='upper right')
        ax_iqr.tick_params(labelsize=8)

        # ── Bottom: stacked raw traces ─────────────────────────────────────
        T_samp = min(int(T_SHOW * FS), x_raw.shape[0])
        t_ax   = np.arange(T_samp) / FS

        good_clean_idx = np.where(
            ~bad_mask & np.array([abs(amp_z[ch]) < BORDER_Z_LO and
                                  snr_z[ch] < BORDER_Z_LO
                                  for ch in range(n_ch)])
        )[0]
        rms_win = np.sqrt((x_raw[:T_samp] ** 2).mean(0))
        scale   = (np.median(rms_win[good_clean_idx])
                   if len(good_clean_idx) else np.median(rms_win))
        slot    = 4.5 * scale

        for ch in range(n_ch):
            trace  = x_raw[:T_samp, ch]
            offset = (n_ch - 1 - ch) * slot
            ax_tr.plot(t_ax, trace + offset,
                       color=colors[ch], lw=lws[ch],
                       alpha=(0.9 if bad_mask[ch] else 0.75 if colors[ch] == 'darkorange' else 0.5))
            ax_tr.text(-0.4, offset, f'{ch+1}',
                       va='center', ha='right', fontsize=6,
                       color=colors[ch], fontweight=('bold' if bad_mask[ch] else 'normal'),
                       clip_on=False)

        ax_tr.set_xlim(0, T_SHOW)
        ax_tr.set_xlabel('Time (s)', fontsize=9)
        ax_tr.set_yticks([])
        ax_tr.set_title(f'Raw traces — first {T_SHOW} s', fontsize=9)
        ax_tr.tick_params(labelsize=8)

        path = f'{OUT_DIR}/figN10_{s}_raw_amp.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  → {path}')


if __name__ == '__main__':
    main()
