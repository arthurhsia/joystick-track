"""
Noise analysis for ECoG joystick-tracking dataset.
Covers: line noise, channel quality, saturation/clipping,
inter-channel correlation, temporal stationarity, artifact epochs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from core.preprocessing import (_B_NOTCH, _A_NOTCH, _line_noise_snr, _mad_z,
                                _channel_amplitude, _build_detect_copy)
from config import (FS, FLAT_AMP_THRESH, AMP_Z_THRESH,
                    LINE_NOISE_Z_THRESH, DATA_DIR, OUT_DIR, SUBJECTS, COLORS)


def load(s):
    d = sio.loadmat(f'{DATA_DIR}/{s}_joystick.mat')
    return d['data'].astype(float)   # (T, C)


def _mad_thresh_raw(x):
    """MAD z-score threshold expressed in the original units of x."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return med + (AMP_Z_THRESH / 0.6745) * mad


def _lnr_thresh_raw(lnsnr):
    """MAD z-score threshold on log(SNR), back-transformed to raw SNR units."""
    log_snr = np.log1p(lnsnr)
    med = np.median(log_snr)
    mad = np.median(np.abs(log_snr - med))
    return float(np.expm1(med + (LINE_NOISE_Z_THRESH / 0.6745) * mad))


# ── FIG N1: Line Noise ────────────────────────────────────────────────────────
print('Fig N1: Line noise ...')
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Line Noise: PSD at 60 Hz harmonics (mean across all channels)',
             fontsize=13, fontweight='bold')
harmonics = [60, 120, 180]
for ax, s in zip(axes.flat, SUBJECTS):
    x = load(s); all_psd = []
    for ch in range(x.shape[1]):
        f, pxx = sig.welch(x[:, ch], fs=FS, nperseg=FS*4)
        all_psd.append(pxx)
    mean_psd = np.array(all_psd).mean(0)
    ax.semilogy(f, mean_psd, color=COLORS[s], lw=0.8)
    ax.set_xlim(0, 200); ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('PSD (AU²/Hz)')
    ax.set_title(f'Subject {s.upper()}')
    for h in harmonics:
        ax.axvline(h, color='red', lw=1, ls='--', alpha=0.7, label=f'{h} Hz' if h == 60 else None)
    if s == 'fp': ax.legend(loc='upper right', fontsize=8)
    for h in harmonics:
        idx_h  = np.argmin(np.abs(f - h))
        idx_lo = np.argmin(np.abs(f - (h - 5)))
        idx_hi = np.argmin(np.abs(f - (h + 5)))
        base = np.median(np.concatenate([mean_psd[idx_lo:idx_h-2], mean_psd[idx_h+2:idx_hi]]))
        snr  = 10 * np.log10(mean_psd[idx_h] / base)
        ax.text(h, mean_psd[idx_h]*2, f'+{snr:.0f}dB', ha='center', fontsize=7, color='red')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/figN1_line_noise.png', dpi=150); plt.close()

# ── FIG N2: Channel quality — rank-ordered (amplitude | line-noise SNR) ──────
print('Fig N2: Channel quality ...')
fig, axes = plt.subplots(len(SUBJECTS), 2, figsize=(12, 3.8 * len(SUBJECTS)))
fig.suptitle('Channel quality — rank-ordered (worst → best)\n'
             'Red dashed = MAD z-score threshold  |  red bars = flagged channels',
             fontsize=11, fontweight='bold')

print('\n' + '─' * 52)
for row, s in enumerate(SUBJECTS):
    x_raw   = load(s)
    x_notch = sig.filtfilt(_B_NOTCH, _A_NOTCH, x_raw, axis=0)
    x_det   = _build_detect_copy(x_notch, detect_hp_hz=2.0)

    amp       = _channel_amplitude(x_det)           # IQR×0.7413 on HP-copy (production metric)
    lnsnr     = _line_noise_snr(x_raw)
    flat_flag = x_raw.std(0) < FLAT_AMP_THRESH
    n         = amp.shape[0]

    thresh_amp = _mad_thresh_raw(amp)
    thresh_lnr = _lnr_thresh_raw(lnsnr)

    flag_amp  = np.abs(_mad_z(amp)) > AMP_Z_THRESH
    flag_lnr  = (_mad_z(np.log1p(lnsnr)) > LINE_NOISE_Z_THRESH
                 if LINE_NOISE_Z_THRESH is not None else np.zeros(n, bool))
    n_bad = int((flat_flag | flag_amp | flag_lnr).sum())
    print(f'{s.upper()}  flat {flat_flag.sum()}  amp {flag_amp.sum()}  '
          f'LN-SNR {flag_lnr.sum()}  → {n_bad}/{n} removed')

    for col, (metric, flag, thresh, ylabel, log_scale) in enumerate([
        (amp,   flag_amp, thresh_amp, 'IQR×0.7413 (AU)', False),
        (lnsnr, flag_lnr, thresh_lnr, 'Line-noise SNR',  True),
    ]):
        ax = axes[row, col]
        order = np.argsort(metric)[::-1]
        sv    = metric[order]
        fl    = flag[order]

        bar_colors = ['tomato' if f else COLORS[s] for f in fl]
        ax.bar(np.arange(1, n + 1), sv, color=bar_colors, width=1.0, alpha=0.85)
        ax.axhline(thresh, color='red', ls='--', lw=1.5,
                   label=f'MAD z>{AMP_Z_THRESH}  ({fl.sum()} flagged)')
        ax.set_xlabel('Channel rank')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{s.upper()} – {ylabel}', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        if log_scale:
            ax.set_yscale('log')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/figN2_channel_quality.png', dpi=150); plt.close()
print('─' * 52)

# ── FIG N3: Inter-channel Correlation ─────────────────────────────────────────
print('Fig N3: Correlation matrix ...')
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
fig.suptitle('Inter-channel Correlation Matrix', fontsize=12, fontweight='bold')
for ax, s in zip(axes.flat, SUBJECTS):
    x = load(s); T_USE = min(60 * FS, x.shape[0])
    xz = x[:T_USE]; xz = (xz - xz.mean(0)) / (xz.std(0) + 1e-9)
    corr = (xz.T @ xz) / T_USE
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_title(f'{s.upper()} – {x.shape[1]} channels')
    ax.set_xlabel('Channel'); ax.set_ylabel('Channel')
    mask = ~np.eye(corr.shape[0], dtype=bool)
    ax.text(0.02, 0.97, f'mean off-diag r = {corr[mask].mean():.3f}',
            transform=ax.transAxes, fontsize=9, va='top', color='k',
            bbox=dict(fc='white', alpha=0.7, boxstyle='round'))
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/figN3_corr_matrix.png', dpi=150); plt.close()

# ── FIG N4: Temporal Stationarity ────────────────────────────────────────────
print('Fig N4: Temporal stationarity ...')
WIN_S = 5; WIN = WIN_S * FS
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
fig.suptitle(f'Temporal Stationarity: Sliding RMS ({WIN_S} s windows)',
             fontsize=12, fontweight='bold')
for ax, s in zip(axes, SUBJECTS):
    x = load(s); n_wins = x.shape[0] // WIN; t_wins = np.arange(n_wins) * WIN_S
    rms_mat = np.array([np.sqrt((x[i*WIN:(i+1)*WIN]**2).mean(0)) for i in range(n_wins)])
    mean_rms = rms_mat.mean(1)
    ax.fill_between(t_wins, np.percentile(rms_mat, 10, axis=1),
                    np.percentile(rms_mat, 90, axis=1), alpha=0.25, color=COLORS[s])
    ax.plot(t_wins, mean_rms, color=COLORS[s], lw=1.5)
    ax.set_ylabel('RMS (AU)'); ax.set_title(f'Subject {s.upper()}')
    thresh   = mean_rms.mean() + 3 * mean_rms.std()
    bad_wins = t_wins[mean_rms > thresh]
    if len(bad_wins):
        ax.scatter(bad_wins, mean_rms[mean_rms > thresh], color='red', zorder=5, s=30,
                   label='> 3 SD artifact epoch')
        ax.legend(fontsize=8)
    ax.axhline(thresh, color='red', lw=0.8, ls='--', alpha=0.5)
axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/figN4_stationarity.png', dpi=150); plt.close()

# ── FIG N5: Amplitude Distributions ─────────────────────────────────────────
print('Fig N5: Amplitude distributions ...')
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Amplitude Distributions (all channels overlaid)', fontsize=12, fontweight='bold')
for ax, s in zip(axes.flat, SUBJECTS):
    x = load(s); T_USE = min(30 * FS, x.shape[0])
    for ch in range(x.shape[1]):
        vals = x[:T_USE, ch]; vals_z = (vals - vals.mean()) / (vals.std() + 1e-9)
        counts, edges = np.histogram(vals_z, bins=200, range=(-8, 8), density=True)
        ax.semilogy((edges[:-1] + edges[1:]) / 2, counts + 1e-6,
                    color=COLORS[s], alpha=0.08, lw=0.5)
    from scipy.stats import norm
    zv = np.linspace(-8, 8, 400)
    ax.semilogy(zv, norm.pdf(zv), 'k--', lw=1.5, label='Gaussian')
    ax.set_xlim(-8, 8); ax.set_ylim(1e-5, 1)
    ax.set_xlabel('z-score'); ax.set_ylabel('Density')
    ax.set_title(f'{s.upper()} – {x.shape[1]} channels'); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/figN5_amplitude_dist.png', dpi=150); plt.close()

# ── FIG N6: Notch filter before/after ────────────────────────────────────────
print('Fig N6: Notch filter comparison ...')
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Effect of 60 Hz notch filter — representative channel per subject',
             fontsize=12, fontweight='bold')
for ax, s in zip(axes.flat, SUBJECTS):
    x    = load(s); snrs = []
    for ch in range(x.shape[1]):
        f, pxx = sig.welch(x[:, ch], fs=FS, nperseg=FS*4)
        idx60  = np.argmin(np.abs(f - 60))
        base   = np.median(pxx[np.argmin(np.abs(f-55)):np.argmin(np.abs(f-58))])
        snrs.append(pxx[idx60] / (base + 1e-9))
    rep_ch = int(np.argmax(snrs)); raw = x[:, rep_ch]
    filt   = sig.filtfilt(_B_NOTCH, _A_NOTCH, raw)
    f, pxx_raw  = sig.welch(raw,  fs=FS, nperseg=FS*4)
    f, pxx_filt = sig.welch(filt, fs=FS, nperseg=FS*4)
    ax.semilogy(f, pxx_raw,  color='gray',    lw=1,   label='raw')
    ax.semilogy(f, pxx_filt, color=COLORS[s], lw=1.2, label='notch (60+120+180 Hz)')
    for h in [60, 120, 180]: ax.axvline(h, color='red', ls='--', lw=0.8, alpha=0.6)
    ax.set_xlim(0, 200); ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('PSD')
    ax.set_title(f'{s.upper()} – ch {rep_ch+1} (highest 60 Hz SNR)'); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/figN6_notch_comparison.png', dpi=150); plt.close()

# ── FIG N7: CAR effect ───────────────────────────────────────────────────────
print('Fig N7: CAR effect ...')
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Common Average Reference (CAR) Effect on PSD', fontsize=12, fontweight='bold')
for ax, s in zip(axes.flat, SUBJECTS):
    x = load(s); rms = np.sqrt((x**2).mean(0))
    rep_ch = int(np.argsort(rms)[len(rms)//2]); raw = x[:, rep_ch]
    car    = raw - x.mean(1)
    f, pxx_raw = sig.welch(raw, fs=FS, nperseg=FS*4)
    f, pxx_car = sig.welch(car, fs=FS, nperseg=FS*4)
    ax.semilogy(f, pxx_raw, color='gray',    lw=1,   alpha=0.8, label='raw')
    ax.semilogy(f, pxx_car, color=COLORS[s], lw=1.2, label='CAR')
    for h in [60, 120, 180]: ax.axvline(h, color='red', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlim(0, 200); ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('PSD')
    ax.set_title(f'{s.upper()} – median-RMS channel (ch {rep_ch+1})'); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/figN7_CAR.png', dpi=150); plt.close()

# ── FIG N8: Bad channel visual review ────────────────────────────────────────
print('Fig N8: Bad channel visual review ...')
T_SHOW = 30

for s in SUBJECTS:
    x_raw   = load(s)
    lnr     = _line_noise_snr(x_raw)
    x_notch = sig.filtfilt(_B_NOTCH, _A_NOTCH, x_raw, axis=0)
    x_det   = _build_detect_copy(x_notch, detect_hp_hz=2.0)

    amp       = _channel_amplitude(x_det)
    clip_rate = (np.abs(x_raw) >= 0.95 * INT16_MAX).mean(0)  # V_raw
    flat_flag = x_raw.std(0) < FLAT_AMP_THRESH

    flag_flat = flat_flag
    flag_amp  = np.abs(_mad_z(amp)) > AMP_Z_THRESH
    flag_clip = clip_rate > CLIP_FRAC_THRESH
    flag_lnr  = (_mad_z(np.log1p(lnr)) > LINE_NOISE_Z_THRESH
                 if LINE_NOISE_Z_THRESH is not None else np.zeros(x_raw.shape[1], bool))
    bad_mask = flag_flat | flag_amp | flag_clip | flag_lnr
    bad_idx  = np.where(bad_mask)[0]
    good_idx = np.where(~bad_mask)[0]

    if len(bad_idx) == 0:
        print(f'  {s.upper()}: 0 bad channels — skipping')
        continue

    n_bad  = len(bad_idx)
    T_samp = min(int(T_SHOW * FS), x_raw.shape[0])
    t_ax   = np.arange(T_samp) / FS

    f_psd, all_psd = sig.welch(x_raw, fs=FS, nperseg=FS * 4, axis=0)  # (F, C)
    all_psd = all_psd.T  # (C, F)

    rms = np.sqrt((x_notch ** 2).mean(0))   # for trace spacing only

    def _reason(ch):
        r = []
        if flag_flat[ch]: r.append('flat')
        if flag_amp[ch]:  r.append(f'amp z={_mad_z(amp)[ch]:.1f}')
        if flag_clip[ch]: r.append('clip')
        if flag_lnr[ch]:  r.append('LN-SNR')
        return '+'.join(r)

    fig, (ax_psd, ax_tr) = plt.subplots(1, 2, figsize=(14, max(5, n_bad * 0.7 + 3)))
    fig.suptitle(f'{s.upper()} — bad channel review  ({n_bad}/{x_raw.shape[1]} flagged)',
                 fontsize=11, fontweight='bold')

    # PSD overlay: log-log so 1/f shows as straight line and spikes pop
    for ch in good_idx:
        ax_psd.loglog(f_psd[1:], all_psd[ch, 1:], color='#bbbbbb', lw=0.4, alpha=0.5)
    for ch in bad_idx:
        ax_psd.loglog(f_psd[1:], all_psd[ch, 1:], color='tomato', lw=1.2, alpha=0.9,
                      label=f'ch{ch+1} ({_reason(ch)})')
    for h in [60, 120, 180]:
        ax_psd.axvline(h, color='red', ls=':', lw=0.8, alpha=0.4)
    ax_psd.set_xlim(1, FS // 2)
    ax_psd.set_xlabel('Frequency (Hz)')
    ax_psd.set_ylabel('PSD (AU²/Hz)')
    ax_psd.set_title('PSD (raw) — gray=good  red=flagged')
    ax_psd.legend(fontsize=7, loc='upper right')

    # Stacked raw traces — scale to median RMS of good channels
    global_scale = np.median(rms[good_idx]) + 1e-9 if len(good_idx) else np.median(rms) + 1e-9
    slot = 10 * global_scale  # spacing between traces
    for i, ch in enumerate(bad_idx):
        trace  = x_raw[:T_samp, ch].astype(float)
        offset = (n_bad - 1 - i) * slot
        ax_tr.plot(t_ax, trace + offset, color='tomato', lw=0.5, alpha=0.85)
        ax_tr.text(-0.3, offset, f'ch{ch+1}\n({_reason(ch)})',
                   va='center', ha='right', fontsize=7, color='tomato')
    ax_tr.set_xlabel('Time (s)')
    ax_tr.set_title(f'Raw traces — first {T_SHOW} s')
    ax_tr.set_xlim(0, T_SHOW)
    ax_tr.set_yticks([])

    plt.tight_layout()
    path = f'{OUT_DIR}/figN8_{s}_bad_ch_review.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  {s.upper()}: {n_bad} flagged  → {path}')

print(f'\nAll noise figures saved to {OUT_DIR}')
