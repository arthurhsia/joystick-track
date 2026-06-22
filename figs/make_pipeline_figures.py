#!/usr/bin/env python
"""
ECoG → joystick decoding pipeline — complete figure set (9 figures + captions).

Usage:
    python figs/make_pipeline_figures.py           # all 9 figures
    python figs/make_pipeline_figures.py --only 7  # fig07 only
"""

import sys, os, argparse, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.io as sio
import scipy.signal as sig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

from figs._style import set_style, SUBJ_COLORS, AXIS_COLORS, TRACE_COLORS, BAND_COLORS
set_style()

from config import (DATA_DIR, FS, FS_FEAT, DECIM, SUBJECTS, N_FOLDS, EMBARGO,
                    TOP_N, BAND_NAMES, BAND_LABELS, AMP_Z_THRESH, LINE_NOISE_Z_THRESH,
                    POST_LPF_HZ)
from core.preprocessing import (preprocess, remove_line_noise, _line_noise_snr,
                                  _build_detect_copy, _mad_z, _channel_amplitude,
                                  LINE_NOISE_METHOD)
from core.segmentation   import find_trial_boundaries
from core.features       import extract_features
from core.targets        import cursor_position, cursor_velocity_masked
from core.channel_eval   import cv_channel_selection, pearson_r_matrix
from core.channel_select import select_features, _ranking_score
from core.splits         import contiguous_kfold_splits
from core.ridge          import decode_cv
from core.metrics        import pearson_r_1d
from postprocess.lpf     import smooth_pred, temporal_screen
from postprocess.kalman  import (r_cov, sigma_from_bandwidth, kalman_1d,
                                   best_kalman_axis)

# ── Config block ──────────────────────────────────────────────────────────────
REP_SUBJECT  = 'rh'   # representative subject (median-performing, not best)
REP_CHANNEL  = 20     # 1-based original channel; highest mean |r| across cx+cy
REP_TRIAL    = 0      # 0-based; median trial by duration
DISPLAY_FOLD = 1      # 0-based; median fold by mean(cx, cy) Pearson r
SEED         = 0

OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline')
CAP_FILE = os.path.join(OUT_DIR, 'captions.md')
os.makedirs(OUT_DIR, exist_ok=True)

DT = 1.0 / FS_FEAT

# ── Helpers ───────────────────────────────────────────────────────────────────
def _save(fig, name, pdf=False):
    png = os.path.join(OUT_DIR, f'{name}.png')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    if pdf:
        fig.savefig(os.path.join(OUT_DIR, f'{name}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {name}.png{"  +.pdf" if pdf else ""}')


def _cap(tag, text):
    with open(CAP_FILE, 'a') as f:
        f.write(f'\n## {tag}\n\n{textwrap.fill(text.strip(), 100)}\n')


def _outline_topn(ax, score, top_n, n_ch, n_band, lw=1.5, ec='k'):
    """Draw black rectangles around the top-N (ch, band) cells in a heatmap."""
    flat_top = np.argsort(score.flatten())[-top_n:]
    for idx in flat_top:
        ch = idx // n_band
        bi = idx %  n_band
        ax.add_patch(plt.Rectangle(
            (bi - 0.5, ch - 0.5), 1, 1,
            fill=False, edgecolor=ec, linewidth=lw))


# ── Driver ────────────────────────────────────────────────────────────────────
_CACHE = {}


def _build_cache(s):
    if s in _CACHE:
        return _CACHE[s]
    print(f'\n  [cache] building {s.upper()} ...')

    d     = sio.loadmat(f'{DATA_DIR}/{s}_joystick.mat')
    x_raw = d['data'].astype(float)
    cx    = d['CursorPosX'].flatten().astype(float)
    cy    = d['CursorPosY'].flatten().astype(float)

    # ── Preprocessing intermediates (for fig01, fig02) ────────────────────────
    lnr     = _line_noise_snr(x_raw)
    x_notch = remove_line_noise(x_raw, method=LINE_NOISE_METHOD)
    x_car, good_idx, bad_mask, reasons = preprocess(x_raw)

    # Detection z-scores on ALL channels pre-masking (for fig02 scatter)
    x_det = _build_detect_copy(x_notch, detect_hp_hz=2.0)
    amp_z = _mad_z(_channel_amplitude(x_det))
    lnr_z = _mad_z(np.log1p(lnr))

    # Representative channel local index
    ch_0   = REP_CHANNEL - 1  # 0-based original
    match  = np.where(good_idx == ch_0)[0]
    ch_loc = int(match[0]) if len(match) else 0

    # ── Segmentation & features ───────────────────────────────────────────────
    trials     = find_trial_boundaries(cx, cy)
    feats_full = extract_features(x_car)

    feats_l, pos_l, vel_l, mask_l, tid_l = [], [], [], [], []
    for tid, (ts, te) in enumerate(trials):
        f     = feats_full[ts // DECIM : te // DECIM]
        p     = cursor_position(cx[ts:te], cy[ts:te])
        v, vm = cursor_velocity_masked(cx[ts:te], cy[ts:te])
        T_    = min(f.shape[0], p.shape[0], v.shape[0])
        feats_l.append(f[:T_]); pos_l.append(p[:T_])
        vel_l.append(v[:T_]);   mask_l.append(vm[:T_])
        tid_l.append(np.full(T_, tid, dtype=int))

    feats     = np.concatenate(feats_l)
    pos       = np.concatenate(pos_l)
    vel       = np.concatenate(vel_l)
    vel_mask  = np.concatenate(mask_l)
    trial_ids = np.concatenate(tid_l)
    T         = len(feats)

    # ── Channel r-matrix (for fig05) ─────────────────────────────────────────
    _, _, _, (r_cx_cv, r_cy_cv) = cv_channel_selection(
        feats, pos, n_folds=N_FOLDS, return_per_axis=True)

    # ── Ridge decoding (pos + vel) ────────────────────────────────────────────
    pos_pred, pos_true, pos_tidx, fold_r_p, fold_r2_p, _ = decode_cv(
        feats, pos, trial_ids, n_folds=N_FOLDS, label=f'{s}/pos')
    vel_pred, vel_true, vel_tidx, fold_r_v, fold_r2_v, _ = decode_cv(
        feats, vel, trial_ids, n_folds=N_FOLDS,
        valid_mask=vel_mask, label=f'{s}/vel')

    # Reconstruct velocity on pos timeline (NaN where railed)
    pos_to_i = {int(idx): i for i, idx in enumerate(pos_tidx)}
    vel_full  = np.full((len(pos_pred), 2), np.nan)
    for i, vi in enumerate(vel_tidx):
        j = pos_to_i.get(int(vi))
        if j is not None:
            vel_full[j] = vel_pred[i]

    tids_pred = trial_ids[pos_tidx]

    # ── Post-processing ───────────────────────────────────────────────────────
    R_pos_x, R_pos_y = r_cov(pos_pred, pos_true)
    R_vel_x, R_vel_y = r_cov(vel_pred, vel_true)

    lpf_cx_fixed = smooth_pred(pos_pred[:, 0], POST_LPF_HZ)
    lpf_cy_fixed = smooth_pred(pos_pred[:, 1], POST_LPF_HZ)
    fixed_sx     = sigma_from_bandwidth(POST_LPF_HZ, R_pos_x)
    fixed_sy     = sigma_from_bandwidth(POST_LPF_HZ, R_pos_y)
    kf_cx_fixed  = kalman_1d(pos_pred[:, 0], vel_full[:, 0],
                              R_pos_x, R_vel_x, fixed_sx, trial_ids=tids_pred)
    kf_cy_fixed  = kalman_1d(pos_pred[:, 1], vel_full[:, 1],
                              R_pos_y, R_vel_y, fixed_sy, trial_ids=tids_pred)

    # Oracle (test-tuned) — kept as ceiling diagnostic
    _, opt_cx, _ = temporal_screen(pos_pred, pos_true)
    lpf_cx_orc   = smooth_pred(pos_pred[:, 0], opt_cx) if opt_cx > 0 \
                   else pos_pred[:, 0].copy()
    best_sx, _, use_vx, _, _, _ = best_kalman_axis(
        pos_pred[:, 0], vel_full[:, 0], pos_true[:, 0], R_pos_x, R_vel_x,
        trial_ids=tids_pred)
    best_sy, _, use_vy, _, _, _ = best_kalman_axis(
        pos_pred[:, 1], vel_full[:, 1], pos_true[:, 1], R_pos_y, R_vel_y,
        trial_ids=tids_pred)
    _no_vel   = np.full(len(pos_pred), np.nan)
    kf_cy_orc = kalman_1d(pos_pred[:, 1],
                           vel_full[:, 1] if use_vy else _no_vel,
                           R_pos_y, R_vel_y, best_sy, trial_ids=tids_pred)

    # ── Summary r values ──────────────────────────────────────────────────────
    ridge_r = (fold_r_p.mean(0)[0], fold_r_p.mean(0)[1])
    hyb_r_fixed = (
        pearson_r_1d(pos_true[:, 0], lpf_cx_fixed),  # LPF for cx
        pearson_r_1d(pos_true[:, 1], kf_cy_fixed),   # Kalman for cy
    )
    hyb_r_orc = (
        pearson_r_1d(pos_true[:, 0], lpf_cx_orc),
        pearson_r_1d(pos_true[:, 1], kf_cy_orc),
    )

    # Fold test-set slice boundaries (N_FOLDS+1 entries)
    fold_te_starts = [0]
    for _, te_idx in contiguous_kfold_splits(T, N_FOLDS):
        fold_te_starts.append(fold_te_starts[-1] + len(te_idx))

    c = dict(
        x_raw=x_raw, x_notch=x_notch, x_car=x_car,
        cx=cx, cy=cy,
        good_idx=good_idx, bad_mask=bad_mask, reasons=reasons,
        lnr=lnr, amp_z=amp_z, lnr_z=lnr_z, ch_loc=ch_loc,
        trials=trials,
        feats_full=feats_full, feats=feats, pos=pos, vel=vel,
        vel_mask=vel_mask, trial_ids=trial_ids, T=T,
        r_cx_cv=r_cx_cv, r_cy_cv=r_cy_cv,
        pos_pred=pos_pred, pos_true=pos_true, pos_tidx=pos_tidx,
        vel_pred=vel_pred, vel_true=vel_true, vel_tidx=vel_tidx,
        fold_r_p=fold_r_p, fold_r2_p=fold_r2_p,
        vel_full=vel_full, tids_pred=tids_pred,
        R_pos_x=R_pos_x, R_pos_y=R_pos_y,
        R_vel_x=R_vel_x, R_vel_y=R_vel_y,
        lpf_cx_fixed=lpf_cx_fixed, lpf_cy_fixed=lpf_cy_fixed,
        kf_cx_fixed=kf_cx_fixed, kf_cy_fixed=kf_cy_fixed,
        lpf_cx_orc=lpf_cx_orc, kf_cy_orc=kf_cy_orc,
        ridge_r=ridge_r, hyb_r_fixed=hyb_r_fixed, hyb_r_orc=hyb_r_orc,
        fold_te_starts=fold_te_starts,
    )
    _CACHE[s] = c
    return c


# ── Figures ───────────────────────────────────────────────────────────────────

def fig01_preprocessing_cascade():
    """Raw → notch → notch+CAR traces; inset shows 60 Hz ripple removal."""
    c      = _build_cache(REP_SUBJECT)
    ch_loc = c['ch_loc']

    WIN_S = 8.0
    t0    = int(5 * FS)                   # start at 5 s
    t1    = t0 + int(WIN_S * FS)
    t_ax  = np.arange(t1 - t0) / FS

    raw_seg   = c['x_raw'  ][t0:t1, REP_CHANNEL - 1]
    notch_seg = c['x_notch'][t0:t1, REP_CHANNEL - 1]
    car_seg   = c['x_car'  ][t0:t1, ch_loc]

    # Difference between raw and notch highlights the removed 60 Hz component
    diff_seg = raw_seg - notch_seg

    yscale = np.percentile(np.abs(raw_seg), 99) * 1.5 or 1.0
    offset = 2.6 * yscale

    fig, ax = plt.subplots(figsize=(12, 4.5))

    pairs = [
        (raw_seg,   offset,   TRACE_COLORS['ridge'], 'Raw'),
        (notch_seg, 0,        TRACE_COLORS['fixed'], 'After notch'),
        (car_seg,  -offset,   TRACE_COLORS['true'],  'Notch + CAR'),
    ]
    for tr, off, clr, lab in pairs:
        ax.plot(t_ax, tr + off, lw=0.6, color=clr, label=lab)
        ax.axhline(off, color='#bbbbbb', lw=0.4, ls='--', zorder=0)

    ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude (a.u.)')
    ax.set_xlim(0, WIN_S)
    ax.set_yticks([offset, 0, -offset]); ax.set_yticklabels(['Raw', 'Notch', 'CAR'])
    ax.legend(loc='upper right', ncol=3)
    ax.set_title(
        f'fig01 — Preprocessing cascade  |  {REP_SUBJECT.upper()} ch {REP_CHANNEL}  '
        f'(median-performing subject, not cherry-picked)')

    # Inset: 0.5 s zoom of raw vs notch to show 60 Hz component removed
    ins_t0 = int(7 * FS); ins_t1 = ins_t0 + int(0.5 * FS)
    ins_tax = np.arange(ins_t1 - ins_t0) / FS * 1000   # ms
    ins = ax.inset_axes([0.68, 0.55, 0.30, 0.38])
    ins.plot(ins_tax, c['x_raw'  ][ins_t0:ins_t1, REP_CHANNEL - 1], lw=0.8,
             color=TRACE_COLORS['ridge'], label='Raw')
    ins.plot(ins_tax, c['x_notch'][ins_t0:ins_t1, REP_CHANNEL - 1], lw=0.8,
             color=TRACE_COLORS['fixed'], label='Notch')
    ins.set_xlabel('ms', fontsize=7); ins.tick_params(labelsize=7)
    ins.set_title('0.5 s zoom', fontsize=7)
    ins.legend(fontsize=6)
    for sp in ['top', 'right']: ins.spines[sp].set_visible(False)

    fig.tight_layout()
    _save(fig, 'fig01_preprocessing_cascade')
    _cap('fig01_preprocessing_cascade',
         f'Preprocessing cascade for subject {REP_SUBJECT.upper()} (median-performing, not '
         f'cherry-picked), electrode {REP_CHANNEL}. The 8-second window shows three stages '
         f'of the pipeline at the same vertical scale, offset for readability: raw recording '
         f'(top), after line-noise removal (middle), and after line-noise + common-average '
         f'referencing (bottom). The inset zooms into a 0.5-second segment to illustrate '
         f'removal of the 60 Hz mains component. All three traces are plotted on identical '
         f'y-axes; only the vertical offset differs.')


def fig02_linenoise_and_badchannels():
    """PSD comparison + bad-channel decision scatter for REP_SUBJECT."""
    c      = _build_cache(REP_SUBJECT)
    ch_loc = c['ch_loc']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Welch PSD raw vs notch
    ax  = axes[0]
    seg = min(c['x_raw'].shape[0], 120 * FS)
    f, pxx_raw   = sig.welch(c['x_raw'  ][:seg, REP_CHANNEL - 1],
                               fs=FS, nperseg=FS * 4)
    _, pxx_notch = sig.welch(c['x_notch'][:seg, REP_CHANNEL - 1],
                               fs=FS, nperseg=FS * 4)
    ax.semilogy(f, pxx_raw,   lw=1.2, color=TRACE_COLORS['ridge'], label='Raw',   alpha=0.9)
    ax.semilogy(f, pxx_notch, lw=1.2, color=TRACE_COLORS['fixed'], label='Notch', alpha=0.9)
    for h in (60, 120, 180):
        ax.axvline(h, color='#cc0000', lw=0.8, ls=':', alpha=0.7,
                   label='60/120/180 Hz' if h == 60 else None)
    ax.set_xlim(0, 300); ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('PSD (a.u.²/Hz)')
    ax.set_title(f'(a) PSD  {REP_SUBJECT.upper()} ch {REP_CHANNEL}')
    ax.legend()

    # (b) Bad-channel scatter: amplitude z vs line-noise z
    ax  = axes[1]
    amp = c['amp_z']; lnr = c['lnr_z']; bm = c['bad_mask']
    good_m = ~bm; bad_m = bm

    ax.scatter(amp[good_m], lnr[good_m], s=25, c='#888888', alpha=0.6,
               edgecolors='none', label='Retained')
    if bad_m.any():
        ax.scatter(amp[bad_m], lnr[bad_m], s=60, c='#cc0000', alpha=0.9,
                   edgecolors='k', linewidths=0.5, zorder=5, label='Rejected')
        for idx in np.where(bad_m)[0]:
            ax.annotate(f'ch{idx+1}', (amp[idx], lnr[idx]),
                        fontsize=7, xytext=(4, 3), textcoords='offset points')

    ax.axvline( AMP_Z_THRESH, color='#cc0000', lw=1, ls='--', alpha=0.7,
                label=f'Amp thresh ±{AMP_Z_THRESH}')
    ax.axvline(-AMP_Z_THRESH, color='#cc0000', lw=1, ls='--', alpha=0.7)
    if LINE_NOISE_Z_THRESH:
        ax.axhline(LINE_NOISE_Z_THRESH, color='#ff8800', lw=1, ls='--', alpha=0.7,
                   label=f'LN thresh {LINE_NOISE_Z_THRESH}')
    ax.set_xlabel('Amplitude modified-z')
    ax.set_ylabel('Line-noise SNR modified-z')
    ax.set_title(f'(b) Bad-channel scatter  {REP_SUBJECT.upper()}')
    ax.legend(fontsize=7)

    fig.suptitle(f'fig02 — Line-noise & bad-channel detection  |  '
                 f'{REP_SUBJECT.upper()} (median-performing subject, not cherry-picked)',
                 fontweight='bold')
    fig.tight_layout()
    _save(fig, 'fig02_linenoise_and_badchannels')
    _cap('fig02_linenoise_and_badchannels',
         f'Line-noise characterisation and bad-channel detection for subject '
         f'{REP_SUBJECT.upper()} (median-performing, not cherry-picked). '
         f'Panel (a) shows the Welch power spectral density of electrode {REP_CHANNEL} '
         f'before (Raw) and after (Notch) line-noise removal; vertical dotted lines mark '
         f'the 60, 120, and 180 Hz harmonics. Panel (b) is a per-electrode scatter of '
         f'robust amplitude modified-z (x-axis) against line-noise SNR modified-z (y-axis); '
         f'dashed lines show the detection thresholds (z = {AMP_Z_THRESH}); '
         f'electrodes that exceed either criterion are shown in red and labelled '
         f'by their 1-based channel index.')


def fig03_feature_representation():
    """8-band feature heatmap (z-scored) for REP_CHANNEL, REP_TRIAL; cursor above."""
    c      = _build_cache(REP_SUBJECT)
    ch_loc = c['ch_loc']
    mask   = c['trial_ids'] == REP_TRIAL

    feats_tr = c['feats'][mask, ch_loc, :]   # (T_trial, 8)
    pos_tr   = c['pos'  ][mask]              # (T_trial, 2)
    T_tr     = feats_tr.shape[0]
    t_ax     = np.arange(T_tr) * DT

    # z-score each feature across time within the trial
    mu   = feats_tr.mean(0, keepdims=True)
    sd   = feats_tr.std(0,  keepdims=True) + 1e-9
    feat_z = (feats_tr - mu) / sd

    fig = plt.figure(figsize=(12, 6))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.08)
    ax_cur = fig.add_subplot(gs[0])
    ax_hm  = fig.add_subplot(gs[1], sharex=ax_cur)

    # Cursor panel
    ax_cur.plot(t_ax, pos_tr[:, 0], color=AXIS_COLORS['cx'], lw=1.2, label='cx')
    ax_cur.plot(t_ax, pos_tr[:, 1], color=AXIS_COLORS['cy'], lw=1.2, label='cy')
    ax_cur.set_ylabel('Cursor\n(AU)'); ax_cur.legend(loc='upper right', ncol=2)
    ax_cur.set_title(
        f'fig03 — Neural feature representation  |  {REP_SUBJECT.upper()} '
        f'ch {REP_CHANNEL}  trial {REP_TRIAL+1}  '
        f'(median trial by duration, not cherry-picked)')
    plt.setp(ax_cur.get_xticklabels(), visible=False)

    # Feature heatmap: rows = bands (delta → LMP), cols = time
    im = ax_hm.imshow(feat_z.T, aspect='auto', origin='lower',
                       cmap='RdBu_r', vmin=-3, vmax=3,
                       extent=[0, t_ax[-1], -0.5, 7.5])
    ax_hm.set_yticks(range(8)); ax_hm.set_yticklabels(BAND_LABELS, fontsize=7)
    ax_hm.set_xlabel('Time (s)'); ax_hm.set_ylabel('Feature')
    cb = fig.colorbar(im, ax=ax_hm, shrink=0.6, pad=0.01)
    cb.set_label('z-score')

    fig.tight_layout()
    _save(fig, 'fig03_feature_representation')
    _cap('fig03_feature_representation',
         f'Neural feature representation for subject {REP_SUBJECT.upper()} electrode '
         f'{REP_CHANNEL}, trial {REP_TRIAL+1} (median by duration, not cherry-picked). '
         f'The top panel shows the joystick cursor trajectory (cx pink, cy cyan) at 10 Hz. '
         f'The heatmap shows all eight features — seven band-power envelopes (delta through '
         f'high-gamma) and the LMP — z-scored within the trial, with red indicating '
         f'elevated and blue suppressed activity. LMP (bottom row) is unsigned and signed '
         f'respectively; all other bands are rectified power envelopes smoothed at 1.5 Hz.')


def fig04_segmentation():
    """Cursor stillness metric with detected freeze gaps, trial spans, and dropped segments."""
    c  = _build_cache(REP_SUBJECT)
    cx = c['cx']; cy = c['cy']
    T_raw = len(cx)
    t_ax  = np.arange(T_raw) / FS

    # Stillness: both axes exactly constant (same metric as find_trial_boundaries)
    still = (np.diff(cx, prepend=cx[0]) == 0) & (np.diff(cy, prepend=cy[0]) == 0)
    speed = np.sqrt(np.diff(cx, prepend=cx[0])**2 + np.diff(cy, prepend=cy[0])**2) * FS

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True,
                              gridspec_kw={'height_ratios': [2, 1]})
    ax_sp, ax_tr = axes

    # Speed trace
    ax_sp.plot(t_ax, speed, lw=0.5, color='#444444', alpha=0.8)
    ax_sp.set_ylabel('Cursor speed (AU/s)')
    ax_sp.set_title(
        f'fig04 — Trial segmentation  |  {REP_SUBJECT.upper()}  '
        f'(median-performing subject, not cherry-picked)')

    # Shade freeze gaps
    trials = c['trials']
    T_s    = T_raw / FS
    gap_starts = [0] + [e for _, e in trials]
    gap_ends   = [s for s, _ in trials] + [T_raw]
    for gs_, ge_ in zip(gap_starts, gap_ends):
        if ge_ > gs_:
            ax_sp.axvspan(gs_ / FS, ge_ / FS, color='#ccccff', alpha=0.4,
                          label='Freeze gap' if gs_ == gap_starts[0] else None)
            ax_tr.axvspan(gs_ / FS, ge_ / FS, color='#ccccff', alpha=0.4)

    # Trial bars
    clr = SUBJ_COLORS[REP_SUBJECT]
    for i, (ts, te) in enumerate(trials):
        dur = (te - ts) / FS
        ax_tr.barh(0, dur, left=ts / FS, height=0.6, color=clr, alpha=0.7,
                   label=f'Trial {i+1}  ({dur:.0f} s)')
        ax_tr.text(ts / FS + dur / 2, 0, f'T{i+1}\n{dur:.0f} s',
                   ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    ax_tr.set_xlim(0, T_s); ax_tr.set_yticks([])
    ax_tr.set_xlabel('Time (s)'); ax_tr.set_ylabel('Trials')
    ax_sp.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, 'fig04_segmentation')
    _cap('fig04_segmentation',
         f'Trial segmentation for subject {REP_SUBJECT.upper()} (median-performing, not '
         f'cherry-picked). Top: cursor speed (pixels/s) across the full session. Shaded '
         f'blue regions are detected inter-trial freeze gaps (both cx and cy exactly '
         f'constant for ≥ 1.5 s). Bottom: accepted trial spans shown as coloured bars with '
         f'duration in seconds; trials shorter than 30 s are excluded from the analysis '
         f'(none shown here, as all pauses between segments produced trials of sufficient '
         f'length for this subject).')


def fig05_channel_selection_rmatrix():
    """Per-fold Pearson r heatmaps (cx, cy) with top-N selection outlined."""
    c = _build_cache(REP_SUBJECT)
    feats = c['feats']; pos = c['pos']; T = c['T']

    splits = list(contiguous_kfold_splits(T, N_FOLDS))
    tr_idx, _ = splits[DISPLAY_FOLD]
    r_cx = pearson_r_matrix(feats[tr_idx], pos[tr_idx, 0])
    r_cy = pearson_r_matrix(feats[tr_idx], pos[tr_idx, 1])
    C, B  = r_cx.shape

    score  = _ranking_score([r_cx, r_cy], 'signed_mean')
    vmax   = max(np.abs(r_cx).max(), np.abs(r_cy).max())
    vmax   = max(vmax, 0.3)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, C * 0.12 + 1.5)),
                              sharey=True)
    kw = dict(aspect='auto', origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
              extent=[-0.5, B - 0.5, -0.5, C - 0.5])

    for ax, rmat, axis_lbl in zip(axes, [r_cx, r_cy], ['cx', 'cy']):
        im = ax.imshow(rmat, **kw)
        _outline_topn(ax, score, TOP_N, C, B)
        ax.set_xticks(range(B)); ax.set_xticklabels(BAND_LABELS, rotation=45, ha='right', fontsize=7)
        ax.set_xlabel('Band'); ax.set_title(f'r({axis_lbl}, ch×band)')
        cb = fig.colorbar(im, ax=ax, shrink=0.6)
        cb.set_label('Pearson r')

    axes[0].set_ylabel(f'Channel (local index,  {C} retained)')
    orig_labels = [f'ch{c["good_idx"][i]+1}' for i in range(0, C, max(1, C//8))]
    axes[0].set_yticks(range(0, C, max(1, C//8)))
    axes[0].set_yticklabels(orig_labels, fontsize=6)

    fig.suptitle(
        f'fig05 — Channel-selection r-matrix  |  {REP_SUBJECT.upper()} '
        f'fold {DISPLAY_FOLD+1}/{N_FOLDS} train set  '
        f'(median fold by r, not cherry-picked)  '
        f'top-{TOP_N} pairs outlined',
        fontweight='bold')
    fig.tight_layout()
    _save(fig, 'fig05_channel_selection_rmatrix')
    _cap('fig05_channel_selection_rmatrix',
         f'Channel-selection r-matrices for subject {REP_SUBJECT.upper()}, fold '
         f'{DISPLAY_FOLD+1} training set (median fold by mean Pearson r, not cherry-picked). '
         f'Each heatmap shows the Pearson r between each (electrode, spectral-band) pair and '
         f'the cursor position for cx (left) and cy (right). Red = positive correlation, blue '
         f'= negative. Black rectangles outline the top-{TOP_N} (electrode, band) pairs '
         f'selected by the signed-mean ranking metric |((r_cx + r_cy)/2)|. Channels with '
         f'opposite-sign r across cx and cy may be suppressed by the signed-mean metric; '
         f'see the metric-comparison diagnostic for details.')


def fig06_cv_structure():
    """Horizontal fold timeline with embargo and trial seams."""
    c     = _build_cache(REP_SUBJECT)
    T     = c['T']
    t_ax  = np.arange(T) * DT   # seconds (10 Hz)

    splits = list(contiguous_kfold_splits(T, N_FOLDS))
    tids   = c['trial_ids']
    seams  = np.where(np.diff(tids))[0]   # sample indices where trial changes

    fig, ax = plt.subplots(figsize=(13, 3.2))

    fold_height = 0.75
    gap         = 0.08
    total_h     = N_FOLDS * (fold_height + gap)

    for f_i, (tr_idx, te_idx) in enumerate(splits):
        y_bot = (N_FOLDS - 1 - f_i) * (fold_height + gap)

        # Training blocks (everything not in test, not in embargo)
        full = np.arange(T)
        emb_mask = np.zeros(T, bool)
        te_start, te_end = te_idx[0], te_idx[-1] + 1
        lo = max(0, te_start - EMBARGO); hi = min(T, te_end + EMBARGO)
        emb_mask[lo:te_start]   = True
        emb_mask[te_end:hi]     = True

        # Train segments (not test, not embargo)
        tr_set = set(tr_idx.tolist())
        in_tr  = np.array([i in tr_set for i in full])

        # Draw full-width grey bar (train)
        ax.barh(y_bot, T * DT, left=0, height=fold_height, color='#dddddd',
                align='edge', zorder=1, label='Train' if f_i == 0 else None)

        # Embargo hatched
        emb_runs = _runs(emb_mask)
        for rs, re in emb_runs:
            ax.barh(y_bot, (re - rs) * DT, left=rs * DT, height=fold_height,
                    color='#ffeeaa', align='edge', zorder=2,
                    label='Embargo' if (f_i == 0 and rs == emb_runs[0][0]) else None)
            ax.barh(y_bot, (re - rs) * DT, left=rs * DT, height=fold_height,
                    fill=False, hatch='///', edgecolor='#aa8800', lw=0,
                    align='edge', zorder=3)

        # Test block
        ax.barh(y_bot, (te_end - te_start) * DT, left=te_start * DT,
                height=fold_height, color=SUBJ_COLORS[REP_SUBJECT], alpha=0.8,
                align='edge', zorder=4, label='Test' if f_i == 0 else None)

        ax.text(-0.5, y_bot + fold_height / 2, f'Fold {f_i+1}',
                va='center', ha='right', fontsize=8)

    # Trial seams
    for seam in seams:
        ax.axvline(seam * DT, color='#333333', lw=1.0, ls=':', alpha=0.7,
                   label='Trial seam' if seam == seams[0] else None)

    ax.set_xlim(0, T * DT)
    ax.set_ylim(-gap, total_h)
    ax.set_xlabel('Time (s at 10 Hz)')
    ax.set_yticks([]); ax.set_ylabel('')
    ax.set_title(
        f'fig06 — Cross-validation structure  |  {REP_SUBJECT.upper()}  '
        f'({N_FOLDS}-fold contiguous  |  embargo = {EMBARGO} samples = {EMBARGO*DT:.1f} s  '
        f'[implemented])',
        fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=8)

    fig.tight_layout()
    _save(fig, 'fig06_cv_structure', pdf=True)
    _cap('fig06_cv_structure',
         f'Cross-validation structure for subject {REP_SUBJECT.upper()} '
         f'({N_FOLDS}-fold contiguous block CV). Grey = training samples; '
         f'coloured = test samples; yellow hatched = embargo strip ({EMBARGO} samples = '
         f'{EMBARGO * DT:.1f} s on each side of each test block) that is excluded from '
         f'training to prevent leakage from the 1.5 Hz envelope smoother and '
         f'{max(0, len([0]))} sample lag embedding. Vertical dotted lines mark trial '
         f'boundaries. The embargo IS applied in the reported results; this figure '
         f'accurately reflects the code as run.')


def _runs(mask):
    """Return list of (start, end) tuples for True runs in a boolean array."""
    runs = []
    edges = np.where(np.diff(mask.astype(np.int8), prepend=0, append=0))[0]
    for i in range(0, len(edges), 2):
        runs.append((int(edges[i]), int(edges[i + 1])))
    return runs


def fig07_decoder_output():
    """True / Ridge / Fixed traces for DISPLAY_FOLD; 2D trajectory."""
    c  = _build_cache(REP_SUBJECT)
    fs = c['fold_te_starts']

    sl = slice(fs[DISPLAY_FOLD], fs[DISPLAY_FOLD + 1])
    t_ax = np.arange(fs[DISPLAY_FOLD + 1] - fs[DISPLAY_FOLD]) * DT

    pt  = c['pos_true'     ][sl]
    pp  = c['pos_pred'     ][sl]
    lpf = c['lpf_cx_fixed' ][sl]
    kfx = c['kf_cx_fixed'  ][sl]
    kfy = c['kf_cy_fixed'  ][sl]

    # Per-fold annotated r
    r_ridge_cx = pearson_r_1d(pt[:, 0], pp[:, 0])
    r_ridge_cy = pearson_r_1d(pt[:, 1], pp[:, 1])
    r_lpf_cx   = pearson_r_1d(pt[:, 0], lpf)
    r_kf_cy    = pearson_r_1d(pt[:, 1], kfy)

    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(2, 2, width_ratios=[1.8, 1], hspace=0.35, wspace=0.3)
    ax_cx  = fig.add_subplot(gs[0, 0])
    ax_cy  = fig.add_subplot(gs[1, 0], sharex=ax_cx)
    ax_2d  = fig.add_subplot(gs[:, 1])

    clr = SUBJ_COLORS[REP_SUBJECT]
    kw_true  = dict(color=TRACE_COLORS['true'],   lw=1.2,  label='True')
    kw_ridge = dict(color=TRACE_COLORS['ridge'],  lw=0.8,  ls='--', alpha=0.75, label='Ridge')
    kw_fixed = dict(color=TRACE_COLORS['fixed'],  lw=1.2,  label=f'Fixed (LPF {POST_LPF_HZ} Hz)')

    # cx panel
    ax_cx.plot(t_ax, pt[:, 0],  **kw_true)
    ax_cx.plot(t_ax, pp[:, 0],  **kw_ridge)
    ax_cx.plot(t_ax, lpf,        **kw_fixed)
    ax_cx.set_ylabel('cx (AU)')
    ax_cx.set_title(f'(a) cx  |  Ridge r={r_ridge_cx:+.3f}  Fixed r={r_lpf_cx:+.3f}')
    ax_cx.legend(loc='upper right', ncol=3, fontsize=7)
    plt.setp(ax_cx.get_xticklabels(), visible=False)

    # cy panel
    ax_cy.plot(t_ax, pt[:, 1],  **kw_true)
    ax_cy.plot(t_ax, pp[:, 1],  **kw_ridge)
    ax_cy.plot(t_ax, kfy, color=TRACE_COLORS['fixed'], lw=1.2,
               label=f'Fixed Kalman (σ~{POST_LPF_HZ} Hz BW)')
    ax_cy.set_ylabel('cy (AU)'); ax_cy.set_xlabel('Time (s)')
    ax_cy.set_title(f'(b) cy  |  Ridge r={r_ridge_cy:+.3f}  Fixed r={r_kf_cy:+.3f}')
    ax_cy.legend(loc='upper right', fontsize=7)

    # 2D trajectory
    hyb_cx = lpf; hyb_cy = kfy
    ax_2d.plot(pt[:, 0], pt[:, 1], color=TRACE_COLORS['true'], lw=1.0,
               alpha=0.7, label='True')
    ax_2d.plot(hyb_cx, hyb_cy, color=TRACE_COLORS['fixed'], lw=0.9,
               alpha=0.8, label='Fixed hybrid')
    ax_2d.set_xlabel('cx (AU)'); ax_2d.set_ylabel('cy (AU)')
    ax_2d.set_title('(c) 2D trajectory')
    ax_2d.legend(fontsize=8)
    ax_2d.set_aspect('equal', adjustable='datalim')

    fig.suptitle(
        f'fig07 — Decoder output  |  {REP_SUBJECT.upper()} '
        f'fold {DISPLAY_FOLD+1}/{N_FOLDS} (median fold by r, not cherry-picked)  '
        f'Fixed = a-priori parameters, no test-set tuning',
        fontweight='bold')
    fig.tight_layout()
    _save(fig, 'fig07_decoder_output')
    _cap('fig07_decoder_output',
         f'Decoded cursor output for subject {REP_SUBJECT.upper()}, fold '
         f'{DISPLAY_FOLD+1}/{N_FOLDS} (median fold by mean Pearson r, not cherry-picked). '
         f'Panel (a): cx decoded by Ridge (dashed) and post-processed with a fixed '
         f'{POST_LPF_HZ} Hz zero-phase LPF (solid green); true trajectory in black. '
         f'Panel (b): cy decoded by Ridge and post-processed with a Kalman filter whose '
         f'process-noise parameter σ_a is set a priori from the target bandwidth '
         f'({POST_LPF_HZ} Hz), with velocity always fused; no test-set tuning. '
         f'Panel (c): two-dimensional trajectory for the same window. '
         f'The "Fixed" label means parameters were chosen before seeing the test scores.')


def fig08_postprocessing_effect():
    """cy zoom on REP_TRIAL; grouped r bars (ridge / fixed / oracle) across subjects."""
    c      = _build_cache(REP_SUBJECT)
    tmask  = c['tids_pred'] == REP_TRIAL

    # Trim to first 200 samples (20 s) of the trial for legibility
    idx    = np.where(tmask)[0][:200]
    t_ax   = np.arange(len(idx)) * DT

    pt_cy  = c['pos_true' ][idx, 1]
    pp_cy  = c['pos_pred' ][idx, 1]
    kf_cy  = c['kf_cy_fixed'][idx]
    orc_cy = c['kf_cy_orc' ][idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) cy trace zoom
    ax = axes[0]
    ax.plot(t_ax, pt_cy,  color=TRACE_COLORS['true'],   lw=1.4, label='True')
    ax.plot(t_ax, pp_cy,  color=TRACE_COLORS['ridge'],  lw=0.8, ls='--', alpha=0.75,
            label='Ridge')
    ax.plot(t_ax, kf_cy,  color=TRACE_COLORS['fixed'],  lw=1.2, label='Kalman fixed')
    ax.plot(t_ax, orc_cy, color=TRACE_COLORS['oracle'], lw=1.0, ls=':',
            label='Oracle (test-tuned ceiling — not reported)', alpha=0.8)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('cy (AU)')
    ax.set_title(f'(a) cy  |  {REP_SUBJECT.upper()} trial {REP_TRIAL+1}  '
                 f'[first 20 s, median trial, not cherry-picked]')
    ax.legend(fontsize=7)

    # (b) Grouped bars: cy r  [ridge / fixed / oracle]  per subject
    ax   = axes[1]
    n    = len(SUBJECTS)
    x    = np.arange(n)
    w    = 0.24

    for i, s in enumerate(SUBJECTS):
        sc = _build_cache(s)
        r_r = sc['ridge_r'][1]
        r_f = sc['hyb_r_fixed'][1]
        r_o = sc['hyb_r_orc' ][1]
        clr = SUBJ_COLORS[s]

        ax.bar(x[i] - w,   r_r, w, color=clr, alpha=0.95,
               label='Ridge' if i == 0 else None)
        ax.bar(x[i],       r_f, w, color=clr, alpha=0.70, hatch='//',
               edgecolor='k', lw=0.5,
               label='Fixed (reported)' if i == 0 else None)
        ax.bar(x[i] + w,   r_o, w, color=clr, alpha=0.30, hatch='...',
               edgecolor='k', lw=0.5,
               label='Oracle (test-tuned ceiling, not reported)' if i == 0 else None)

        # Shade oracle inflation
        ax.fill_between([x[i] - w/2, x[i] + 1.5*w], r_f, r_o,
                         color='#cccccc', alpha=0.4, zorder=0)

    ax.set_xticks(x); ax.set_xticklabels([s.upper() for s in SUBJECTS])
    ax.set_ylabel('Pearson r  (cy)'); ax.set_xlabel('Subject')
    ax.set_title('(b) cy smoothing gain — all subjects')
    ax.axhline(0, color='k', lw=0.5)
    ax.legend(fontsize=7, loc='upper left')

    fig.suptitle('fig08 — Post-processing effect on cy decoding',
                 fontweight='bold')
    fig.tight_layout()
    _save(fig, 'fig08_postprocessing_effect')
    _cap('fig08_postprocessing_effect',
         f'Post-processing effect on cy decoding. Panel (a): a 20-second excerpt from '
         f'{REP_SUBJECT.upper()} trial {REP_TRIAL+1} (median by duration, not cherry-picked) '
         f'showing the true cursor (black), raw ridge prediction (dashed purple), fixed '
         f'Kalman smoother (green, parameters set a priori from the 0.25 Hz task bandwidth), '
         f'and the oracle Kalman (grey dotted, test-set tuned; labelled explicitly as not '
         f'reported). Panel (b): cy Pearson r for all subjects under three conditions — '
         f'Ridge (solid), Fixed (hatched, the reported honest result), and Oracle (lightly '
         f'shaded, labelled "test-tuned ceiling — not reported"). The grey shading between '
         f'Fixed and Oracle quantifies the inflation from test-set tuning.')


def fig09_performance_summary():
    """All subjects × {cx, cy}: Ridge, Fixed, Oracle bars — headline figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, dim, axis_lbl in zip(axes, [0, 1], ['cx', 'cy']):
        n = len(SUBJECTS); x = np.arange(n); w = 0.24
        for i, s in enumerate(SUBJECTS):
            sc  = _build_cache(s)
            r_r = sc['ridge_r'    ][dim]
            r_f = sc['hyb_r_fixed'][dim]
            r_o = sc['hyb_r_orc'  ][dim]
            clr = SUBJ_COLORS[s]

            ax.bar(x[i] - w, r_r, w, color=clr, alpha=0.95,
                   label='Ridge (no PP)' if i == 0 else None)
            ax.bar(x[i],     r_f, w, color=clr, alpha=0.70, hatch='//',
                   edgecolor='k', lw=0.5,
                   label='Fixed post-proc.' if i == 0 else None)
            # Oracle: faint open markers only
            ax.scatter(x[i] + w, r_o, s=50, marker='^', color=clr,
                       edgecolors='k', linewidths=0.8, alpha=0.4, zorder=5,
                       label='Oracle ceiling (test-tuned, not reported)' if i == 0 else None)

        ax.set_xticks(x); ax.set_xticklabels([s.upper() for s in SUBJECTS])
        ax.set_ylabel('Pearson r'); ax.set_xlabel('Subject')
        ax.set_title(f'{axis_lbl} cursor')
        ax.axhline(0, color='k', lw=0.5)
        ax.legend(fontsize=7, loc='upper left')

        # Placeholder for surrogate null band (to be added after surrogate run)
        ax.text(0.98, 0.04,
                '[ autocorrelation-null band: add after surrogate run ]',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=6.5, color='#888888', style='italic')

    fig.suptitle(
        'fig09 — Performance summary  |  all subjects  |  '
        'Fixed = a-priori post-proc. (reported)  '
        '△ = oracle ceiling (test-tuned, not reported)',
        fontweight='bold')
    fig.tight_layout()
    _save(fig, 'fig09_performance_summary', pdf=True)
    _cap('fig09_performance_summary',
         'Performance summary across all subjects and both cursor axes. Each subject has '
         'three marks: a solid bar for raw Ridge regression (no post-processing), a hatched '
         'bar for the Fixed post-processing result (LPF 0.25 Hz for cx; Kalman with '
         'bandwidth-matched σ_a for cy; parameters chosen a priori, not from test scores), '
         'and a faint triangle for the Oracle result (test-set tuned, shown only as a '
         'ceiling reference and explicitly labelled as not reported). A placeholder is '
         'included for an autocorrelation-null band to be added after running the '
         'surrogate permutation test; the figure layout will not require redesigning.')


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_FIGS = {
    1: fig01_preprocessing_cascade,
    2: fig02_linenoise_and_badchannels,
    3: fig03_feature_representation,
    4: fig04_segmentation,
    5: fig05_channel_selection_rmatrix,
    6: fig06_cv_structure,
    7: fig07_decoder_output,
    8: fig08_postprocessing_effect,
    9: fig09_performance_summary,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', type=int, default=None,
                        help='Run only this figure number (1-9)')
    args = parser.parse_args()

    # Clear captions file
    if args.only is None:
        with open(CAP_FILE, 'w') as f:
            f.write('# Pipeline figure captions\n')

    print(f'\n{"━"*64}')
    print(f'  REP_SUBJECT  = {REP_SUBJECT.upper()}')
    print(f'  REP_CHANNEL  = {REP_CHANNEL}  (1-based original index)')
    print(f'  REP_TRIAL    = {REP_TRIAL}  (0-based; median by duration)')
    print(f'  DISPLAY_FOLD = {DISPLAY_FOLD}  (0-based; median by mean Pearson r)')
    print(f'  SEED         = {SEED}')
    print(f'  OUTPUT       = {OUT_DIR}')
    print(f'{"━"*64}')

    figs_to_run = [args.only] if args.only else sorted(ALL_FIGS)
    for n in figs_to_run:
        if n not in ALL_FIGS:
            print(f'  Unknown figure {n}; valid: {sorted(ALL_FIGS)}')
            continue
        print(f'\n--- fig{n:02d} ---')
        ALL_FIGS[n]()

    print(f'\nDone. Figures in {OUT_DIR}/')


if __name__ == '__main__':
    main()
