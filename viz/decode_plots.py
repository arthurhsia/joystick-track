"""
Position decoding result plots: fold r, summary bar, traces, trajectory, coefficients.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from config import SUBJECTS, COLORS, BAND_NAMES, N_FOLDS, FS_FEAT, TOP_N, LAGS
from core.channel_select import select_features


def plot_fold_r(results, out_dir, model='ridge'):
    """figD1: Decoder Pearson r per subject per fold."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Decoder Pearson r  [{model}]  —  {N_FOLDS}-fold contiguous CV',
                 fontsize=13, fontweight='bold')
    for ax, component, label in zip(axes, [0, 1], ['cx  (horizontal)', 'cy  (vertical)']):
        for i, s in enumerate(SUBJECTS):
            per_fold = results[s]['fold_dec_r'][:, component]
            x = np.arange(1, N_FOLDS + 1) + i * 0.15 - 0.3
            ax.plot(x, per_fold, 'o-', color=COLORS[s], lw=1.5, label=s.upper())
            ax.axhline(per_fold.mean(), color=COLORS[s], ls='--', lw=0.8, alpha=0.5)
        ax.set_xlabel('Fold (held-out)'); ax.set_ylabel('r')
        ax.set_title(f'Component: {label}')
        ax.set_xticks(range(1, N_FOLDS + 1))
        ax.axhline(0, color='k', lw=0.8, ls='-', alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/figD1_decode_r_folds.png', dpi=150)
    plt.close()
    print('Saved figD1_decode_r_folds.png')


def plot_r_summary_bar(results, out_dir, model='ridge'):
    """figD2: Mean r per subject — cx vs cy bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f'Mean decoder r per subject  [{model}]  —  {N_FOLDS}-fold contiguous CV', fontsize=13, fontweight='bold')
    x_pos   = np.arange(len(SUBJECTS))
    r_cx    = [results[s]['fold_dec_r'][:, 0].mean() for s in SUBJECTS]
    r_cy    = [results[s]['fold_dec_r'][:, 1].mean() for s in SUBJECTS]
    r_cx_sd = [results[s]['fold_dec_r'][:, 0].std()  for s in SUBJECTS]
    r_cy_sd = [results[s]['fold_dec_r'][:, 1].std()  for s in SUBJECTS]
    ax.bar(x_pos - 0.2, r_cx, 0.35, yerr=r_cx_sd, capsize=4,
           color=[COLORS[s] for s in SUBJECTS], alpha=0.9, label='cx')
    ax.bar(x_pos + 0.2, r_cy, 0.35, yerr=r_cy_sd, capsize=4,
           color=[COLORS[s] for s in SUBJECTS], alpha=0.45, label='cy')
    ax.set_xticks(x_pos); ax.set_xticklabels([s.upper() for s in SUBJECTS])
    ax.set_ylabel('Mean r ± SD across folds'); ax.axhline(0, color='k', lw=0.8)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/figD2_decode_r_summary.png', dpi=150)
    plt.close()
    print('Saved figD2_decode_r_summary.png')


def plot_position_traces(results, out_dir):
    """figD3: Predicted vs true position traces (first 60 s of test data)."""
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle('Predicted (orange) vs True (blue) position — first 60 s of test data',
                 fontsize=12, fontweight='bold')
    SHOW_N = 60 * FS_FEAT
    for row, s in enumerate(SUBJECTS):
        r  = results[s]
        n  = min(SHOW_N, len(r['pos_pred']))
        t  = np.arange(n) / FS_FEAT
        pp = r['pos_pred'][:n]; pt = r['pos_true'][:n]
        for col, (comp, lbl) in enumerate([(0, 'cx'), (1, 'cy')]):
            ax = axes[row, col]
            ax.plot(t, pt[:, comp], color='steelblue',  lw=1, alpha=0.8, label='true')
            ax.plot(t, pp[:, comp], color='darkorange', lw=1, alpha=0.8, label='pred')
            rc = r['fold_dec_r'][:, comp].mean()
            ax.set_title(f'{s.upper()} — {lbl}  (mean r={rc:+.3f})', fontsize=9)
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Position (AU)')
            if row == 0 and col == 0: ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/figD3_position_traces.png', dpi=150)
    plt.close()
    print('Saved figD3_position_traces.png')


def plot_trajectory(results, out_dir):
    """figD4: Predicted vs true 2D trajectory."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle('Predicted (orange) vs True (blue) 2-D trajectory\n'
                 '(position predicted directly — no integration)',
                 fontsize=12, fontweight='bold')
    for ax, s in zip(axes.flat, SUBJECTS):
        r    = results[s]
        step = max(1, len(r['pos_pred']) // 5000)
        ax.plot(r['pos_true'][::step, 0], r['pos_true'][::step, 1],
                color='steelblue', lw=0.5, alpha=0.5, label='true')
        ax.plot(r['pos_pred'][::step, 0], r['pos_pred'][::step, 1],
                color='darkorange', lw=0.5, alpha=0.5, label='pred')
        mean_r = r['fold_dec_r'].mean()
        ax.set_title(f'{s.upper()}  mean r={mean_r:+.3f}')
        ax.set_xlabel('cx (AU)'); ax.set_ylabel('cy (AU)')
        ax.set_aspect('equal'); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/figD4_trajectory.png', dpi=150)
    plt.close()
    print('Saved figD4_trajectory.png')


def plot_coef_magnitude(results, out_dir, model='ridge'):
    """figD5: Decoder coefficient magnitudes coloured by band."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f'Decoder coefficient magnitudes [{model}] — last fold\n'
                 'each bar = one lagged feature; colour = frequency band',
                 fontsize=12, fontweight='bold')
    band_cmap = lambda bi: plt.cm.tab10(bi / max(len(BAND_NAMES) - 1, 1))
    n_lags    = len(LAGS)

    for ax, s in zip(axes.flat, SUBJECTS):
        coef = results[s]['fold_coef'][-1]
        mag  = np.abs(coef).mean(1)
        n_features_total = len(mag)
        dummy = np.zeros((1,) + results[s]['test_r'].shape)
        _, meta = select_features(dummy, results[s]['test_r'], top_n=TOP_N)
        band_per_lag = [bi for (_, bi) in meta for _ in range(n_lags)]
        bar_colors = [band_cmap(bi) for bi in band_per_lag[:n_features_total]]
        ax.bar(np.arange(n_features_total), mag, color=bar_colors, alpha=0.85)
        seen = sorted(set(band_per_lag[:n_features_total]))
        handles = [Patch(color=band_cmap(bi), label=BAND_NAMES[bi]) for bi in seen]
        ax.legend(handles=handles, fontsize=7, ncol=2)
        ax.set_xlabel(f'Feature index (top {TOP_N} × {n_lags} lags)')
        ax.set_ylabel('|coef| mean(cx, cy)'); ax.set_title(s.upper())

    plt.tight_layout()
    plt.savefig(f'{out_dir}/figD5_coef_magnitude.png', dpi=150)
    plt.close()
    print('Saved figD5_coef_magnitude.png')


def print_summary_table(results, model='ridge'):
    """Print Ridge decoding summary table to stdout."""
    hdr = (f'{"subj":4s}  {"pos cx r":>9} {"pos cy r":>9} {"pos R²x":>8} {"pos R²y":>8}  '
           f'{"vel vx r":>9} {"vel vy r":>9} {"vel R²x":>8} {"vel R²y":>8}  '
           f'{"jnt cx r":>9} {"jnt cy r":>9} {"jnt R²x":>8} {"jnt R²y":>8}  best')
    print(f'\n{"━"*len(hdr)}')
    print(f'  {hdr}')
    print(f'  {"─"*len(hdr)}')
    for s in SUBJECTS:
        rp   = results[s]['fold_dec_r'].mean(0)
        r2p  = results[s]['fold_dec_r2'].mean(0)
        rv   = results[s]['fold_dec_r_v'].mean(0)
        r2v  = results[s]['fold_dec_r2_v'].mean(0)
        rjp  = results[s]['fold_dec_r_jp'].mean(0)
        r2jp = results[s]['fold_dec_r2_jp'].mean(0)
        means = {'pos': rp.mean(), 'vel': rv.mean(), 'jnt': rjp.mean()}
        best  = max(means, key=means.get)
        print(f'  {s.upper():4s}  '
              f'{rp[0]:+9.4f} {rp[1]:+9.4f} {r2p[0]:+8.4f} {r2p[1]:+8.4f}  '
              f'{rv[0]:+9.4f} {rv[1]:+9.4f} {r2v[0]:+8.4f} {r2v[1]:+8.4f}  '
              f'{rjp[0]:+9.4f} {rjp[1]:+9.4f} {r2jp[0]:+8.4f} {r2jp[1]:+8.4f}  {best}')
    print(f'{"━"*len(hdr)}')
