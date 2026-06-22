"""
Velocity decoding result plots: pos/vel/joint bars and velocity traces.
"""

import numpy as np
import matplotlib.pyplot as plt
from config import SUBJECTS, COLORS, N_FOLDS, FS_FEAT


def plot_pos_vel_joint_bars(results, out_dir, model='ridge'):
    """figV1: Position vs Velocity vs Joint — per-axis summary bars."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Position / Velocity / Joint decoding  [{model}, {N_FOLDS}-fold CV]',
                 fontsize=13, fontweight='bold')
    x = np.arange(len(SUBJECTS)); w = 0.22
    for ax, comp, lbl in zip(axes, [0, 1], ['x / cx', 'y / cy']):
        pos_m = [results[s]['fold_dec_r'][:, comp].mean()    for s in SUBJECTS]
        vel_m = [results[s]['fold_dec_r_v'][:, comp].mean()  for s in SUBJECTS]
        jnt_m = [results[s]['fold_dec_r_jp'][:, comp].mean() for s in SUBJECTS]
        pos_e = [results[s]['fold_dec_r'][:, comp].std()    for s in SUBJECTS]
        vel_e = [results[s]['fold_dec_r_v'][:, comp].std()  for s in SUBJECTS]
        jnt_e = [results[s]['fold_dec_r_jp'][:, comp].std() for s in SUBJECTS]
        cols  = [COLORS[s] for s in SUBJECTS]
        ax.bar(x - w, pos_m, w, yerr=pos_e, capsize=3,
               color=cols, alpha=0.9, label='position')
        ax.bar(x,     vel_m, w, yerr=vel_e, capsize=3,
               color=cols, alpha=0.45, hatch='//', label='velocity')
        ax.bar(x + w, jnt_m, w, yerr=jnt_e, capsize=3,
               color=cols, alpha=0.45, hatch='xx', label='joint')
        ax.axhline(0, color='k', lw=0.8, alpha=0.4)
        ax.set_xticks(x); ax.set_xticklabels([s.upper() for s in SUBJECTS])
        ax.set_ylabel('r'); ax.set_title(f'Component: {lbl}')
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/figV1_pos_vs_vel.png', dpi=150)
    plt.close()
    print('Saved figV1_pos_vs_vel.png')


def plot_velocity_traces(results, out_dir):
    """figV2: Predicted vs true velocity traces (first 60 s of valid test data)."""
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle('Predicted (orange) vs True (blue) velocity — first 60 s of valid test data',
                 fontsize=12, fontweight='bold')
    SHOW_N = 60 * FS_FEAT
    for row, s in enumerate(SUBJECTS):
        r  = results[s]
        n  = min(SHOW_N, len(r['vel_pred']))
        t  = np.arange(n) / FS_FEAT
        vp = r['vel_pred'][:n]; vt = r['vel_true'][:n]
        for col, (comp, lbl) in enumerate([(0, 'vx'), (1, 'vy')]):
            ax = axes[row, col]
            ax.plot(t, vt[:, comp], color='steelblue',  lw=1, alpha=0.8, label='true')
            ax.plot(t, vp[:, comp], color='darkorange', lw=1, alpha=0.8, label='pred')
            rc = r['fold_dec_r_v'][:, comp].mean()
            ax.set_title(f'{s.upper()} — {lbl}  (mean r={rc:+.3f})', fontsize=9)
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Velocity (AU/s)')
            if row == 0 and col == 0: ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/figV2_velocity_traces.png', dpi=150)
    plt.close()
    print('Saved figV2_velocity_traces.png')
