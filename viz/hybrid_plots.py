"""
Hybrid decoder result plots: temporal screen, σ_a grid search, comparison bars, traces.
"""

import numpy as np
import matplotlib.pyplot as plt
from config import SUBJECTS, COLORS

DT = 0.1   # 1 / FS_FEAT


def print_hybrid_summary(all_results):
    """Print Ridge | Fixed LPF | Oracle LPF summary with r and R²."""
    W = 120
    print(f'\n{"━"*W}')
    print(f'  {"SUBJ":4}  {"AXIS":4}  '
          f'{"Ridge r":>8}  {"Ridge R²":>8}  '
          f'{"Fixed r":>8}  {"Fixed R²":>8}  {"Δr":>7}  '
          f'{"Oracle r":>8}  {"Oracle R²":>8}  {"infl.":>7}')
    print(f'  {"─"*4}  {"─"*4}  {"─"*8}  {"─"*8}  {"─"*8}  {"─"*8}  {"─"*7}  {"─"*8}  {"─"*8}  {"─"*7}')
    for s in SUBJECTS:
        res = all_results[s]
        for dim_i, axis in enumerate(['cx', 'cy']):
            r_ridge  = res['ridge_r'][dim_i];    r2_ridge  = res['ridge_r2'][dim_i]
            r_fixed  = res['hyb_r_fixed'][dim_i]; r2_fixed  = res['hyb_r2_fixed'][dim_i]
            r_oracle = res['hyb_r'][dim_i];       r2_oracle = res['hyb_r2'][dim_i]
            print(f'  {s.upper():4}  {axis:4}  '
                  f'{r_ridge:+8.4f}  {r2_ridge:+8.4f}  '
                  f'{r_fixed:+8.4f}  {r2_fixed:+8.4f}  {r_fixed-r_ridge:+7.4f}  '
                  f'{r_oracle:+8.4f}  {r2_oracle:+8.4f}  {r_oracle-r_fixed:+7.4f}')
    print(f'{"━"*W}')


def plot_temporal_screen(all_results, out_dir):
    """figK1: Pearson r vs LPF cutoff per subject and axis."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Step 3 — Temporal screen: Pearson r vs LPF cutoff\n'
                 '(Evaluated independently for cx and cy)',
                 fontsize=12, fontweight='bold')
    for ax, s in zip(axes.flat, SUBJECTS):
        rows   = all_results[s]['screen']
        labels = [r[0] for r in rows]
        cx_r   = [r[1] for r in rows]
        cy_r   = [r[2] for r in rows]
        x      = np.arange(len(labels))
        ax.plot(x, cx_r, 'o-',  color=COLORS[s], lw=2,          label='cx')
        ax.plot(x, cy_r, 's--', color=COLORS[s], lw=2, alpha=0.7, label='cy')
        ax.axhline(cx_r[0], color=COLORS[s], lw=0.8, ls=':', alpha=0.5)
        ax.set_title(f'{s.upper()}  '
                     f'opt cx={all_results[s]["opt_cx"] or "—"} Hz  '
                     f'opt cy={all_results[s]["opt_cy"] or "—"} Hz')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Pearson r'); ax.legend(fontsize=9)
        ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{out_dir}/figK1_temporal_screen.png', dpi=150)
    plt.close(fig)
    print('Saved figK1_temporal_screen.png')



def plot_comparison_bars(all_results, out_dir):
    """figK3: Ridge vs Fixed vs Oracle hybrid Pearson r bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Ridge  vs  Fixed (no leakage)  vs  Oracle (test-set tuned) — Hybrid Pearson r',
                 fontsize=11, fontweight='bold')
    for ax, dim_i, dim_lbl in zip(axes, [0, 1], ['cx', 'cy']):
        n   = len(SUBJECTS); x = np.arange(n); w = 0.22
        clr = [COLORS[s] for s in SUBJECTS]
        r_r = [all_results[s]['ridge_r'][dim_i]    for s in SUBJECTS]
        r_f = [all_results[s]['hyb_r_fixed'][dim_i] for s in SUBJECTS]
        r_o = [all_results[s]['hyb_r'][dim_i]       for s in SUBJECTS]
        offsets = [-w, 0.0, w]
        groups  = [
            (r_r, 'Ridge',  0.95, None, None, None),
            (r_f, 'Fixed',  0.70, 'k',  1.0,  '//'),
            (r_o, 'Oracle', 0.50, 'k',  1.5,  '..'),
        ]
        all_bars = []
        for (vals, lbl, alpha, ec, lw, hatch), off in zip(groups, offsets):
            kw = dict(label=lbl, alpha=alpha, color=clr)
            if ec:    kw['edgecolor'] = ec
            if lw:    kw['linewidth'] = lw
            if hatch: kw['hatch']     = hatch
            bars = ax.bar(x + off, vals, w, **kw)
            all_bars.extend(bars)
        ax.set_xticks(x); ax.set_xticklabels([s.upper() for s in SUBJECTS], fontsize=11)
        ax.set_ylabel('Pearson r'); ax.set_title(f'{dim_lbl} cursor', fontsize=12)
        ax.axhline(0, color='k', lw=0.5); ax.legend(fontsize=9)
        for bar in all_bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.005 * (1 if h >= 0 else -1),
                    f'{h:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)
    plt.tight_layout()
    fig.savefig(f'{out_dir}/figK3_comparison.png', dpi=150)
    plt.close(fig)
    print('Saved figK3_comparison.png')


def plot_hybrid_traces(all_results, out_dir):
    """figK4: Trace overlay — True, Ridge, Fixed LPF, Oracle LPF (first 50 s)."""
    TRACE_LEN = min(500, min(len(all_results[s]['pos_true']) for s in SUBJECTS))
    t_ax      = np.arange(TRACE_LEN) * DT

    fig, axes = plt.subplots(len(SUBJECTS), 2, figsize=(16, 3 * len(SUBJECTS)), sharex=True)
    fig.suptitle('Traces — True (black)  ·  Ridge --  ·  Fixed LPF -.  ·  Oracle LPF ···\n'
                 '(first 50 s)',
                 fontsize=11)
    for row_i, s in enumerate(SUBJECTS):
        r   = all_results[s]
        pt  = r['pos_true'][:TRACE_LEN];    pp = r['pos_pred'][:TRACE_LEN]
        lpf = r['hyb_pos_fixed'][:TRACE_LEN]; orc = r['hyb_pos'][:TRACE_LEN]
        for col_i, dim in enumerate([0, 1]):
            ax = axes[row_i, col_i]; clr = COLORS[s]
            ax.plot(t_ax, pt[:, dim],  color='k', lw=1.0,          label='true')
            ax.plot(t_ax, pp[:, dim],  color=clr, lw=0.8, ls='--', alpha=0.5,  label='Ridge')
            ax.plot(t_ax, lpf[:, dim], color=clr, lw=1.2, ls='-.', alpha=0.85, label='Fixed LPF')
            ax.plot(t_ax, orc[:, dim], color=clr, lw=1.5, ls=':',              label='Oracle LPF')
            ax.set_ylabel(f'{s.upper()} {"cx" if dim == 0 else "cy"}')
            if row_i == 0: ax.legend(fontsize=8, loc='upper right')
    axes[-1, 0].set_xlabel('Time (s)'); axes[-1, 1].set_xlabel('Time (s)')
    plt.tight_layout()
    fig.savefig(f'{out_dir}/figK4_traces.png', dpi=150)
    plt.close(fig)
    print('Saved figK4_traces.png')
