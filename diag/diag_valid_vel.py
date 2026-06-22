"""
Diagnostic: velocity on valid (non-rail) samples for all subjects.

Plots per subject:
  Row 0  vx and vy with rail-excluded regions shaded grey
  Row 1  speed on valid samples only
  Row 2  position (cx, cy) with rail regions shaded
  Row 3  velocity histogram  (valid samples)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from config import FS, FS_FEAT, DATA_DIR, OUT_DIR, RAIL_TOL
from core.targets    import cursor_velocity_masked
from core.segmentation import find_trial_boundaries

SHOW_S = 60   # seconds to display

for s in ['fp', 'gf', 'rh', 'rr']:
    d  = sio.loadmat(f'{DATA_DIR}/{s}_joystick.mat')
    cx = d['CursorPosX'].flatten().astype(float)
    cy = d['CursorPosY'].flatten().astype(float)

    trials = find_trial_boundaries(cx, cy)
    print(f'{s.upper()}: {len(trials)} trials')

    longest = max(trials, key=lambda t: t[1] - t[0])
    ts, te  = longest
    cx_t    = cx[ts:te]; cy_t = cy[ts:te]

    vel, mask = cursor_velocity_masked(cx_t, cy_t)
    t_d  = np.arange(len(vel)) / FS_FEAT

    N    = min(int(SHOW_S * FS_FEAT), len(vel))
    vel  = vel[:N]; mask = mask[:N]; t_d = t_d[:N]
    N_fs = min(SHOW_S * FS, te - ts)
    cx_s = cx_t[:N_fs]; cy_s = cy_t[:N_fs]
    t_fs = np.arange(N_fs) / FS

    n_valid = mask.sum(); pct = 100 * n_valid / len(mask)
    print(f'  longest trial: {(te-ts)/FS:.0f} s  |  valid: {n_valid}/{len(mask)} ({pct:.1f}%)')

    fig, axes = plt.subplots(4, 1, figsize=(16, 13))
    fig.suptitle(f'{s.upper()} — velocity on valid (non-rail) samples  '
                 f'[longest trial, first {SHOW_S} s]\n'
                 f'grey = rail-excluded  |  valid: {pct:.1f}% of samples',
                 fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(t_d, vel[:, 0], lw=0.7, color='steelblue', alpha=0.9, label='vx')
    ax.plot(t_d, vel[:, 1], lw=0.7, color='darkorange', alpha=0.9, label='vy')
    changes = np.where(np.diff((~mask).astype(int), prepend=0, append=0))[0]
    for i in range(0, len(changes), 2):
        x0 = t_d[changes[i]] if changes[i] < len(t_d) else t_d[-1]
        x1 = t_d[changes[i+1]-1] if changes[i+1]-1 < len(t_d) else t_d[-1]
        ax.axvspan(x0, x1, color='gray', alpha=0.3, lw=0)
    ax.set_ylabel('Velocity (AU/s)'); ax.legend(fontsize=8, loc='upper right')
    ax.set_title('vx / vy  (grey = at-rail, excluded from model)')

    ax = axes[1]
    speed   = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)
    t_valid = t_d[mask]; s_valid = speed[mask]
    ax.scatter(t_valid, s_valid, s=1, color='purple', alpha=0.5)
    ax.set_ylabel('Speed (AU/s)'); ax.set_title('Speed — valid samples only')

    ax = axes[2]
    ax.plot(t_fs, cx_s, lw=0.5, color='steelblue', alpha=0.8, label='cx')
    ax.plot(t_fs, cy_s, lw=0.5, color='darkorange', alpha=0.8, label='cy')
    at_rail_fs = (
        (cx_s <= RAIL_TOL) | (cx_s >= 32767 - RAIL_TOL) |
        (cy_s <= RAIL_TOL) | (cy_s >= 32767 - RAIL_TOL)
    )
    chg = np.where(np.diff(at_rail_fs.astype(int), prepend=0, append=0))[0]
    for i in range(0, len(chg), 2):
        x0 = t_fs[chg[i]] if chg[i] < len(t_fs) else t_fs[-1]
        x1 = t_fs[min(chg[i+1], len(t_fs)-1)]
        ax.axvspan(x0, x1, color='gray', alpha=0.3, lw=0)
    ax.axhline(0,     ls='--', lw=0.8, color='k', alpha=0.3)
    ax.axhline(32767, ls='--', lw=0.8, color='k', alpha=0.3)
    ax.set_ylabel('Position (AU)'); ax.legend(fontsize=8, loc='upper right')
    ax.set_title('Raw cursor position  (grey = at-rail)')

    ax = axes[3]
    vx_valid = vel[mask, 0]; vy_valid = vel[mask, 1]
    v_max = max(np.abs(vx_valid).max(), np.abs(vy_valid).max())
    bins  = np.linspace(-v_max, v_max, 80)
    ax.hist(vx_valid, bins=bins, alpha=0.6, color='steelblue',  density=True, label='vx valid')
    ax.hist(vy_valid, bins=bins, alpha=0.6, color='darkorange', density=True, label='vy valid')
    ax.set_xlabel('Velocity (AU/s)'); ax.set_ylabel('Density')
    ax.set_title('Valid-only velocity distribution'); ax.legend(fontsize=8)

    for ax in axes: ax.set_xlim(0, SHOW_S)
    plt.tight_layout()
    out = f'{OUT_DIR}/diag_valid_vel_{s}.png'
    plt.savefig(out, dpi=150); plt.close()
    print(f'  Saved {out}')

print('\nDone.')
