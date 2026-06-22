"""
Generate a clean, readable pipeline flowchart — simplified version.
Run: python flowchart.py
Saves: figs/pipeline_flowchart.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from config import OUT_DIR

# ── Palette ───────────────────────────────────────────────────────────────────
COL = dict(
    input   = '#D6EAF8',
    prep    = '#FDEBD0',
    feat    = '#D5F5E3',
    target  = '#FDEDEC',
    cv      = '#EDE7F6',
    select  = '#FFFDE7',
    decode  = '#FFF3E0',
    post    = '#E0F7FA',
    hybrid  = '#FFF8E1',
    result  = '#ECEFF1',
    arrow   = '#555555',
)

FIG_W, FIG_H = 15, 32
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)
ax.axis('off')
fig.patch.set_facecolor('white')


# ── Drawing helpers ───────────────────────────────────────────────────────────
def box(cx, cy, w, h, lines, color, fs=9.5, bold_first=True):
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle='round,pad=0.18',
                           facecolor=color, edgecolor='#444444',
                           linewidth=1.4, zorder=3)
    ax.add_patch(rect)
    n  = len(lines)
    dy = h / (n + 1)
    for i, line in enumerate(lines):
        y  = cy + h/2 - dy*(i + 1)
        fw = 'bold' if (i == 0 and bold_first) else 'normal'
        fc = '#1a1a1a' if (i == 0 and bold_first) else '#333333'
        ax.text(cx, y, line, ha='center', va='center', fontsize=fs,
                fontweight=fw, color=fc, zorder=4,
                family='monospace' if line.startswith('→') else 'sans-serif')


def arrow(x0, y0, x1, y1, label=''):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=COL['arrow'], lw=1.7),
                zorder=2)
    if label:
        mx = (x0+x1)/2 + 0.18; my = (y0+y1)/2
        ax.text(mx, my, label, fontsize=8, color='#666', va='center',
                fontstyle='italic', zorder=5)


# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(FIG_W/2, FIG_H - 0.45,
        'ECoG → Joystick Cursor Decoding — Full Pipeline',
        ha='center', va='top', fontsize=15, fontweight='bold')
ax.text(FIG_W/2, FIG_H - 0.95,
        '4 subjects (FP, GF, RH, RR)  ·  60-ch ECoG  ·  Ridge + Hybrid decoder',
        ha='center', va='top', fontsize=9.5, color='#555')


# ── Layout constants ──────────────────────────────────────────────────────────
CX   = 7.5      # main spine centre
BW   = 11.5     # wide box width
SH   = 0.55     # half-height of a standard 2-line box  (box height = 2*SH)
GAP  = 0.50     # vertical gap between boxes


# ── Y positions ───────────────────────────────────────────────────────────────
y = FIG_H - 1.7    # running cursor, decremented each box

def next_y(box_h=1.1, gap=GAP):
    """Return centre-y for next box, then decrement cursor."""
    global y
    c  = y - box_h/2
    y -= box_h + gap
    return c

Y = {}
Y['input']  = next_y(1.1)
Y['notch']  = next_y(0.9)
Y['badch']  = next_y(0.9)
Y['car']    = next_y(0.9)
Y['seg']    = next_y(0.9, gap=GAP+0.2)

# fork
Y['fork_split'] = y      # y-level of the horizontal connector
y -= 0.35
Y['feat']   = next_y(1.3)
Y['lmp']    = next_y(1.0)
Y['pos']    = Y['feat']
Y['vel']    = Y['lmp']
Y['fork_join'] = y       # y-level of the join connector
y -= 0.35

Y['concat'] = next_y(0.9, gap=GAP+0.1)

# CV loop header (visual only — dashed rect drawn later)
cv_top_y = y + 0.4
Y['norm']   = next_y(0.9)
Y['chsel']  = next_y(0.9)
Y['lag']    = next_y(0.9)
Y['ridge']  = next_y(1.3)
Y['eval']   = next_y(0.9, gap=GAP+0.2)
cv_bot_y = y + GAP

Y['lpf']    = next_y(1.0, gap=GAP+0.1)
Y['kalman'] = next_y(1.1, gap=GAP+0.1)
Y['hybrid'] = next_y(1.1, gap=GAP+0.1)
Y['result'] = next_y(4.2)


# ── INPUT ─────────────────────────────────────────────────────────────────────
box(CX, Y['input'], BW, 1.1, [
    'INPUT DATA',
    'ECoG  x_raw (T, 60ch) @ 1 kHz  ·  Joystick (cx, cy) @ 1 kHz  ·  int16, range 0–32 767',
], COL['input'])

arrow(CX, Y['input'] - 0.55, CX, Y['notch'] + 0.45)

# ── NOTCH ─────────────────────────────────────────────────────────────────────
box(CX, Y['notch'], BW, 0.9, [
    'NOTCH FILTER',
    'Remove 60 / 120 / 180 Hz mains noise  (zero-phase IIR notch, Q = 35)',
], COL['prep'])

arrow(CX, Y['notch'] - 0.45, CX, Y['badch'] + 0.45)

# ── BAD CHANNEL ───────────────────────────────────────────────────────────────
box(CX, Y['badch'], BW, 0.9, [
    'BAD CHANNEL DETECTION',
    'Flag: clipping > 0.1 %  OR  RMS z-score > 3.0  →  good_idx',
], COL['prep'])

arrow(CX, Y['badch'] - 0.45, CX, Y['car'] + 0.45)

# ── CAR ───────────────────────────────────────────────────────────────────────
box(CX, Y['car'], BW, 0.9, [
    'COMMON AVERAGE REFERENCE (CAR)',
    'Subtract mean across good channels  →  x_car (T, C_good)',
], COL['prep'])

arrow(CX, Y['car'] - 0.45, CX, Y['seg'] + 0.45)

# ── TRIAL SEGMENTATION ────────────────────────────────────────────────────────
box(CX, Y['seg'], BW, 0.9, [
    'TRIAL SEGMENTATION',
    'Split at freeze gaps (cursor still ≥ 1.5 s)  ·  keep trials ≥ 30 s  →  4–7 trials / subject',
], COL['prep'])


# ── FORK ──────────────────────────────────────────────────────────────────────
CX_L = 3.7;  CX_R = 11.3   # left = features, right = targets
FW   = 6.8

fork_y = Y['seg'] - 0.45 - 0.15
ax.plot([CX, CX], [Y['seg'] - 0.45, fork_y], color=COL['arrow'], lw=1.7)
ax.plot([CX_L, CX_R], [fork_y, fork_y], color=COL['arrow'], lw=1.7)
ax.annotate('', xy=(CX_L, Y['feat'] + 0.65), xytext=(CX_L, fork_y),
            arrowprops=dict(arrowstyle='->', color=COL['arrow'], lw=1.7))
ax.annotate('', xy=(CX_R, Y['pos'] + 0.65), xytext=(CX_R, fork_y),
            arrowprops=dict(arrowstyle='->', color=COL['arrow'], lw=1.7))
ax.text(CX_L - 2.0, fork_y + 0.12, 'neural signal  per trial',
        fontsize=8.5, color='#555', fontstyle='italic')
ax.text(CX_R + 0.15, fork_y + 0.12, 'joystick  per trial',
        fontsize=8.5, color='#555', fontstyle='italic')

# FEATURE BRANCH
box(CX_L, Y['feat'], FW, 1.3, [
    'BAND POWER ENVELOPES  (bands 0–6)',
    'δ  θ  α  β  lγ  hγ1  hγ2',
    'bandpass  →  |·|  →  1.5 Hz smooth  →  ×100 decimate',
], COL['feat'])

arrow(CX_L, Y['feat'] - 0.65, CX_L, Y['lmp'] + 0.50)

box(CX_L, Y['lmp'], FW, 1.0, [
    'LMP — Local Motor Potential  (band 7)',
    '1.5 Hz low-pass only  ·  signed, NOT rectified  →  feats (T//100, C, 8)',
], COL['feat'])

# TARGET BRANCH
box(CX_R, Y['pos'], FW, 1.3, [
    'CURSOR POSITION',
    'Stride-decimate ×100',
    '→  pos (T//100, 2)',
], COL['target'])

arrow(CX_R, Y['pos'] - 0.65, CX_R, Y['vel'] + 0.50)

box(CX_R, Y['vel'], FW, 1.0, [
    'VELOCITY + RAIL MASKING',
    'Diff, mask saturated edges (+100 ms), smooth, decimate  →  vel + valid_mask',
], COL['target'])

# JOIN
join_y = Y['lmp'] - 0.50 - 0.15
ax.plot([CX_L, CX_L], [Y['lmp'] - 0.50, join_y], color=COL['arrow'], lw=1.7)
ax.plot([CX_R, CX_R], [Y['vel'] - 0.50, join_y], color=COL['arrow'], lw=1.7)
ax.plot([CX_L, CX_R], [join_y, join_y], color=COL['arrow'], lw=1.7)
ax.annotate('', xy=(CX, Y['concat'] + 0.45), xytext=(CX, join_y),
            arrowprops=dict(arrowstyle='->', color=COL['arrow'], lw=1.7))

# ── CONCAT ────────────────────────────────────────────────────────────────────
box(CX, Y['concat'], BW, 0.9, [
    'CONCATENATE TRIALS',
    'feats  ·  pos  ·  vel  ·  valid_mask  ·  trial_ids  (lag boundary guard)',
], COL['prep'])

arrow(CX, Y['concat'] - 0.45, CX, Y['norm'] + 0.45)


# ── CV LOOP BORDER ────────────────────────────────────────────────────────────
cv_left = CX - BW/2 - 0.4; cv_right = CX + BW/2 + 0.4
rect_cv = plt.Rectangle(
    (cv_left, cv_bot_y), cv_right - cv_left, cv_top_y - cv_bot_y,
    linewidth=1.8, edgecolor='#7B1FA2', facecolor='#F3E5F5',
    linestyle='--', zorder=0, alpha=0.45)
ax.add_patch(rect_cv)
ax.text(cv_left + 0.22, cv_top_y - 0.10,
        'CONTIGUOUS 5-FOLD CROSS-VALIDATION  (all fitting on TRAIN fold only)',
        fontsize=9, fontweight='bold', color='#7B1FA2', va='top')

# ── NORMALISE ─────────────────────────────────────────────────────────────────
box(CX, Y['norm'], BW, 0.9, [
    'FEATURE NORMALISATION  (fit on TRAIN only)',
    'z-score per (channel, band) using TRAIN μ and σ',
], COL['cv'])

arrow(CX, Y['norm'] - 0.45, CX, Y['chsel'] + 0.45)

# ── CHANNEL SELECTION ─────────────────────────────────────────────────────────
box(CX, Y['chsel'], BW, 0.9, [
    'CHANNEL SELECTION  (TRAIN only)',
    'Top 10 by |mean(r_cx, r_cy)|  —  signed mean, so axis-specific channels cancel',
], COL['select'])

arrow(CX, Y['chsel'] - 0.45, CX, Y['lag'] + 0.45)

# ── LAG MATRIX ────────────────────────────────────────────────────────────────
box(CX, Y['lag'], BW, 0.9, [
    'LAG MATRIX  (per trial — no boundary leakage)',
    'Lags 0–400 ms  →  X_lag (T, 50)  [10 features × 5 time offsets]',
], COL['select'])

arrow(CX, Y['lag'] - 0.45, CX, Y['ridge'] + 0.65)

# ── RIDGE ─────────────────────────────────────────────────────────────────────
box(CX, Y['ridge'], BW, 1.3, [
    'RIDGE REGRESSION  (RidgeCV,  α ∈ {1e3 … 1e12})',
    'Separate decoders for cx and cy  ·  Velocity decoder (valid samples only)',
    '→  pos_pred (N_test, 2)  ·  vel_pred (N_test, 2)',
], COL['decode'])

arrow(CX, Y['ridge'] - 0.65, CX, Y['eval'] + 0.45)

# ── EVALUATE ──────────────────────────────────────────────────────────────────
box(CX, Y['eval'], BW, 0.9, [
    'EVALUATE PER FOLD PER AXIS',
    'Pearson r ∈ [−1, 1]  (primary)  ·  R² = 1 − SS_res / SS_tot  (secondary)',
], COL['decode'])

arrow(CX, Y['eval'] - 0.45, CX, Y['lpf'] + 0.50)


# ── LPF SCREEN ────────────────────────────────────────────────────────────────
box(CX, Y['lpf'], BW, 1.0, [
    'STEP 3 — TEMPORAL LPF SCREEN',
    'Test zero-phase LPF at 0.25 / 0.5 / 1.0 / 2.0 Hz  per axis',
    'All 4 subjects peak at 0.25 Hz  (task ~0.17 Hz / 6 s;  0.25 Hz is empirical optimum)',
], COL['post'])

arrow(CX, Y['lpf'] - 0.50, CX, Y['kalman'] + 0.55)

# ── KALMAN ────────────────────────────────────────────────────────────────────
box(CX, Y['kalman'], BW, 1.1, [
    'STEP 4 — PER-AXIS KALMAN FILTER',
    '2-state [position, velocity]  ·  Two independent Kalmans (cx/cy)',
    'Grid-search σ_a × 20 pts  ·  Velocity gating: use vel only if it helps',
], COL['post'])

arrow(CX, Y['kalman'] - 0.55, CX, Y['hybrid'] + 0.55)

# ── HYBRID ────────────────────────────────────────────────────────────────────
box(CX, Y['hybrid'], BW, 1.1, [
    'HYBRID OUTPUT',
    'cx  ←  zero-phase LPF  (no phase lag — better where ridge cx is strong)',
    'cy  ←  Kalman  (fuses velocity when not railed — better where ridge cy is weak)',
], COL['hybrid'])

arrow(CX, Y['hybrid'] - 0.55, CX, Y['result'] + 2.0)

# ── RESULTS — title bar ───────────────────────────────────────────────────────
res_top = Y['result'] + 2.0
res_bot = Y['result'] - 2.0
res_l   = CX - BW/2; res_r = CX + BW/2

# background rect
rect_res = plt.Rectangle((res_l, res_bot), BW, res_top - res_bot,
                           facecolor=COL['result'], edgecolor='#444',
                           linewidth=1.4, zorder=3, clip_on=False)
ax.add_patch(rect_res)

# title
ax.text(CX, res_top - 0.25,
        'RESULTS  —  Pearson r  (mean across 5 folds)',
        ha='center', va='center', fontsize=10.5, fontweight='bold', zorder=4)

# ── proper table ──────────────────────────────────────────────────────────────
col_labels = ['Subject', 'Ridge cx', 'Ridge cy', 'Hybrid cx', 'Hybrid cy', 'Δ cx', 'Δ cy']
table_data  = [
    ['FP', '+0.45', '+0.26', '+0.62', '+0.41', '+0.17', '+0.15'],
    ['GF', '+0.45', '+0.46', '+0.54', '+0.66', '+0.09', '+0.21'],
    ['RH', '+0.69', '+0.50', '+0.74', '+0.71', '+0.05', '+0.21'],
    ['RR', '+0.61', '+0.52', '+0.71', '+0.56', '+0.10', '+0.04'],
]

# Convert data-coord bbox → axes fraction
tbl_left   = (res_l + 0.3) / FIG_W
tbl_right  = (res_r - 0.3) / FIG_W
tbl_bottom = (res_bot + 0.2) / FIG_H
tbl_top    = (res_top - 0.55) / FIG_H
tbl_bbox   = [tbl_left, tbl_bottom, tbl_right - tbl_left, tbl_top - tbl_bottom]

tbl = ax.table(cellText=table_data, colLabels=col_labels,
               bbox=tbl_bbox, cellLoc='center', zorder=5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

# Style header
for j in range(len(col_labels)):
    cell = tbl[0, j]
    cell.set_facecolor('#AED6F1')
    cell.set_text_props(fontweight='bold', color='#1A1A1A')
    cell.set_edgecolor('#555')

# Style data rows + colour Δ columns
delta_pos = '#1a6e1a'; delta_neg = '#b00000'
row_colors = ['#F0F4F8', 'white', '#F0F4F8', 'white']
for i, row in enumerate(table_data):
    for j in range(len(col_labels)):
        cell = tbl[i+1, j]
        cell.set_facecolor(row_colors[i])
        cell.set_edgecolor('#888')
        if j >= 5:   # Δ columns
            val = row[j]
            if val.startswith('+') and val != '+0.00':
                cell.set_text_props(color=delta_pos, fontweight='bold')
            elif val.startswith('−'):
                cell.set_text_props(color=delta_neg, fontweight='bold')


# ── LEGEND — horizontal strip at very bottom ──────────────────────────────────
legend_entries = [
    ('Input data',           COL['input']),
    ('Preprocessing',        COL['prep']),
    ('Feature extraction',   COL['feat']),
    ('Target extraction',    COL['target']),
    ('CV / normalisation',   COL['cv']),
    ('Channel select / lag', COL['select']),
    ('Regression / eval',    COL['decode']),
    ('Post-processing',      COL['post']),
    ('Hybrid output',        COL['hybrid']),
]
handles = [mpatches.Patch(facecolor=c, edgecolor='#555', label=l)
           for l, c in legend_entries]
ax.legend(handles=handles, loc='lower center', fontsize=8, ncol=5,
          framealpha=0.9, title='Pipeline stage', title_fontsize=8.5,
          bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(pad=0.5)
out = f'{OUT_DIR}/pipeline_flowchart.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}')
