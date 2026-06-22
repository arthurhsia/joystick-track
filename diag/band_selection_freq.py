"""
Count how often each frequency band appears in the top-10 feature selection
across all subjects × folds × axes (cx, cy).

Saves: figs/figX_band_selection_freq.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from config import SUBJECTS, BAND_LABELS, BAND_NAMES, TOP_N, N_FOLDS, OUT_DIR
from pipeline.loader     import load_subject
from core.splits         import contiguous_kfold_splits
from core.channel_eval   import pearson_r

counts = np.zeros(len(BAND_NAMES), dtype=int)   # tally per band

for s in SUBJECTS:
    print(f'  {s.upper()} ...')
    d     = load_subject(s)
    feats = d['feats']    # (T, C, 8)
    pos   = d['pos']      # (T, 2)
    T     = feats.shape[0]

    for tr_idx, te_idx in contiguous_kfold_splits(T, k=N_FOLDS):
        mu     = feats[tr_idx].mean(axis=0, keepdims=True)
        sd     = feats[tr_idx].std(axis=0,  keepdims=True) + 1e-9
        z_full = (feats - mu) / sd

        tr_r = pearson_r(z_full[tr_idx], pos[tr_idx])   # (C, 8) mean |r| across cx+cy

        # replicate select_features: top TOP_N (ch, band) pairs by |r|
        flat   = np.abs(tr_r).ravel()
        top_idx = np.argsort(flat)[::-1][:TOP_N]
        band_idx = top_idx % tr_r.shape[1]   # band dimension
        for b in band_idx:
            counts[b] += 1

print(f'\nBand selection counts (total={counts.sum()}):')
for i, (name, c) in enumerate(zip(BAND_NAMES, counts)):
    print(f'  {name:12s}  {c:4d}  ({100*c/counts.sum():.1f}%)')

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

colors = [
    '#AED6F1', '#A9DFBF', '#FAD7A0', '#F1948A',
    '#BB8FCE', '#85C1E9', '#45B39D', '#F0B27A'
]
bars = ax.bar(range(len(BAND_NAMES)), counts, color=colors[:len(BAND_NAMES)],
              edgecolor='#333', linewidth=0.8)

total = counts.sum()
for bar, c in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{c}\n({100*c/total:.0f}%)',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(range(len(BAND_NAMES)))
ax.set_xticklabels(BAND_LABELS, fontsize=10)
ax.set_ylabel('Times selected (all subjects × folds)', fontsize=11)
ax.set_title(
    f'Band selection frequency — top {TOP_N} features per fold\n'
    f'{len(SUBJECTS)} subjects × {N_FOLDS} folds  (total = {total} selections)',
    fontsize=12, fontweight='bold'
)
ax.axhline(total / len(BAND_NAMES), color='k', ls='--', lw=1,
           label=f'Chance ({total//len(BAND_NAMES)} / band)')
ax.legend(fontsize=9)
ax.set_ylim(0, counts.max() * 1.25)
plt.tight_layout()

out = f'{OUT_DIR}/figX_band_selection_freq.png'
plt.savefig(out, dpi=150)
plt.close()
print(f'Saved {out}')
