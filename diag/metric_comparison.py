"""
Diagnostic: does signed_mean channel ranking discard informative channels?

Steps 3–5 of the selection-metric investigation.

  3. Cancellation victims: |signed_mean| < 0.1 but both |r_cx|, |r_cy| > 0.3
  4. Top-N selection overlap (Jaccard) across metrics
  5. Decode impact: held-out r and R² under each metric

Usage:
    python diag/metric_comparison.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from pipeline.loader    import load_subject
from core.channel_eval  import cv_channel_selection
from core.channel_select import _ranking_score, select_features
from core.ridge         import decode_cv
from config             import SUBJECTS, TOP_N, N_FOLDS, BAND_NAMES

METRICS = ['signed_mean', 'mean_abs', 'l2']

CANCEL_ABS_THRESH  = 0.1   # |signed_mean| below this
CANCEL_AXIS_THRESH = 0.3   # both per-axis |r| above this


def _top_n_set(ranking, top_n):
    """Return frozenset of (ch, band) flat indices for top-n pairs."""
    r_flat   = ranking.flatten()
    top_flat = np.argsort(r_flat)[-top_n:]
    return frozenset(top_flat.tolist())


def _jaccard(a, b):
    return len(a & b) / len(a | b)


def run():
    for s in SUBJECTS:
        print(f'\n{"═"*64}')
        print(f'  {s.upper()}')
        print(f'{"═"*64}')

        data      = load_subject(s)
        feats     = data['feats']
        pos       = data['pos']
        trial_ids = data['trial_ids']
        good_idx  = data['good_idx']
        C, B      = feats.shape[1], feats.shape[2]

        # ── Compute per-axis CV r ─────────────────────────────────────────
        print(f'  Computing per-axis CV r ({N_FOLDS}-fold) …')
        _, _, _, per_axis = cv_channel_selection(
            feats, pos, n_folds=N_FOLDS, return_per_axis=True)
        r_cx, r_cy = per_axis   # each (C, B)

        signed_mean = (r_cx + r_cy) / 2

        # ── Step 3: Cancellation victims ──────────────────────────────────
        victims = np.argwhere(
            (np.abs(signed_mean) < CANCEL_ABS_THRESH) &
            (np.abs(r_cx)        > CANCEL_AXIS_THRESH) &
            (np.abs(r_cy)        > CANCEL_AXIS_THRESH)
        )
        print(f'\n  Step 3 — cancellation victims '
              f'(|signed_mean|<{CANCEL_ABS_THRESH}, both |r|>{CANCEL_AXIS_THRESH})')
        if len(victims) == 0:
            print('    none — signed_mean cancellation is not a problem for this subject')
        else:
            print(f'    {"ch":>5}  {"band":>8}  {"r_cx":>7}  {"r_cy":>7}  {"signed_mean":>12}')
            for ch, bi in victims:
                orig_ch = good_idx[ch] + 1
                print(f'    {orig_ch:>5}  {BAND_NAMES[bi]:>8}  '
                      f'{r_cx[ch,bi]:>+7.4f}  {r_cy[ch,bi]:>+7.4f}  '
                      f'{signed_mean[ch,bi]:>+12.4f}')

        if len(victims) == 0:
            # No cancellation — still run steps 4 & 5 to show metric equivalence
            pass

        # ── Step 4: Selection overlap ──────────────────────────────────────
        print(f'\n  Step 4 — top-{TOP_N} selection overlap')
        rankings = {}
        sets     = {}
        for m in METRICS:
            rankings[m] = _ranking_score([r_cx, r_cy], m)
            sets[m]     = _top_n_set(rankings[m], TOP_N)

        ref_set = sets['signed_mean']
        ref_pairs = sorted(ref_set)

        def _pair_label(flat_idx):
            ch = flat_idx // B; bi = flat_idx % B
            return f'ch{good_idx[ch]+1}/{BAND_NAMES[bi]}'

        print(f'    signed_mean: {sorted(_pair_label(i) for i in ref_set)}')
        for m in ['mean_abs', 'l2']:
            j = _jaccard(ref_set, sets[m])
            added   = sets[m] - ref_set
            dropped = ref_set - sets[m]
            print(f'    {m:>11}: Jaccard={j:.2f}  '
                  f'added={sorted(_pair_label(i) for i in added)}  '
                  f'dropped={sorted(_pair_label(i) for i in dropped)}')

        # ── Step 5: Decoding impact ────────────────────────────────────────
        print(f'\n  Step 5 — decoding impact ({N_FOLDS}-fold contiguous CV)')
        print(f'    {"metric":>12}  {"cx_r":>7}  {"cx_R²":>7}  {"cy_r":>7}  {"cy_R²":>7}')
        print(f'    {"─"*12}  {"─"*7}  {"─"*7}  {"─"*7}  {"─"*7}')
        for m in METRICS:
            _, _, _, fold_r, fold_r2, _ = decode_cv(
                feats, pos, trial_ids, n_folds=N_FOLDS,
                label=f'{s}/{m}', metric=m)
            mr  = fold_r.mean(0)
            mr2 = fold_r2.mean(0)
            print(f'    {m:>12}  {mr[0]:>+7.4f}  {mr2[0]:>+7.4f}  '
                  f'{mr[1]:>+7.4f}  {mr2[1]:>+7.4f}')


if __name__ == '__main__':
    run()
