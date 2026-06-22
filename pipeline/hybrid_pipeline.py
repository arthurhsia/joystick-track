"""
Hybrid decoding pipeline: Ridge → LPF post-processing.
"""

import numpy as np
from config import SUBJECTS, OUT_DIR, N_FOLDS, POST_LPF_HZ
from core.metrics           import pearson_r_1d, r2 as _r2
from core.ridge             import decode_cv
from pipeline.loader        import load_subject
from postprocess.lpf        import temporal_screen, smooth_pred
import viz.hybrid_plots as hyb_plots


def run_hybrid():
    """
    Run LPF post-processing for all subjects.
    Fixed path: 0.25 Hz LPF (no test-set tuning).
    Oracle path: data-driven LPF cutoff selection (diagnostic upper bound).
    Returns all_results dict keyed by subject.
    """
    all_results = {}

    for s in SUBJECTS:
        print(f'\n{"═"*64}')
        print(f'  {s.upper()} — loading & preprocessing ...')

        data      = load_subject(s)
        feats     = data['feats']
        pos       = data['pos']
        trial_ids = data['trial_ids']
        T_total   = len(feats)
        print(f'  T={T_total}')

        print('  Ridge position ...')
        pos_pred, pos_true, pos_tidx, fold_r_p, fold_r2_p, _ = decode_cv(
            feats, pos, trial_ids, n_folds=N_FOLDS, label='pos')

        ridge_r_cx  = fold_r_p.mean(0)[0];  ridge_r_cy  = fold_r_p.mean(0)[1]
        ridge_r2_cx = fold_r2_p.mean(0)[0]; ridge_r2_cy = fold_r2_p.mean(0)[1]

        # ── Fixed path (no test-set tuning) ──────────────────────────────────
        lpf_cx_fixed    = smooth_pred(pos_pred[:, 0], POST_LPF_HZ)
        lpf_cy_fixed    = smooth_pred(pos_pred[:, 1], POST_LPF_HZ)
        lpf_r_cx_fixed  = pearson_r_1d(pos_true[:, 0], lpf_cx_fixed)
        lpf_r_cy_fixed  = pearson_r_1d(pos_true[:, 1], lpf_cy_fixed)
        lpf_r2_cx_fixed = _r2(pos_true[:, 0], lpf_cx_fixed)
        lpf_r2_cy_fixed = _r2(pos_true[:, 1], lpf_cy_fixed)

        print(f'\n  ─── Fixed path  [LPF {POST_LPF_HZ} Hz] ───')
        print(f'  cx  r={lpf_r_cx_fixed:+.4f}  R²={lpf_r2_cx_fixed:+.4f}')
        print(f'  cy  r={lpf_r_cy_fixed:+.4f}  R²={lpf_r2_cy_fixed:+.4f}')

        # ── Oracle path (test-set tuned — diagnostic only) ───────────────────
        print(f'\n  ─── Oracle path [temporal LPF screen] ───')
        rows, opt_cx, opt_cy = temporal_screen(pos_pred, pos_true)
        for label, r_cx, r_cy in rows:
            base_cx, base_cy = rows[0][1], rows[0][2]
            d_cx = r_cx - base_cx; d_cy = r_cy - base_cy
            mark = ' ←' if (d_cx + d_cy) > 0.02 else ''
            print(f'    {label:>8}  cx r={r_cx:+.4f} (Δ{d_cx:+.4f})  '
                  f'cy r={r_cy:+.4f} (Δ{d_cy:+.4f}){mark}')

        lpf_cx_orc    = smooth_pred(pos_pred[:, 0], opt_cx) if opt_cx > 0 else pos_pred[:, 0].copy()
        lpf_cy_orc    = smooth_pred(pos_pred[:, 1], opt_cy) if opt_cy > 0 else pos_pred[:, 1].copy()
        lpf_r_cx_orc  = pearson_r_1d(pos_true[:, 0], lpf_cx_orc)
        lpf_r_cy_orc  = pearson_r_1d(pos_true[:, 1], lpf_cy_orc)
        lpf_r2_cx_orc = _r2(pos_true[:, 0], lpf_cx_orc)
        lpf_r2_cy_orc = _r2(pos_true[:, 1], lpf_cy_orc)

        print(f'\n  {"─"*60}')
        print(f'  {"":4}  {"Ridge":>10}  {"Fixed LPF":>22}  {"Oracle LPF":>22}')
        print(f'  {"─"*4}  {"─"*10}  {"─"*22}  {"─"*22}')
        for axis, r_ridge, r_fixed, r_oracle in [
            ('cx', ridge_r_cx, lpf_r_cx_fixed, lpf_r_cx_orc),
            ('cy', ridge_r_cy, lpf_r_cy_fixed, lpf_r_cy_orc),
        ]:
            print(f'  {axis:4}  r={r_ridge:+.4f}    '
                  f'r={r_fixed:+.4f} (Δ{r_fixed-r_ridge:+.4f})    '
                  f'r={r_oracle:+.4f} (infl.{r_oracle-r_fixed:+.4f})')

        all_results[s] = dict(
            ridge_r      = (ridge_r_cx,      ridge_r_cy),
            ridge_r2     = (ridge_r2_cx,     ridge_r2_cy),
            pos_pred     = pos_pred,
            pos_true     = pos_true,
            hyb_r_fixed  = (lpf_r_cx_fixed,  lpf_r_cy_fixed),
            hyb_r2_fixed = (lpf_r2_cx_fixed, lpf_r2_cy_fixed),
            hyb_pos_fixed= np.column_stack([lpf_cx_fixed, lpf_cy_fixed]),
            screen       = rows,
            opt_cx       = opt_cx,
            opt_cy       = opt_cy,
            hyb_r        = (lpf_r_cx_orc,    lpf_r_cy_orc),
            hyb_r2       = (lpf_r2_cx_orc,   lpf_r2_cy_orc),
            hyb_pos      = np.column_stack([lpf_cx_orc, lpf_cy_orc]),
        )

    hyb_plots.print_hybrid_summary(all_results)
    hyb_plots.plot_temporal_screen(all_results, OUT_DIR)
    hyb_plots.plot_comparison_bars(all_results, OUT_DIR)
    hyb_plots.plot_hybrid_traces(all_results, OUT_DIR)

    print(f'\nAll hybrid figures saved to {OUT_DIR}')
    return all_results
