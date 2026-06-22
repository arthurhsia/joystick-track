"""
Ridge decoding pipeline: position, velocity, and joint pos+vel.
"""

from config import SUBJECTS, OUT_DIR, N_FOLDS
from core.channel_eval  import cv_channel_selection, print_r_summary
from core.ridge         import decode_cv, decode_cv_joint
from pipeline.loader    import load_subject
import viz.channel_maps  as ch_maps
import viz.decode_plots  as dec_plots
import viz.velocity_plots as vel_plots

MODEL = 'ridge'


def run_ridge():
    """
    Run the full ridge decoding pipeline for all subjects.
    Returns results dict keyed by subject.
    """
    results = {}

    for s in SUBJECTS:
        print(f'\n{"═"*56}\n  {s.upper()}\n{"═"*56}')
        data = load_subject(s)
        feats     = data['feats']
        pos       = data['pos']
        vel       = data['vel']
        vel_mask  = data['vel_mask']
        trial_ids = data['trial_ids']
        good_idx  = data['good_idx']

        print(f'  channels : {data["n_ch"]} total  '
              f'{data["bad_mask"].sum()} removed  {len(good_idx)} kept')
        print(f'  trials   : {data["n_trials"]}')
        print(f'  features : {feats.shape}   pos : {pos.shape}')
        print(f'  vel valid: {vel_mask.sum()}/{len(vel_mask)} '
              f'({100*vel_mask.mean():.1f}%)')

        print(f'  channel selection pos ({N_FOLDS}-fold contiguous CV) ...')
        test_r, train_r, fold_r = cv_channel_selection(feats, pos, n_folds=N_FOLDS)
        print_r_summary(test_r, train_r, good_idx)

        print(f'  channel selection vel ({N_FOLDS}-fold contiguous CV, masked) ...')
        test_r_v, train_r_v, fold_r_v = cv_channel_selection(
            feats, vel, n_folds=N_FOLDS, valid_mask=vel_mask)
        print_r_summary(test_r_v, train_r_v, good_idx)

        print(f'\n  decoding pos [{MODEL}] — {N_FOLDS}-fold ...')
        pos_pred, pos_true, pos_tidx, fold_dec_r, fold_dec_r2, fold_coef = decode_cv(
            feats, pos, trial_ids, model=MODEL, n_folds=N_FOLDS, label='pos')
        mean_r_pos  = fold_dec_r.mean(0)
        mean_r2_pos = fold_dec_r2.mean(0)
        print(f'  pos : cx r={mean_r_pos[0]:+.4f} R²={mean_r2_pos[0]:+.4f}  '
              f'cy r={mean_r_pos[1]:+.4f} R²={mean_r2_pos[1]:+.4f}')

        print(f'\n  decoding vel [{MODEL}] — {N_FOLDS}-fold (masked) ...')
        vel_pred, vel_true, vel_tidx, fold_dec_r_v, fold_dec_r2_v, fold_coef_v = decode_cv(
            feats, vel, trial_ids, model=MODEL, n_folds=N_FOLDS,
            valid_mask=vel_mask, label='vel')
        mean_r_vel  = fold_dec_r_v.mean(0)
        mean_r2_vel = fold_dec_r2_v.mean(0)
        print(f'  vel : vx r={mean_r_vel[0]:+.4f} R²={mean_r2_vel[0]:+.4f}  '
              f'vy r={mean_r_vel[1]:+.4f} R²={mean_r2_vel[1]:+.4f}')

        print(f'\n  decoding joint [ridge] — {N_FOLDS}-fold (masked) ...')
        (jpos_pred, jvel_pred, jpos_true, jvel_true, jtidx,
         fold_dec_r_jp, fold_dec_r_jv,
         fold_dec_r2_jp, fold_dec_r2_jv, fold_coef_j) = decode_cv_joint(
            feats, pos, vel, vel_mask, trial_ids, n_folds=N_FOLDS)
        mean_r_jp  = fold_dec_r_jp.mean(0)
        mean_r2_jp = fold_dec_r2_jp.mean(0)
        mean_r_jv  = fold_dec_r_jv.mean(0)
        mean_r2_jv = fold_dec_r2_jv.mean(0)
        print(f'  joint pos: cx r={mean_r_jp[0]:+.4f} R²={mean_r2_jp[0]:+.4f}  '
              f'cy r={mean_r_jp[1]:+.4f} R²={mean_r2_jp[1]:+.4f}')
        print(f'  joint vel: vx r={mean_r_jv[0]:+.4f} R²={mean_r2_jv[0]:+.4f}  '
              f'vy r={mean_r_jv[1]:+.4f} R²={mean_r2_jv[1]:+.4f}')

        results[s] = dict(
            test_r=test_r, train_r=train_r, fold_r=fold_r,
            test_r_v=test_r_v, train_r_v=train_r_v, fold_r_v=fold_r_v,
            good_idx=good_idx, bad_mask=data['bad_mask'],
            elec=data['elec'], n_ch=data['n_ch'],
            pos_pred=pos_pred, pos_true=pos_true, pos_tidx=pos_tidx,
            vel_pred=vel_pred, vel_true=vel_true, vel_tidx=vel_tidx,
            jpos_pred=jpos_pred, jvel_pred=jvel_pred,
            jpos_true=jpos_true, jvel_true=jvel_true, jtidx=jtidx,
            fold_dec_r=fold_dec_r,   fold_dec_r2=fold_dec_r2,   fold_coef=fold_coef,
            fold_dec_r_v=fold_dec_r_v, fold_dec_r2_v=fold_dec_r2_v, fold_coef_v=fold_coef_v,
            fold_dec_r_jp=fold_dec_r_jp, fold_dec_r2_jp=fold_dec_r2_jp,
            fold_dec_r_jv=fold_dec_r_jv, fold_dec_r2_jv=fold_dec_r2_jv,
            fold_coef_j=fold_coef_j,
            n_trials=data['n_trials'], vel_mask=vel_mask,
        )

    ch_maps.plot_r_heatmap(results, OUT_DIR)
    ch_maps.plot_per_fold_r(results, OUT_DIR)
    ch_maps.plot_channel_topo(results, OUT_DIR)
    dec_plots.plot_fold_r(results, OUT_DIR, model=MODEL)
    dec_plots.plot_r_summary_bar(results, OUT_DIR, model=MODEL)
    dec_plots.plot_position_traces(results, OUT_DIR)
    dec_plots.plot_trajectory(results, OUT_DIR)
    dec_plots.plot_coef_magnitude(results, OUT_DIR, model=MODEL)
    dec_plots.print_summary_table(results, model=MODEL)
    vel_plots.plot_pos_vel_joint_bars(results, OUT_DIR, model=MODEL)
    vel_plots.plot_velocity_traces(results, OUT_DIR)

    print(f'\nAll figures saved to {OUT_DIR}')
    return results
