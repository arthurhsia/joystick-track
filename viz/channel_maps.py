"""
Channel r heatmaps: per-subject band×channel r maps and per-fold r curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from config import SUBJECTS, COLORS, BAND_NAMES, BAND_LABELS


def plot_r_heatmap(results, out_dir):
    """figR1: Cross-validated Pearson r map (bands × channels, all subjects)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cross-validated Pearson r map  (test fold, contiguous 5-fold CV mean)\n'
                 'Cursor position (cx+cy mean) ~ channel envelope  |  ★ = best |r| per band',
                 fontsize=12, fontweight='bold')
    for ax, s in zip(axes.flat, SUBJECTS):
        r    = results[s]['test_r']
        vmax = min(np.abs(r).max(), 0.5)
        im   = ax.imshow(r.T, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label='r')
        ax.set_xlabel('Channel index (good channels only)')
        ax.set_yticks(range(len(BAND_LABELS)))
        ax.set_yticklabels(BAND_LABELS, fontsize=8)
        ax.set_title(f'{s.upper()}  ({r.shape[0]}/{results[s]["n_ch"]} ch kept)  '
                     f'peak |r|={np.abs(r).max():.3f}')
        for bi in range(len(BAND_NAMES)):
            best = np.abs(r[:, bi]).argmax()
            ax.scatter(best, bi, marker='*', color='black', s=100, zorder=5)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/figR1_r_heatmap.png', dpi=150)
    plt.close()
    print('Saved figR1_r_heatmap.png')


def plot_per_fold_r(results, out_dir):
    """figR2: Per-fold best-channel |r| per band (diagnose fold-to-fold variability)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Per-fold channel r (contiguous k-fold test)\n'
                 'each line = one band, value = best |r| channel that fold',
                 fontsize=11, fontweight='bold')
    for ax, s in zip(axes.flat, SUBJECTS):
        fold_r = results[s]['fold_r']
        n_t    = fold_r.shape[0]
        for bi, label in enumerate(BAND_LABELS):
            per_fold = np.abs(fold_r[:, :, bi]).max(axis=1)
            ax.plot(range(1, n_t + 1), per_fold, marker='o', lw=1.5, label=label)
        ax.set_xlabel('Fold (held-out)')
        ax.set_ylabel('|r| (best channel)')
        ax.set_title(s.upper())
        ax.set_xticks(range(1, n_t + 1))
        ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/figR2_per_fold_r.png', dpi=150)
    plt.close()
    print('Saved figR2_per_fold_r.png')


def plot_channel_topo(results, out_dir):
    """figR3: Channel importance heatmap on MNI glass brain (lateral view)."""
    from nilearn import plotting as nlplot, datasets
    import nibabel as nib
    from scipy.ndimage import gaussian_filter
    import tempfile, os
    import matplotlib.image as mpimg

    template = datasets.load_mni152_template(resolution=2)
    affine   = template.affine
    shape    = template.shape[:3]
    inv_aff  = np.linalg.inv(affine)

    tmp_paths = []
    for s in SUBJECTS:
        r        = results[s]
        elec     = r['elec']
        good_idx = r['good_idx']
        test_r   = r['test_r']

        importance = np.abs(test_r).max(axis=1)   # (C,) max |r| across bands
        coords     = elec[good_idx].copy()        # (C, 3) MNI mm
        coords[:, 0] = np.abs(coords[:, 0])       # mirror to right hemisphere

        # Deposit each electrode's importance into a volume, then Gaussian-smooth
        vol = np.zeros(shape)
        for coord, imp in zip(coords, importance):
            vox = np.round(nib.affines.apply_affine(inv_aff, coord)).astype(int)
            if np.all(vox >= 0) and np.all(vox < shape):
                vol[tuple(vox)] = max(vol[tuple(vox)], imp)
        vol = gaussian_filter(vol, sigma=4)   # ~8 mm FWHM at 2 mm resolution
        stat_img = nib.Nifti1Image(vol, affine)

        fd, tmp = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        display = nlplot.plot_glass_brain(
            stat_img,
            display_mode='r',
            cmap='hot_r', colorbar=True,
            vmin=0, vmax=vol.max(),
            threshold=vol.max() * 0.02,
            title=f'{s.upper()}  ({len(good_idx)}/{r["n_ch"]} ch kept)')
        display.savefig(tmp, dpi=150)
        display.close()
        tmp_paths.append(tmp)

    # Stitch the 4 per-subject brain images into a 2×2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Channel importance heatmap on MNI brain\n'
                 'max |r| across bands (pos CV) — Gaussian-smoothed, lateral view',
                 fontsize=12, fontweight='bold')
    for ax, tmp in zip(axes.flat, tmp_paths):
        ax.imshow(mpimg.imread(tmp))
        ax.axis('off')
        os.unlink(tmp)

    import os
    os.makedirs(f'{out_dir}/pipeline', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/pipeline/fig10_channel_topo.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved figs/pipeline/fig10_channel_topo.png')
