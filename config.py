"""
Shared constants for the ECoG joystick-tracking decoding pipeline.
"""

import os

# ── Sampling & feature rates ──────────────────────────────────────────────────
FS        = 1000   # recording rate (Hz)
FS_FEAT   = 10     # feature rate after decimation (Hz)
DECIM     = FS // FS_FEAT
SMOOTH_HZ = 1.5    # LPF cutoff for band envelope smoothing

# ── Cross-validation ──────────────────────────────────────────────────────────
N_SPLITS   = 5
BLOCK_S    = 2    # stratification block length (seconds at FS_FEAT)

EMBARGO_S = 2.0                              # seconds; covers 1.5 Hz envelope memory (~1 s) + 400 ms max lag
EMBARGO   = int(round(EMBARGO_S * FS_FEAT))  # = 20 samples at 10 Hz

# ── Bad-channel thresholds ────────────────────────────────────────────────────
FLAT_AMP_THRESH     = 1.0     # channel std below this → dead/flat (int16 units; PREP find_bad_by_nan_flat)
AMP_Z_THRESH        = 3.5     # modified z-score of IQR×0.7413 across channels (PREP find_bad_by_deviation)
LINE_NOISE_Z_THRESH = 3.5     # modified z-score of log(line-noise SNR); None = disabled

# ── Line-noise removal ────────────────────────────────────────────────────────
# 'notch'        : IIR notch + filtfilt — zero-phase but creates spectral holes at 60/120/180 Hz
# 'spectrum_fit' : PREP/CleanLine multi-taper spectral fit (via mne) — subtracts only the
#                  sinusoidal component; no spectral hole; tolerates mains-frequency drift
LINE_NOISE_METHOD = 'spectrum_fit'

# ── Spectral bands ────────────────────────────────────────────────────────────
# (name, lo_hz, hi_hz, butterworth_order)
# hg2 is included but note the recording BPF attenuates above ~200 Hz
BANDS = [
    ('delta',   0.5,    4.0,   2),
    ('theta',   4.0,    8.0,   2),
    ('alpha',   8.0,   14.0,   4),
    ('beta',   14.0,   30.0,   4),
    ('lgamma', 30.0,   70.0,   4),
    ('hg1',    70.0,  150.0,   4),
    ('hg2',   150.0,  300.0,   4),
]
BAND_NAMES  = [b[0] for b in BANDS] + ['lmp']
BAND_LABELS = ['δ 0.5–4', 'θ 4–8', 'α 8–14', 'β 14–30',
               'lγ 30–70', 'hγ1 70–150', 'hγ2 150–300', 'LMP <1.5 Hz']

LMP_HZ = 1.5   # LMP low-pass cutoff; signed, unrectified

# ── Velocity rail masking ─────────────────────────────────────────────────────
RAIL_TOL    = 5     # distance from int16 boundary [0, 32767] to flag as railed
RAIL_PAD_MS = 100   # dilation padding around each rail epoch (ms)

# ── Decoder ───────────────────────────────────────────────────────────────────
TOP_N         = 10   # top (channel, band) pairs selected globally by |r|
LAGS          = list(range(5))   # 0–4 samples at FS_FEAT = 0–400 ms
RIDGE_ALPHAS  = [10**i for i in range(3, 13)]   # 1e3 → 1e12 (z-scored features & target)
N_FOLDS       = 5    # contiguous time-block CV folds
POST_LPF_HZ   = 0.25  # fixed post-processing LPF cutoff (Hz); matches tracking task bandwidth

# Channel-ranking metric used by select_features.
# 'signed_mean' : |(r_cx + r_cy)/2|  — default; cancels opposite-sign pairs
# 'mean_abs'    : (|r_cx| + |r_cy|)/2 — axis-symmetric
# 'l2'          : sqrt(r_cx² + r_cy²)  — axis-symmetric, rewards strong single-axis r
SELECTION_METRIC = 'signed_mean'

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = '/Users/arthurhsia/Desktop/joystick_track'
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR  = os.path.join(BASE_DIR, 'figs')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Subjects & display ────────────────────────────────────────────────────────
SUBJECTS = ['fp', 'gf', 'rh', 'rr']
COLORS   = {'fp': '#1f77b4', 'gf': '#ff7f0e', 'rh': '#2ca02c', 'rr': '#d62728'}
