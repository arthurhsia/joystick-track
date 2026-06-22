"""
Trial segmentation: detect inter-trial freeze gaps in cursor position.
"""

import numpy as np
from config import FS


def find_trial_boundaries(cx, cy, min_still=1500, min_trial_s=30):
    """
    Return (start, end) sample pairs for each active trial at FS Hz.

    Inter-trial gaps: both cx and cy exactly constant for ≥ min_still samples.
    Trials shorter than min_trial_s seconds are dropped.
    """
    cx = cx.astype(np.float64)
    cy = cy.astype(np.float64)
    still = (np.diff(cx, prepend=cx[0]) == 0) & (np.diff(cy, prepend=cy[0]) == 0)

    edges    = np.where(np.diff(still.astype(np.int8), prepend=0, append=0))[0]
    r_starts = edges[:-1]
    r_lens   = np.diff(edges)
    r_still  = still[r_starts]

    pause_mask   = r_still & (r_lens >= min_still)
    pause_starts = r_starts[pause_mask]
    pause_ends   = pause_starts + r_lens[pause_mask]

    T = len(cx)
    trial_starts = np.concatenate([[0], pause_ends])
    trial_ends   = np.concatenate([pause_starts, [T]])

    min_samples = int(min_trial_s * FS)
    return [(int(s), int(e)) for s, e in zip(trial_starts, trial_ends)
            if e - s >= min_samples]
