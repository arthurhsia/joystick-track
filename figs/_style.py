"""Shared matplotlib style and colour palettes for the pipeline figure set."""

import matplotlib.pyplot as plt

# Match the palettes already used in config.py / hybrid_plots.py
SUBJ_COLORS = {
    'fp': '#1f77b4',
    'gf': '#ff7f0e',
    'rh': '#2ca02c',
    'rr': '#d62728',
}

AXIS_COLORS = {'cx': '#e377c2', 'cy': '#17becf'}

TRACE_COLORS = {
    'true':   '#222222',
    'ridge':  '#9467bd',
    'fixed':  '#2ca02c',
    'oracle': '#aaaaaa',
}

BAND_COLORS = [
    '#4878cf', '#6acc65', '#d65f5f', '#b47cc7',
    '#c4ad66', '#77bedb', '#f28e2b', '#e15759',
]


def set_style():
    plt.rcParams.update({
        'font.family':       'sans-serif',
        'font.size':          9,
        'axes.titlesize':    10,
        'axes.titleweight':  'bold',
        'axes.labelsize':     9,
        'xtick.labelsize':    8,
        'ytick.labelsize':    8,
        'legend.fontsize':    8,
        'legend.framealpha':  0.8,
        'figure.dpi':        100,
        'savefig.dpi':       300,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'lines.linewidth':    1.2,
        'pdf.fonttype':      42,
    })
