"""
plotstyle.py
============
Shared Matplotlib style for the CRADLE figures.

Colour-vision-deficiency-safe Okabe-Ito palette (the group standard), serif
fonts to blend with an IEEEtran manuscript, and a ``savefig`` helper that emits
BOTH a ``.png`` (for quick viewing / README) and a ``.pdf`` (for LaTeX).
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Okabe-Ito colour-blind-safe palette ---------------------------------- #
BLUE   = "#0072B2"   # encrypting / primary
VERM   = "#D55E00"   # locked (service denial)
GREEN  = "#009E73"   # restored / recovery
ORANGE = "#E69F00"   # exfiltrating / breach
SKY    = "#56B4E9"   # compromised
PURPLE = "#CC79A7"   # detection
YELLOW = "#F0E442"   # auxiliary
BLACK  = "#000000"
GRAY   = "#7F7F7F"

CYCLE = [BLUE, VERM, GREEN, ORANGE, SKY, PURPLE, GRAY]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.grid": True,
    "grid.color": "#BBBBBB",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": plt.cycler(color=CYCLE),
    "lines.linewidth": 1.8,
    "legend.frameon": False,
    "legend.fontsize": 8,
})

FIG1 = (3.4, 2.55)      # single-column figure
FIG2 = (7.0, 2.7)       # double-column / two-panel


def newfig(xlabel="Time $t$ (days)", ylabel="", title="", figsize=FIG1, xlim=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9)
    if xlim is not None:
        ax.set_xlim(*xlim)
    return fig, ax


def savefig(fig, stem: str, figdir: str):
    """Write both PNG and PDF of `fig` to `figdir/stem.{png,pdf}`."""
    os.makedirs(figdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, f"{stem}.{ext}"))
    plt.close(fig)
