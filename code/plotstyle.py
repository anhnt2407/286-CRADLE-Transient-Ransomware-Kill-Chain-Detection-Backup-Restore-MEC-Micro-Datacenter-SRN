"""
plotstyle.py
============
Shared Matplotlib style for the CRADLE figures.

All figure text is set in Times New Roman (text and math) so the plots match the
IEEEtran manuscript body font.  The palette is monochrome: series are separated
by line style and by small distinct markers rather than by colour, which keeps
the figures legible in greyscale print.  ``savefig`` emits BOTH a ``.png`` (for
quick viewing / README) and a ``.pdf`` (for LaTeX).
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- monochrome greyscale levels ------------------------------------------ #
K0 = "#000000"   # black
K1 = "#3F3F3F"   # dark grey
K2 = "#6E6E6E"   # mid grey
K3 = "#9A9A9A"   # light grey
K4 = "#C4C4C4"   # very light grey
K5 = "#E2E2E2"   # near white

CYCLE = [K0, K1, K2, K3, K4]

# distinct small markers, used in the order series are drawn
MARKERS = ["o", "s", "^", "d", "v", ">", "<", "p"]
# distinct dash patterns, used in the order series are drawn
DASHES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]

MS = 3.0          # small marker size
LW = 1.2          # line width

_SERIF = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "font.size": 9,
    "font.family": "serif",
    "font.serif": _SERIF,
    # math text in Times New Roman as well
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "mathtext.cal": "Times New Roman:italic",
    "axes.grid": True,
    "grid.color": K4,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.7,
    "axes.axisbelow": True,
    "axes.edgecolor": K0,
    "axes.labelcolor": K0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": K0,
    "ytick.color": K0,
    "axes.prop_cycle": plt.cycler(color=CYCLE),
    "lines.linewidth": LW,
    "lines.markersize": MS,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "text.color": K0,
})

FIG1 = (3.4, 2.55)      # single-column figure
FIG2 = (7.0, 2.7)       # double-column / two-panel


def newfig(xlabel="Time (days)", ylabel="", title="", figsize=FIG1, xlim=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9)
    if xlim is not None:
        ax.set_xlim(*xlim)
    return fig, ax


def series(ax, x, y, k: int, label=None, every=None, **kw):
    """
    Draw curve ``k`` of a figure with the k-th greyscale level, dash pattern and
    small marker.  ``every`` thins the markers on a dense analytic grid.
    """
    style = dict(color=CYCLE[k % len(CYCLE)], linewidth=LW, label=label,
                 marker=MARKERS[k % len(MARKERS)], markersize=MS,
                 markerfacecolor="none", markeredgewidth=0.7)
    ls = DASHES[k % len(DASHES)]
    if isinstance(ls, tuple):
        style["linestyle"] = ls
    else:
        style["linestyle"] = ls
    if every is not None:
        style["markevery"] = every
    style.update(kw)
    return ax.plot(x, y, **style)[0]


def savefig(fig, stem: str, figdir: str):
    """Write both PNG and PDF of `fig` to `figdir/stem.{png,pdf}`."""
    os.makedirs(figdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, f"{stem}.{ext}"))
    plt.close(fig)
