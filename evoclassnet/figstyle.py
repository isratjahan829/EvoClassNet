"""Publication figure style.

One place that fixes typography, colour and geometry for every figure, so the set reads as one
piece of work rather than eleven separately styled plots.

Conventions follow what medical and multidisciplinary journals ask for: a sans-serif face at
7-9 pt, hairline axes with no top or right spine, no chart junk, column widths quoted in
millimetres, and a colour-blind-safe palette. The palette is Okabe-Ito, which stays legible under
deuteranopia, protanopia and tritanopia and survives greyscale printing; where colour carries
meaning it is reinforced by position or by a direct label, never by hue alone.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Journal column widths in inches (89 mm, 120 mm, 183 mm).
SINGLE, MEDIUM, DOUBLE = 3.504, 4.724, 7.205

# Okabe-Ito, colour-blind safe.
OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", vermillion="#D55E00", purple="#CC79A7", black="#000000")
GREY = "#6E6E6E"
LIGHT = "#D9D9D9"
INK = "#1A1A1A"

# Model families keep one colour across every figure.
FAMILY = {"untrained": OI["green"], "position-indexed": OI["vermillion"],
          "transferable": OI["blue"], "published": OI["purple"], "this work": OI["orange"]}
# Split identity, used where two protocols are shown side by side.
SPLIT = {"random": GREY, "position": OI["blue"]}


def use():
    """Apply the style globally. Call once at the top of a figure script."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "FreeSans"],
        "font.size": 7.5,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 9.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "axes.titlepad": 5,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,
        "patch.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed as TrueType, not outlines
        "text.color": INK,
    })


def panel(ax, letter, dx=-0.085, dy=1.06, size=10):
    """Bold lower-case panel label, positioned in axes coordinates."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size, fontweight="bold",
            va="top", ha="left", color=INK)


def zeroline(ax, x=True, value=0.0):
    """A hairline reference at zero, drawn behind the data."""
    f = ax.axvline if x else ax.axhline
    f(value, color=INK, lw=0.6, zorder=0)


def despine(ax, left=True, bottom=True):
    for side, keep in (("left", left), ("bottom", bottom)):
        ax.spines[side].set_visible(keep)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, path, also_pdf=True):
    """Write a 600 dpi PNG and, alongside it, a vector PDF for typesetting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path
