"""Shared matplotlib style for the paper figures (figures4papers house style, AAAI column).

Rules applied here: sans-serif (Helvetica, Arial fallback); no top/right spines; axis linewidth 0.8;
frameless legend; no gridlines; 9 pt text everywhere (the AAAI in-figure minimum at final size);
muted palette (deep blue for ours, light grey with a black edge for the baseline, red only for a
failure mark); vector PDF with Type 42 (embedded TrueType) fonts.
"""
import matplotlib as mpl

mpl.use("pdf")

# figures4papers palette (scientific-figure-making/references/design-theory.md)
BLUE = "#0F4D92"        # anchor blue: the proposed method (early layer, ours)
BLUE2 = "#3775BA"       # secondary blue
GREY_LIGHT = "#CFCECE"  # neutral support: the same-layer baseline bars
GREY = "#767676"        # neutral: error bars, secondary arrows and labels
GREY_DARK = "#4D4D4D"   # neutral: box edges, secondary bold labels
INK = "#272727"         # near-black text and lines
RED = "#B64342"         # failure marks only

FONT_PT = 9.0           # AAAI: in-figure text >= 9 pt at final size

# An explicit family list enables per-glyph fallback: Helvetica for everything, Arial Unicode MS only
# for glyphs Helvetica lacks (the Chinese and Vietnamese tokens in Figure 1); both embed as TrueType.
mpl.rcParams.update({
    "font.family": ["Helvetica", "Arial", "Arial Unicode MS", "DejaVu Sans"],
    "font.size": FONT_PT,
    "axes.titlesize": FONT_PT,
    "axes.labelsize": FONT_PT,
    "xtick.labelsize": FONT_PT,
    "ytick.labelsize": FONT_PT,
    "legend.fontsize": FONT_PT,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})
