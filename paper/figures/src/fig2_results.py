"""Figure 2: exact recovery of whole answers, Patchscope at the same layer (prior work) vs an early layer (ours).

Single AAAI column (3.3 in). Dumbbell chart: one row per item set, an arrow from the same-layer rate to the
early-layer rate. Rates are the percentage of held-out items for which a majority of the 8 copy-prompt variants
write the exact string. Numbers copied from figures/src/fig2_identity.tex (run directories listed there):
14B read at layer 32, written at layer 4; 8B read at layer 29, written at layer 8 (chosen after layer 4 failed).
Token lenses score 0% on every set (they output single tokens); this is stated in the caption.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import paperstyle as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "fig2_results.pdf")

OURS = "#1F5AA6"
BASE = "#8C8C8C"

# (group, label, same-layer %, early-layer %)
ROWS = [
    ("Qwen3-14B", "Two-word names", 20.6, 46.9),
    ("Qwen3-14B", "  fresh items", 8.6, 33.7),
    ("Qwen3-14B", "Sub-word strings", 12.9, 31.0),
    ("Qwen3-14B", "  fresh items", 4.4, 16.9),
    ("Qwen3-8B", "Two-word names", 25.0, 49.4),
    ("Qwen3-8B", "Sub-word strings", 16.8, 37.9),
]

fig, ax = plt.subplots(figsize=(3.3, 2.35))
ys, y, prev = [], 0.0, None
for g, *_ in ROWS:
    if prev is not None and g != prev:
        y += 0.9          # gap between the two models
    ys.append(y)
    y += 1.0
    prev = g

for (g, lab, a, b), yc in zip(ROWS, ys):
    ax.annotate("", xy=(b - 1.3, yc), xytext=(a + 1.3, yc),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=OURS, lw=1.1,
                                shrinkA=0, shrinkB=0), zorder=2)
    ax.plot(a, yc, "o", ms=5.2, mfc="white", mec=BASE, mew=1.2, zorder=3)
    ax.plot(b, yc, "o", ms=5.6, mfc=OURS, mec=OURS, zorder=3)
    if a < 8:   # too close to the 0% line: put the value above the dot
        ax.text(a, yc - 0.32, f"{a:.0f}", ha="center", va="bottom", color=BASE, fontsize=st.FONT_PT)
    else:
        ax.text(a - 2.0, yc, f"{a:.0f}", ha="right", va="center", color=BASE, fontsize=st.FONT_PT)
    ax.text(b + 2.0, yc, f"{b:.0f}%", ha="left", va="center", color=OURS, fontsize=st.FONT_PT, fontweight="bold")

# row labels and model group labels
ax.set_yticks(ys)
ax.set_yticklabels([r[1] for r in ROWS])
ax.tick_params(axis="y", length=0, pad=4)
for g in ("Qwen3-14B", "Qwen3-8B"):
    first = ys[[r[0] for r in ROWS].index(g)]
    ax.text(-0.02, first - 0.85, g, transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=st.FONT_PT, fontweight="bold", color=st.INK)


ax.set_xlim(0, 62)
ax.set_xticks([0, 20, 40, 60])
ax.set_xticklabels(["0", "20", "40", "60%"])
ax.set_ylim(ys[-1] + 0.8, ys[0] - 1.35)
ax.spines["left"].set_color("#BDBDBD")
ax.spines["left"].set_linewidth(0.8)
ax.grid(axis="x", color="#E6E6E6", lw=0.6, zorder=0)
ax.set_axisbelow(True)
ax.set_xlabel("Exact-match accuracy", labelpad=3)

handles = [Line2D([], [], marker="o", ls="none", ms=5.2, mfc="white", mec=BASE, mew=1.2),
           Line2D([], [], marker="o", ls="none", ms=5.6, mfc=OURS, mec=OURS)]
fig.legend(handles, ["same layer (prior work)", "early layer (ours)"], loc="upper center", ncol=2,
           bbox_to_anchor=(0.56, 1.02), frameon=False, handletextpad=0.3, columnspacing=1.2, fontsize=st.FONT_PT)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
print("wrote", os.path.normpath(OUT))
