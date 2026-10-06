"""Figure 3: does it matter where the state is written? Qwen3-14B, state read at layer 32, 175 held-out two-word items.

Single AAAI column (3.3 in), same visual language as Figure 2. One row per way of writing the state; one marker per
carrier. Copy prompt: majority of 8 variants write the exact string. Continuation prompt: the exact string is a prefix
of the output. Numbers copied from figures/src/fig3_mechanism.tex (run directories listed there).
"""
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import paperstyle as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "fig3_insertion.pdf")

COPY_C = "#1F5AA6"
CONT_C = "#7FA7D9"

ROWS = ["Same layer (32)", "Every layer up to 32", "Every layer up to 4", "Layer 4 only"]
COPY = [20.6, 34.9, 47.4, 46.9]
CONT = [37.7, 68.6, 73.7, 76.0]

fig, ax = plt.subplots(figsize=(3.3, 1.7))
ys = list(range(len(ROWS)))
for yc, a, b in zip(ys, COPY, CONT):
    ax.plot([a, b], [yc, yc], color="#D9D9D9", lw=2.2, zorder=1, solid_capstyle="round")
    ax.plot(a, yc, "o", ms=5.6, color=COPY_C, zorder=3)
    ax.plot(b, yc, "D", ms=5.0, color=CONT_C, mec="#4F79B0", mew=0.6, zorder=3)
    ax.text(a - 4.0, yc, f"{a:.0f}%", ha="right", va="center", color=COPY_C, fontsize=st.FONT_PT)
    ax.text(b + 4.0, yc, f"{b:.0f}%", ha="left", va="center", color="#4F79B0", fontsize=st.FONT_PT)

ax.set_yticks(ys)
ax.set_yticklabels(ROWS)
ax.tick_params(axis="y", length=0, pad=4)
ax.set_ylim(len(ROWS) - 0.5, -0.6)
ax.set_xlim(0, 95)
ax.set_xticks([0, 25, 50, 75])
ax.set_xticklabels(["0", "25", "50", "75%"])
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color="#E6E6E6", lw=0.6, zorder=0)
ax.set_axisbelow(True)
ax.set_xlabel("Exact-match accuracy", labelpad=3)

handles = [Line2D([], [], marker="o", ls="none", ms=5.6, color=COPY_C),
           Line2D([], [], marker="D", ls="none", ms=5.0, color=CONT_C, mec="#4F79B0", mew=0.6)]
fig.legend(handles, ["copy prompt", "continuation prompt"], loc="upper center", ncol=2,
           bbox_to_anchor=(0.6, 1.03), frameon=False, handletextpad=0.3, columnspacing=1.2, fontsize=st.FONT_PT)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
print("wrote", os.path.normpath(OUT))
