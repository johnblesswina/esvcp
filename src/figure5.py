"""Figure 5 — MLCI bit-plane decomposition and payload allocation.
Answers Reviewer 1's request for a pictorial explanation of multi-layer
embedding. All panels are real bit-planes of a deposited cover image."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
import esvcp_fixed as E

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
INK, MUTE, BLU, PUR = "#1A1A1A", "#5A5A5A", "#1B4F72", "#4A3B6B"

cov = np.array(Image.open("data/cover/1.pgm"))
valid, _ = E.eligibility(cov)

fig = plt.figure(figsize=(7.09, 4.55), dpi=300)
fig.patch.set_facecolor("white")

# ── row 1: the eight bit-planes ──────────────────────────────────────
for k in range(8):
    ax = fig.add_axes([0.035 + k * 0.1195, 0.585, 0.105, 0.245])
    ax.imshow((cov >> k) & 1, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    used = k <= 2
    for s in ax.spines.values():
        s.set_linewidth(1.4 if used else 0.6)
        s.set_color(BLU if used else "#BBBBBB")
    ax.set_title(f"$B_{k}$", fontsize=8, fontweight="bold" if used else "normal",
                 color=BLU if used else MUTE, pad=3)
    lbl = {0: "60% payload", 1: "30% payload", 2: "10% payload"}.get(k, "untouched")
    ax.set_xlabel(lbl, fontsize=5.9, color=BLU if used else MUTE,
                  fontweight="bold" if used else "normal", labelpad=2)

fig.text(0.5, 0.875, "(a)  Bit-plane decomposition of the cover image",
         ha="center", fontsize=8.5, fontweight="bold", color=INK)
fig.text(0.5, 0.855,
         "$B_0$–$B_2$ carry the payload;  $B_3$–$B_7$ are never modified and "
         "therefore reproduce the eligibility mask at the receiver",
         ha="center", fontsize=6.2, style="italic", color=MUTE)

# ── row 2 left: eligibility mask ─────────────────────────────────────
ax = fig.add_axes([0.035, 0.115, 0.225, 0.345])
ax.imshow(valid, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_linewidth(0.8); s.set_color(PUR)
ax.set_title("(b)  Eligible set $\\Omega_{elig}$", fontsize=8,
             fontweight="bold", color=INK, pad=4)
ax.set_xlabel(f"white = $\\tilde{{H}} \\geq 0.20$   ({valid.mean()*100:.1f}% of pixels)",
              fontsize=6.2, color=MUTE, labelpad=3)

# ── row 2 right: allocation schematic ────────────────────────────────
ax = fig.add_axes([0.315, 0.115, 0.650, 0.345])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("(c)  Payload allocation across planes", fontsize=8,
             fontweight="bold", color=INK, pad=4)

ax.add_patch(FancyBboxPatch((0.02, 0.72), 0.30, 0.20,
             boxstyle="round,pad=0,rounding_size=0.03", linewidth=0.8,
             edgecolor=PUR, facecolor="#F2EFF6"))
ax.text(0.17, 0.865, "chaotic permutation $\\pi$", ha="center", fontsize=6.6,
        fontweight="bold", color=PUR)
ax.text(0.17, 0.79, "filtered to $\\Omega_{elig}$\n$L$ eligible positions",
        ha="center", va="center", fontsize=6.1, color=INK)

shares = [("$B_0$", 0.60, "#1B4F72"), ("$B_1$", 0.30, "#3C7CB5"),
          ("$B_2$", 0.10, "#8FBEE0")]
x0 = 0.40
for lab, frac, c in shares:
    w = frac * 0.56
    ax.add_patch(FancyBboxPatch((x0, 0.72), w, 0.20,
                 boxstyle="round,pad=0,rounding_size=0.02",
                 linewidth=0.8, edgecolor="white", facecolor=c))
    ax.text(x0 + w / 2, 0.82, lab, ha="center", va="center",
            fontsize=6.6, fontweight="bold", color="white")
    ax.text(x0 + w / 2, 0.665, f"{int(frac*100)}%", ha="center", va="center",
            fontsize=6.2, fontweight="bold", color=c)
    x0 += w
ax.add_patch(FancyArrowPatch((0.325, 0.82), (0.395, 0.82), arrowstyle="-|>",
             mutation_scale=7, linewidth=0.8, color=MUTE))

ax.text(0.5, 0.545,
        "Each eligible pixel receives exactly one bit, in exactly one plane.",
        ha="center", fontsize=6.8, fontweight="bold", color=INK)
ax.text(0.5, 0.435,
        "The three index sets are disjoint slices of a single permuted sequence, so the\n"
        "plane allocation distributes the payload across planes rather than multiplying\n"
        "capacity. The ceiling is $|\\Omega_{elig}| / (M \\cdot N)$, measured at 0.52 bpp.",
        ha="center", va="top", fontsize=6.2, color=MUTE, linespacing=1.6)

ax.add_patch(FancyBboxPatch((0.005, 0.015), 0.99, 0.125,
             boxstyle="round,pad=0,rounding_size=0.03", linewidth=0.8,
             edgecolor=BLU, facecolor="white", linestyle=(0, (2.5, 2))))
ax.text(0.5, 0.078,
        "Extraction reverses the order:  rebuild $\\Omega_{elig}$ from $B_3$–$B_7$  "
        "\u2192  regenerate $\\pi$  \u2192  read $B_0B_1B_2$",
        ha="center", va="center", fontsize=6.3, fontweight="bold", color=BLU)

fig.savefig("out2/Figure5.tif", dpi=600, bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"})
fig.savefig("out2/Figure5.png", dpi=300, bbox_inches="tight")
print("wrote out2/Figure5.tif and out2/Figure5.png")
