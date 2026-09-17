"""Figure 1 — overall ESVCP architecture.
Two parallel tracks: share generation and embedding. No crossing connectors.
Two-column width, 600 dpi TIFF."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})

INK, MUTE = "#1A1A1A", "#5A5A5A"
PUR, BLU, GRN, GRY = "#4A3B6B", "#1B4F72", "#145A32", "#6B6B6B"

fig = plt.figure(figsize=(7.09, 4.25), dpi=300)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def box(x, y, w, h, title, lines, edge, fill, fs_t=7.4, fs_b=6.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=0.010",
                 linewidth=0.85, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h - 0.036, title, ha="center", va="center",
            fontsize=fs_t, fontweight="bold", color=edge)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.075 - i * 0.037, ln, ha="center",
                va="center", fontsize=fs_b, color=INK)


def arr(x1, y1, x2, y2, c=MUTE, lw=0.9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=8, linewidth=lw, color=c))


ax.text(0.5, 0.965, "ESVCP — five-stage architecture", ha="center", va="center",
        fontsize=10, fontweight="bold", color=INK)

# ── share generation track ───────────────────────────────────────────
ax.text(0.035, 0.885, "Share generation path", ha="left", va="center",
        fontsize=6.8, fontweight="bold", style="italic", color=GRN)

box(0.035, 0.625, 0.455, 0.235, "Stage 1 — SASG",
    ["input: cover $I_c$ and secret image $S$",
     "spectral residual saliency  \u2192  Otsu threshold $\\tau$",
     "partition $\\Omega_{sal}$ (protected) / $\\Omega_{free}$ (modifiable)",
     "$V_1 = R$,   $V_2 = V_1 \\oplus S$",
     "carrier embed $C_k=(I_c\\wedge$0xFE$)\\vee V_k$ on $\\Omega_{free}$"],
    GRN, "#EBF3EE")

box(0.585, 0.660, 0.380, 0.165, "Share carriers",
    ["$C_1$, $C_2$   \u2014   51.73 dB, SSIM 0.9962",
     "$\\Omega_{sal}$ bit-identical to cover",
     "$C_1\\wedge 1 \\oplus C_2\\wedge 1 = S$  (exact)"], GRN, "white")

arr(0.490, 0.742, 0.585, 0.742, c=GRN)

# ── embedding track ──────────────────────────────────────────────────
ax.text(0.035, 0.560, "Embedding path", ha="left", va="center",
        fontsize=6.8, fontweight="bold", style="italic", color=BLU)

W, G = 0.212, 0.038
xs = [0.035 + i * (W + G) for i in range(4)]
YB, HB = 0.300, 0.235

box(xs[0], YB, W, HB, "Stage 2 — entropy map",
    ["input: cover $I_c$", "mask high planes $I_c\\wedge$0xF8",
     "$\\tilde{H}(i,j)$, 7$\\times$7 window", "scaled to [0,1]"], PUR, "#F2EFF6")

box(xs[1], YB, W, HB, "Stage 3 — MLCI index",
    ["input: secret bits, seed $x_0$", "$x_{n+1}=\\mu x_n(1-x_n)$",
     "$\\mu = 3.9999$, warm-up 100", "$\\pi=\\mathrm{argsort}(x_1\\ldots x_{MN})$"],
    BLU, "#ECF2F7")

box(xs[2], YB, W, HB, "Stage 4 — AEDQ",
    ["entropy-domain qualification", "$\\Omega_{elig}=\\{\\tilde{H}\\geq 0.20\\}$",
     "$\\pi$ filtered to $\\Omega_{elig}$", "ceiling 0.52 bpp"], PUR, "#F2EFF6")

box(xs[3], YB, W, HB, "Stage 5 — embedding",
    ["$B_0$ 60%  $B_1$ 30%  $B_2$ 10%", "one bit per eligible pixel",
     "$B_3$–$B_7$ left unmodified", "$I_s$ — 49.75 dB, SSIM 0.9977"], BLU, "#ECF2F7")

for i in range(3):
    arr(xs[i] + W, YB + HB / 2, xs[i + 1], YB + HB / 2)

# ── extraction ───────────────────────────────────────────────────────
box(0.035, 0.055, 0.930, 0.150, "Extraction  —  receiver holds the stego-image and the key only",
    ["recompute $\\tilde{H}$ from $I_s\\wedge$0xF8  \u2192  rebuild $\\Omega_{elig}$  "
     "\u2192  regenerate $\\pi$ from $x_0$  \u2192  read $B_0B_1B_2$",
     "the cover image is never required   \u21d2   BER = 0 on every test image"],
    GRY, "#F5F5F5", 7.2, 6.2)

arr(xs[3] + W / 2, YB, xs[3] + W / 2, 0.205)

fig.savefig("out2/Figure1.tif", dpi=600, bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"})
fig.savefig("out2/Figure1.png", dpi=300, bbox_inches="tight")
print("wrote out2/Figure1.tif and out2/Figure1.png")
