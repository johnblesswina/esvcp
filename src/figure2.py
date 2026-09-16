"""Figure 2 - ESVCP architecture. Publication format.
Two-column width (180 mm), 600 dpi TIFF, LZW.
All numeric values are measured (n = 50 BOSSBase v1.01 images)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
})

INK = "#1A1A1A"
MUTE = "#5A5A5A"

COLS = [
    dict(
        panel="a", name="AEDQ",
        sub="Adaptive Entropy-Domain Qualification",
        accent="#4A3B6B", tint="#F2EFF6",
        boxes=[
            ("Cover image", ["$I_c$ :  $M \\times N$, 8-bit grayscale"]),
            ("High-plane mask", ["$I_c \\wedge$ 0xF8   (retains $B_3$–$B_7$)",
                                 "invariant under LSB embedding"]),
            ("Local entropy", ["$\\tilde{H}(i,j)$ over 7$\\times$7 window",
                               "$H = -\\sum_v p_v \\log_2 p_v$,  scaled to [0,1]"]),
            ("Eligibility test", ["$\\Omega_{elig} = \\{(i,j) : \\tilde{H}(i,j) \\geq 0.2\\}$"]),
            ("Eligible set", ["identically recoverable by receiver",
                              "from the stego-image alone"]),
        ],
        result=["Capacity ceiling  0.52 bpp", "(sd 0.20, range 0.13–0.89)"],
    ),
    dict(
        panel="b", name="MLCI",
        sub="Multi-Layer LSB with Chaotic Permutation Indexing",
        accent="#1B4F72", tint="#ECF2F7",
        boxes=[
            ("Secret and key", ["bitstream $S$,  seed $x_0 \\in (0,1)$"]),
            ("Chaotic sequence", ["$x_{n+1} = \\mu x_n (1 - x_n)$,  $\\mu = 3.9999$",
                                  "100-iteration warm-up discarded"]),
            ("Permutation index", ["$\\pi = \\mathrm{argsort}(x_1 \\ldots x_{MN})$",
                                   "filtered to $\\Omega_{elig}$"]),
            ("Plane allocation", ["$B_0$ 60%    $B_1$ 30%    $B_2$ 10%",
                                  "disjoint index slices of $\\pi$"]),
            ("Stego-image", ["$I_s$ :  $B_3$–$B_7$ unmodified"]),
        ],
        result=["PSNR 49.75 dB,  SSIM 0.9977", "BER 0,  key space $2^{64}$"],
    ),
    dict(
        panel="c", name="SASG",
        sub="Semantic-Aware Share Generation",
        accent="#145A32", tint="#EBF3EE",
        boxes=[
            ("Cover and secret", ["$I_c$ and secret image $S$"]),
            ("Saliency map", ["spectral residual",
                              "$SR = \\log A - h_{3\\times3} * \\log A$"]),
            ("Region partition", ["Otsu threshold $\\tau$",
                                  "$\\Omega_{sal}$ protected  /  $\\Omega_{free}$ modifiable"]),
            ("Share pair", ["$V_1 = R$,   $V_2 = V_1 \\oplus S$",
                            "$V_1 \\oplus V_2 = S$ at every position"]),
            ("Carrier embedding", ["$C_k = (I_c \\wedge$ 0xFE$) \\vee V_k$ on $\\Omega_{free}$",
                                   "$C_k = I_c$ on $\\Omega_{sal}$"]),
        ],
        result=["Share carriers  51.73 dB", "vs 4.79 dB classical (2,2)-VCS"],
    ),
]

FW, FH = 7.09, 5.45
SV = 7.4 / FH          # vertical rescale so content fills the canvas
fig = plt.figure(figsize=(FW, FH), dpi=300)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

LM, RM = 0.035, 0.035
CGAP = 0.042
CW = (1 - LM - RM - 2 * CGAP) / 3
TOP = 0.925
HDR = 0.055 * SV
VGAP = 0.0165 * SV
LINE_H = 0.0175 * SV
PAD_T, PAD_B = 0.019 * SV, 0.011 * SV

for c, col in enumerate(COLS):
    x0 = LM + c * (CW + CGAP)
    xc = x0 + CW / 2
    A = col["accent"]

    ax.text(x0, TOP + 0.028 * SV, f"({col['panel']})", ha="left", va="bottom",
            fontsize=10, fontweight="bold", color=INK)

    ax.add_patch(Rectangle((x0, TOP - HDR), CW, HDR,
                           facecolor=A, edgecolor="none"))
    ax.text(xc, TOP - HDR * 0.36, col["name"], ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="white")
    ax.text(xc, TOP - HDR * 0.76, col["sub"], ha="center", va="center",
            fontsize=5.4, color="white")

    y = TOP - HDR - VGAP
    for i, (label, lines) in enumerate(col["boxes"]):
        bh = PAD_T + PAD_B + LINE_H * len(lines) + 0.014 * SV
        ax.add_patch(FancyBboxPatch(
            (x0, y - bh), CW, bh,
            boxstyle="round,pad=0,rounding_size=0.006",
            linewidth=0.7, edgecolor=A, facecolor=col["tint"]))
        ax.text(xc, y - PAD_T + 0.001 * SV, label, ha="center", va="center",
                fontsize=6.8, fontweight="bold", color=A)
        for j, ln in enumerate(lines):
            ax.text(xc, y - PAD_T - 0.013 * SV - j * LINE_H, ln,
                    ha="center", va="center", fontsize=6.0, color=INK)
        if i < len(col["boxes"]) - 1:
            ax.add_patch(FancyArrowPatch((xc, y - bh), (xc, y - bh - VGAP),
                         arrowstyle="-|>", mutation_scale=6.5,
                         linewidth=0.7, color=A))
        y -= bh + VGAP
    col["_bottom"] = y + VGAP
    col["_x0"], col["_xc"] = x0, xc

a, b = COLS[0], COLS[1]
ylat = min(a["_bottom"], b["_bottom"]) + 0.028 * SV * SV
ax.add_patch(FancyArrowPatch((a["_x0"] + CW, ylat), (b["_x0"], ylat),
                             arrowstyle="-|>", mutation_scale=6.5,
                             linewidth=0.8, color=MUTE))
ax.text((a["_x0"] + CW + b["_x0"]) / 2, ylat + 0.006 * SV,
        "$\\Omega_{elig}$", ha="center", va="bottom", fontsize=6.2, color=MUTE)

y_link = min(c["_bottom"] for c in COLS) - 0.028 * SV
for col in COLS:
    ax.add_patch(FancyArrowPatch((col["_xc"], col["_bottom"]),
                                 (col["_xc"], y_link + 0.002),
                                 arrowstyle="-", linewidth=0.7,
                                 color=col["accent"], linestyle=(0, (2.5, 2))))

for col in COLS:
    rh = 0.050 * SV
    ax.add_patch(FancyBboxPatch((col["_x0"], y_link - rh), CW, rh,
                 boxstyle="round,pad=0,rounding_size=0.006",
                 linewidth=0.7, edgecolor=col["accent"], facecolor="white",
                 linestyle=(0, (2.5, 2))))
    for j, ln in enumerate(col["result"]):
        ax.text(col["_xc"], y_link - 0.017 * SV - j * 0.0160 * SV, ln,
                ha="center", va="center", fontsize=6.1,
                fontweight="bold" if j == 0 else "normal", color=col["accent"])

ax.text(0.5, y_link - (0.050 + 0.022) * SV,
        "Measured values: $n$ = 50 BOSSBase v1.01 images at 0.5 bpp payload",
        ha="center", va="top", fontsize=5.8, style="italic", color=MUTE)

fig.savefig("out2/Figure2.tif", dpi=600, bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"})
fig.savefig("out2/Figure2.png", dpi=300, bbox_inches="tight")
print("wrote out2/Figure2.tif and out2/Figure2.png")
