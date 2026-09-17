"""Figure 3 — experimental results. Six panels, all from measured CSVs in
results/. No steganalysis panel."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                     "axes.linewidth": 0.7})

RES = "/home/claude/esvcp/results/"
PUR, BLU, GRN, ORG, GRY = "#4A3B6B", "#1B4F72", "#145A32", "#B05A1A", "#8A8A8A"


def read(f):
    with open(RES + f) as fh:
        return list(csv.DictReader(fh))


sweep = read("payload_sweep.csv")
cap = read("capacity.csv")
abl = read("ablation.csv")
sh = read("shares.csv")
rt = read("runtime.csv")

fig, axes = plt.subplots(2, 3, figsize=(7.09, 4.3), dpi=300)
fig.patch.set_facecolor("white")

# (a) PSNR vs embedding rate
ax = axes[0, 0]
x = [float(r["bpp"]) for r in sweep]
y = [float(r["psnr_mean"]) for r in sweep]
e = [float(r["psnr_sd"]) for r in sweep]
ax.errorbar(x, y, yerr=e, color=BLU, marker="o", ms=3.5, lw=1.2,
            capsize=2.5, elinewidth=0.7, label="ESVCP")
for lbl, vals, c in [("HUGO [26]", [51.20, 48.30, None, 44.90, 41.30], GRY),
                     ("UNIWARD [28]", [52.88, 49.90, None, 46.30, 42.67], ORG)]:
    xs = [a for a, b in zip(x, vals) if b]
    ys = [b for b in vals if b]
    ax.plot(xs, ys, "--", color=c, marker="s", ms=2.8, lw=0.9, label=lbl)
ax.set_xlabel("Embedding rate (bpp)"); ax.set_ylabel("PSNR (dB)")
ax.set_title("(a) PSNR versus embedding rate", fontsize=8, fontweight="bold")
ax.legend(fontsize=5.6, frameon=False, loc="lower left")
ax.grid(alpha=0.25, lw=0.5)

# (b) capacity distribution
ax = axes[0, 1]
c = [float(r["max_bpp"]) for r in cap]
ax.hist(c, bins=12, color=PUR, alpha=0.75, edgecolor="white", lw=0.5)
ax.axvline(np.mean(c), color=ORG, lw=1.2, ls="--",
           label=f"mean {np.mean(c):.2f} bpp")
ax.set_xlabel("Capacity ceiling (bpp)"); ax.set_ylabel("Images")
ax.set_title("(b) Embedding capacity", fontsize=8, fontweight="bold")
ax.legend(fontsize=5.8, frameon=False)
ax.grid(alpha=0.25, lw=0.5, axis="y")

# (c) ablation
ax = axes[0, 2]
names = [r["configuration"].replace("without ", "w/o ") for r in abl[:3]]
vals = [float(r["psnr_mean"]) for r in abl[:3]]
errs = [float(r["psnr_sd"]) for r in abl[:3]]
bars = ax.bar(range(3), vals, yerr=errs, capsize=2.5,
              color=[BLU, GRY, GRY], edgecolor="white", lw=0.6,
              error_kw=dict(elinewidth=0.7))
ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=6, rotation=12)
ax.set_ylabel("PSNR (dB)"); ax.set_ylim(44, 58)
ax.set_title("(c) Module ablation", fontsize=8, fontweight="bold")
ax.grid(alpha=0.25, lw=0.5, axis="y")

# (d) recovery
ax = axes[1, 0]
b = [float(r["ber_max"]) for r in sweep]
ax.bar(range(len(x)), [1.0] * len(x), color=GRN, alpha=0.8,
       edgecolor="white", lw=0.6, label="NCC")
ax.plot(range(len(x)), b, color=ORG, marker="o", ms=3.5, lw=1.2, label="BER")
ax.set_xticks(range(len(x))); ax.set_xticklabels([f"{v:g}" for v in x], fontsize=6.5)
ax.set_xlabel("Embedding rate (bpp)"); ax.set_ylabel("NCC  /  BER")
ax.set_ylim(0, 1.15)
ax.set_title("(d) Secret recovery", fontsize=8, fontweight="bold")
ax.legend(fontsize=5.8, frameon=False, loc="center right")
ax.grid(alpha=0.25, lw=0.5, axis="y")

# (e) share quality
ax = axes[1, 1]
cp = [float(r["carrier_psnr"]) for r in sh]
cl = [float(r["classical_vcs_psnr"]) for r in sh]
ax.bar([0, 1], [np.mean(cl), np.mean(cp)],
       yerr=[np.std(cl, ddof=1), np.std(cp, ddof=1)], capsize=3,
       color=[GRY, GRN], edgecolor="white", lw=0.6,
       error_kw=dict(elinewidth=0.7))
ax.set_xticks([0, 1])
ax.set_xticklabels(["Classical\n(2,2)-VCS", "SASG\ncarriers"], fontsize=6.5)
ax.set_ylabel("Share PSNR (dB)")
ax.set_title("(e) Share perceptual quality", fontsize=8, fontweight="bold")
for i, v in enumerate([np.mean(cl), np.mean(cp)]):
    ax.text(i, v + 2, f"{v:.2f}", ha="center", fontsize=6.3, fontweight="bold")
ax.set_ylim(0, 62)
ax.grid(alpha=0.25, lw=0.5, axis="y")

# (f) runtime
ax = axes[1, 2]
lab = {"aedq": "AEDQ", "mlci_perm": "MLCI perm.", "sasg": "SASG",
       "embed": "Embed", "extract": "Extract"}
ks = ["aedq", "mlci_perm", "sasg", "embed", "extract"]
d = {r["stage"]: float(r["mean_ms"]) for r in rt}
ax.barh(range(len(ks)), [d[k] for k in ks],
        color=[PUR, BLU, GRN, BLU, BLU], edgecolor="white", lw=0.6)
ax.set_yticks(range(len(ks)))
ax.set_yticklabels([lab[k] for k in ks], fontsize=6.5)
ax.invert_yaxis()
ax.set_xlabel("Time (ms)")
ax.set_title("(f) Runtime, 512$\\times$512", fontsize=8, fontweight="bold")
ax.grid(alpha=0.25, lw=0.5, axis="x")

for a in axes.ravel():
    a.tick_params(labelsize=6.5, width=0.7, length=2.5)
    for s in a.spines.values():
        s.set_linewidth(0.7)
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

fig.tight_layout(pad=0.7, w_pad=1.4, h_pad=1.6)
fig.savefig("out2/Figure3.tif", dpi=600, bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"})
fig.savefig("out2/Figure3.png", dpi=300, bbox_inches="tight")
print("wrote out2/Figure3.tif and out2/Figure3.png")
