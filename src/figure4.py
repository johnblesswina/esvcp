import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image

IDS = ["1", "3", "20", "34"]
LBL = ["BOSSBase 1", "BOSSBase 20", "BOSSBase 3", "BOSSBase 34"]
m = {r["image"]: r for r in json.load(open("../results/measurements.json"))}

cov = [np.array(Image.open(f"../data/covers/{i}.pgm")) for i in IDS]
stg = [np.array(Image.open(f"../results/{i}_stego.pgm")) for i in IDS]
Hm = [np.load(f"../results/{i}_H.npy") for i in IDS]
Sm = [np.load(f"../results/{i}_sal.npy") for i in IDS]
res = [np.clip(np.abs(c.astype(float) - s.astype(float)) * 10, 0, 255)
       for c, s in zip(cov, stg)]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig = plt.figure(figsize=(7.09, 9.45), dpi=300)
gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.60, wspace=0.08,
                       left=0.075, right=0.995, top=0.945, bottom=0.015)

rowhead = [
    "(a)  Cover images $I_c$  —  BOSSBase v1.01",
    "(b)  ESVCP stego-images $I_s$  —  payload fully recovered (BER = 0)",
    r"(c)  Residual maps $|I_c-I_s|\times10$",
    r"(d)  Local entropy $\tilde{H}(i,j)$  —  AEDQ priority",
    r"(e)  Saliency masks $M_{sal}$  —  spectral residual, Otsu threshold",
]

for r in range(5):
    for c in range(4):
        ax = fig.add_subplot(gs[r, c])
        d = m[IDS[c]]
        if r == 0:
            ax.imshow(cov[c], cmap="gray", vmin=0, vmax=255)
            sub = LBL[c]
        elif r == 1:
            ax.imshow(stg[c], cmap="gray", vmin=0, vmax=255)
            sub = (f"{d['psnr']:.2f} dB / SSIM {d['ssim']:.4f}\n"
                   f"{d['eff_bpp']:.3f} bpp, BER = 0")
        elif r == 2:
            ax.imshow(res[c], cmap="gray", vmin=0, vmax=255)
            sub = f"mean |err| = {d['mae']:.3f} GL"
        elif r == 3:
            ax.imshow(Hm[c], cmap="inferno", vmin=0, vmax=1)
            sub = rf"$\tilde{{H}}_{{mean}}$ = {d['H_mean']:.2f}"
        else:
            ax.imshow(Sm[c], cmap="gray", vmin=0, vmax=1)
            sub = rf"$\Omega_{{sal}}$ = {d['sal_pct']:.2f}%"
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_color("#444444")
        ax.set_xlabel(sub, fontsize=7.0, labelpad=2.5)
        if c == 0:
            ax.text(-0.045, 1.05, rowhead[r], transform=ax.transAxes,
                    fontsize=8.6, fontweight="bold", ha="left", va="bottom")

fig.savefig("../results/Figure4.tif", dpi=600, bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"})
fig.savefig("../results/Figure4.png", dpi=300, bbox_inches="tight")

with open("../results/Figure4_values.csv", "w") as f:
    f.write("image,psnr_dB,ssim,mae_greylevels,entropy_mean,omega_sal_pct,"
            "bits_embedded,effective_bpp,eligible_pixels,max_bpp,ber\n")
    for i in IDS:
        d = m[i]
        f.write(f"{i},{d['psnr']:.4f},{d['ssim']:.6f},{d['mae']:.4f},"
                f"{d['H_mean']:.4f},{d['sal_pct']:.2f},{d['embedded']},"
                f"{d['eff_bpp']:.4f},{d['eligible']},{d['max_bpp']:.4f},{d['ber']:.6f}\n")

print("mean PSNR %.2f dB | mean SSIM %.4f | mean max capacity %.3f bpp" % (
    np.mean([m[i]["psnr"] for i in IDS]),
    np.mean([m[i]["ssim"] for i in IDS]),
    np.mean([m[i]["max_bpp"] for i in IDS])))
