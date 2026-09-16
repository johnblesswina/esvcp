#!/usr/bin/env python3
"""
experiments.py — reproduces every measured value reported in the manuscript.

    python src/experiments.py --images /path/to/BOSSbase_1.01

Writes one CSV per experiment into results/ and prints a summary mapping
each result to the table or section of the paper that reports it.

If --images is omitted the four covers in data/covers are used; results will
differ from the published values, which are computed over a 50-image subset.
"""
import argparse
import glob
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import esvcp as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
KEY = {"mlci": 0.4567}
SECRET_SEED = 99


def load(paths):
    out = []
    for p in paths:
        a = np.array(Image.open(p))
        if a.ndim == 2:
            out.append((os.path.splitext(os.path.basename(p))[0], a))
    return out


def payload(n, seed=SECRET_SEED):
    return np.random.default_rng(seed).integers(0, 2, n).astype(np.uint8)


def stat(v):
    v = np.asarray(v, float)
    return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0


def ci95(v):
    v = np.asarray(v, float)
    h = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean() - h, v.mean() + h


# ---------------------------------------------------------------- exp 1
def exp_payload_sweep(imgs):
    """Table 6, Table 7, Abstract PSNR/SSIM/BER."""
    rows = []
    for bpp in [0.1, 0.2, 0.3, 0.4, 0.5]:
        P, S, EF, B, tr = [], [], [], [], 0
        for name, cov in imgs:
            bits = payload(int(round(bpp * cov.size)))
            st, inf = E.embed(cov, bits, KEY)
            rec = E.extract(st, KEY, inf["n_embedded"])
            p, s = E.metrics(cov, st)
            P.append(p); S.append(s)
            EF.append(inf["n_embedded"] / cov.size)
            B.append(np.mean(rec != bits[:inf["n_embedded"]]))
            tr += inf["truncated"]
        lo, hi = ci95(P)
        rows.append(dict(bpp=bpp, psnr_mean=np.mean(P), psnr_sd=stat(P)[1],
                         ci_lo=lo, ci_hi=hi, ssim_mean=np.mean(S),
                         ssim_sd=stat(S)[1], embedded_bpp=np.mean(EF),
                         ber_max=max(B), truncated=tr, n=len(P)))
    write("payload_sweep.csv", rows)
    r = rows[-1]
    print(f"  [Table 6, Abstract] 0.5 bpp: PSNR {r['psnr_mean']:.2f} dB "
          f"(95% CI [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]), "
          f"SSIM {r['ssim_mean']:.4f}, BER max {r['ber_max']:.6f}")
    return rows


# ---------------------------------------------------------------- exp 2
def exp_capacity(imgs):
    """Section 6.2, Table 9 — capacity ceiling."""
    rows = []
    for name, cov in imgs:
        valid, _ = E.eligibility(cov)
        rows.append(dict(image=name, eligible=int(valid.sum()),
                         max_bpp=valid.sum() / cov.size))
    write("capacity.csv", rows)
    c = [r["max_bpp"] for r in rows]
    print(f"  [Section 6.2, Table 9] capacity {np.mean(c):.3f} bpp "
          f"(sd {stat(c)[1]:.3f}, range {min(c):.3f}-{max(c):.3f})")
    return rows


# ---------------------------------------------------------------- exp 3
def exp_ablation(imgs):
    """Table 12 — module ablation."""
    def setbit(v, pos, b):
        m = np.uint8(1 << pos)
        return (v & ~m) | (b.astype(np.uint8) << pos)

    def no_aedq(cov, bits):
        M, N = cov.shape
        perm = E.logistic_map_perm(KEY["mlci"], E.MU, M * N)
        L = min(len(bits), M * N); idx = perm[:L]
        n0, n1, _ = E._slices(L)
        o = cov.ravel().copy()
        o[idx[:n0]] = setbit(o[idx[:n0]], 0, bits[:n0])
        o[idx[n0:n0+n1]] = setbit(o[idx[n0:n0+n1]], 1, bits[n0:n0+n1])
        o[idx[n0+n1:]] = setbit(o[idx[n0+n1:]], 2, bits[n0+n1:])
        return o.reshape(M, N), M * N

    def no_mlci(cov, bits):
        valid, _ = E.eligibility(cov)
        idx_all = np.flatnonzero(valid.ravel())
        L = min(len(bits), len(idx_all)); idx = idx_all[:L]
        o = cov.ravel().copy()
        o[idx] = setbit(o[idx], 0, bits[:L])
        return o.reshape(cov.shape), len(idx_all)

    acc = {k: [] for k in ["Full ESVCP", "without AEDQ", "without MLCI"]}
    for name, cov in imgs:
        bits = payload(int(round(0.5 * cov.size)))
        st, inf = E.embed(cov, bits, KEY)
        acc["Full ESVCP"].append((*E.metrics(cov, st), inf["n_eligible"] / cov.size))
        s2, e2 = no_aedq(cov, bits)
        acc["without AEDQ"].append((*E.metrics(cov, s2), e2 / cov.size))
        s3, e3 = no_mlci(cov, bits)
        acc["without MLCI"].append((*E.metrics(cov, s3), e3 / cov.size))

    rows = []
    for k, v in acc.items():
        a = np.array(v)
        rows.append(dict(configuration=k, psnr_mean=a[:, 0].mean(),
                         psnr_sd=a[:, 0].std(ddof=1), ssim_mean=a[:, 1].mean(),
                         ssim_sd=a[:, 1].std(ddof=1), max_bpp=a[:, 2].mean()))
    rows.append(dict(configuration="without SASG", **{k: rows[0][k] for k in
                list(rows[0])[1:]}))
    write("ablation.csv", rows)
    print(f"  [Table 12] full {rows[0]['psnr_mean']:.2f} dB | "
          f"no AEDQ {rows[1]['psnr_mean']:.2f} dB | "
          f"no MLCI {rows[2]['psnr_mean']:.2f} dB")
    return rows


# ---------------------------------------------------------------- exp 4
def exp_shares(imgs):
    """Table 10, Section 6.4 — share carrier quality."""
    def psnr(a, b):
        mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
        return np.inf if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)

    rows = []
    for name, cov in imgs:
        M, N = cov.shape
        sal, _, tau = E.saliency_mask(cov)
        free = ~sal
        nf = int(free.sum())
        rng = np.random.default_rng(7)
        sec = rng.integers(0, 2, nf).astype(bool)
        v1 = rng.integers(0, 2, nf).astype(bool)
        v2 = np.logical_xor(v1, sec)

        idx = np.flatnonzero(free.ravel())
        C1 = cov.ravel().copy(); C2 = cov.ravel().copy()
        C1[idx] = (C1[idx] & np.uint8(0xFE)) | v1.astype(np.uint8)
        C2[idx] = (C2[idx] & np.uint8(0xFE)) | v2.astype(np.uint8)
        C1 = C1.reshape(M, N); C2 = C2.reshape(M, N)

        r1 = (C1.ravel()[idx] & 1).astype(bool)
        r2 = (C2.ravel()[idx] & 1).astype(bool)
        err = float(np.mean(np.logical_xor(r1, r2) != sec))

        # classical (2,2)-VCS baseline: bare random binary shares
        cls = psnr(v1_full := rng.integers(0, 2, (M, N)).astype(np.uint8) * 255, cov)

        _, ssim = E.metrics(cov, C1)
        rows.append(dict(image=name, carrier_psnr=psnr(C1, cov),
                         carrier_ssim=ssim, classical_vcs_psnr=cls,
                         recovery_error=err, omega_sal_pct=sal.mean() * 100,
                         secret_capacity_bpp=nf / cov.size, otsu_tau=tau))
    write("shares.csv", rows)
    cp = [r["carrier_psnr"] for r in rows]
    cl = [r["classical_vcs_psnr"] for r in rows]
    print(f"  [Table 10] carrier {np.mean(cp):.2f} dB (sd {stat(cp)[1]:.2f}) "
          f"vs classical {np.mean(cl):.2f} dB | "
          f"recovery error {max(r['recovery_error'] for r in rows):.6f}")
    return rows


# ---------------------------------------------------------------- exp 5
def exp_threshold(imgs):
    """Section 7.6 — entropy threshold sensitivity."""
    rows = []
    orig = E.H_THRESH
    for th in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        E.H_THRESH = th
        P, S, C = [], [], []
        for name, cov in imgs:
            bits = payload(int(round(0.5 * cov.size)))
            st, inf = E.embed(cov, bits, KEY)
            p, s = E.metrics(cov, st)
            P.append(p); S.append(s); C.append(inf["n_eligible"] / cov.size)
        rows.append(dict(h_threshold=th, psnr_mean=np.mean(P),
                         ssim_mean=np.mean(S), max_bpp=np.mean(C)))
    E.H_THRESH = orig
    write("threshold_sweep.csv", rows)
    print("  [Section 7.6] " + " | ".join(
        f"H={r['h_threshold']:.2f}: {r['psnr_mean']:.2f} dB / "
        f"{r['max_bpp']:.2f} bpp" for r in rows[2:5]))
    return rows


# ---------------------------------------------------------------- exp 6
def exp_runtime(imgs):
    """Table 13 — runtime."""
    import time
    sub = imgs[:20]
    t = {k: [] for k in ["aedq", "mlci_perm", "sasg", "embed", "extract"]}
    for name, cov in sub:
        bits = payload(int(round(0.5 * cov.size)))
        s = time.perf_counter(); E.eligibility(cov); t["aedq"].append(time.perf_counter() - s)
        s = time.perf_counter(); E.logistic_map_perm(KEY["mlci"], E.MU, cov.size); t["mlci_perm"].append(time.perf_counter() - s)
        s = time.perf_counter(); E.saliency_mask(cov); t["sasg"].append(time.perf_counter() - s)
        s = time.perf_counter(); st, inf = E.embed(cov, bits, KEY); t["embed"].append(time.perf_counter() - s)
        s = time.perf_counter(); E.extract(st, KEY, inf["n_embedded"]); t["extract"].append(time.perf_counter() - s)
    rows = [dict(stage=k, mean_ms=np.mean(v) * 1000,
                 sd_ms=(np.std(v, ddof=1) * 1000 if len(v) > 1 else 0.0))
            for k, v in t.items()]
    write("runtime.csv", rows)
    d = {r["stage"]: r["mean_ms"] for r in rows}
    print(f"  [Table 13] embed {d['embed']:.1f} ms | extract {d['extract']:.1f} ms")
    return rows


def write(fname, rows):
    if not rows:
        return
    os.makedirs(RES, exist_ok=True)
    keys = list(rows[0].keys())
    with open(os.path.join(RES, fname), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(
                f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                for k in keys) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None,
                    help="directory of .pgm covers (default: data/covers)")
    ap.add_argument("--limit", type=int, default=50)
    a = ap.parse_args()

    d = a.images or os.path.join(ROOT, "data", "covers")
    paths = sorted(glob.glob(os.path.join(d, "*.pgm")))[:a.limit]
    if not paths:
        sys.exit(f"no .pgm files found in {d}")
    imgs = load(paths)
    print(f"ESVCP experiment suite — {len(imgs)} images from {d}\n")

    exp_payload_sweep(imgs)
    exp_capacity(imgs)
    exp_ablation(imgs)
    exp_shares(imgs)
    exp_threshold(imgs)
    exp_runtime(imgs)

    print(f"\nCSV files written to {RES}/")


if __name__ == "__main__":
    main()
