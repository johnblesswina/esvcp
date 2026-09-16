# ESVCP — Entropy-Adaptive Multilayer LSB Embedding for Semantic Visual Cryptography

Reference implementation and reproduction package for the manuscript
*"Entropy-Adaptive Multilayer LSB Embedding for Semantic Visual Cryptography with
Exact Payload Recovery."*

Every numerical value reported in the paper is produced by the scripts in `src/`
and written to `results/` as CSV. Nothing in the paper is hard-coded.

---

## Quick verification

The central claim is that the embedded payload is recovered exactly from the
stego-image alone, without access to the cover. Four stego-images are deposited
in `data/stego/` so this can be checked independently:

```bash
pip install -r requirements.txt
python src/verify.py
```

Expected output:

```
 BOSSBase      bits     bpp        BER  PSNR (dB)     SSIM
----------------------------------------------------------
        1   131,072   0.500   0.000000      48.83   0.9973
        3   131,072   0.500   0.000000      48.83   0.9973
       20   102,249   0.390   0.000000      49.92   0.9965
       34   131,072   0.500   0.000000      48.84   0.9967
----------------------------------------------------------
PASS — payload recovered exactly from every stego-image,
       using only the stego-image and the shared key.
```

`verify.py` never opens a cover image except to recompute PSNR and SSIM for
display. The extraction index is rebuilt from the stego-image.

---

## Reproducing the published results

```bash
# download BOSSBase v1.01 and point the suite at it
python src/experiments.py --images /path/to/BOSSbase_1.01 --limit 50
```

Published values were computed over a 50-image subset. Omitting `--images` uses
the four covers in `data/covers/` and will give different statistics.

| Script writes | Reproduces |
|---|---|
| `results/payload_sweep.csv` | Table 6, Table 7, Abstract PSNR / SSIM / BER |
| `results/capacity.csv` | Section 6.2, Table 9 |
| `results/ablation.csv` | Table 12 |
| `results/shares.csv` | Table 10, Section 6.4 |
| `results/threshold_sweep.csv` | Section 7.6 |
| `results/runtime.csv` | Table 13 |

Figures:

```bash
python src/figure2.py     # architecture diagram
python src/figure4.py     # visual quality panels
```

---

## Parameters

| Parameter | Value | Where |
|---|---|---|
| Logistic map seed `x0` | 0.4567 | `KEY["mlci"]` |
| Logistic map parameter `mu` | 3.9999 | `esvcp.MU` |
| Warm-up iterations | 100 | `logistic_map_perm` |
| Entropy threshold `H` | 0.20 | `esvcp.H_THRESH` |
| Entropy window | 7 x 7 | `entropy_map` |
| Plane allocation | 60 / 30 / 10 % | `_slices` |
| Saliency threshold | Otsu (computed) | `saliency_mask` |
| Payload RNG seed | 99 (experiments), 12345 (verify) | see scripts |

Runtime figures are machine-dependent; `results/runtime.csv` records whatever
the host produces.

---

## Method

**AEDQ** computes local entropy on bit-planes B3–B7 and admits only pixels above
a fixed threshold. Because those planes are never written by embedding, the
receiver derives an identical eligible set from the stego-image alone. This is
what makes cover-free extraction possible.

**MLCI** generates a logistic-map permutation, filters it to the eligible set,
and allocates the payload across B0, B1 and B2 in a 60/30/10 % split. The three
index sets are disjoint slices of one sequence, so each eligible pixel carries at
most one bit — plane allocation distributes the payload, it does not multiply
capacity.

**SASG** partitions the cover by spectral-residual saliency under an Otsu
threshold, forms shares as `V2 = V1 XOR S`, and carries each share in the LSB
plane of a grayscale carrier over the non-salient region. Salient pixels remain
bit-identical to the cover.

---

## Changes from the previous release

The version of this repository accompanying the original submission contained
defects that affected reported results. They are documented here rather than
silently overwritten.

1. **Extraction index mismatch.** `esvcp_extract.m` regenerated the permutation
   without applying the entropy filter used by `mlci_embed.m`, so the two
   routines addressed different pixels. Measured BER was 0.4996. Both now use the
   same filtered index.

2. **Eligibility not reconstructible by the receiver.** Eligibility was computed
   from the cover, which the receiver does not hold. Recomputing it from the
   stego agreed on 99.91 % of pixels, enough to desynchronise the whole permuted
   sequence. Eligibility is now computed on B3–B7 only.

3. **Error diffusion destroyed the payload.** `aedq_embed` ran after
   `mlci_embed` and added diffused error to pixel values, overwriting the
   embedded bits. Measured effect: BER 0.0000 to ~0.49, PSNR down 0.2 dB. Removed
   from the embedding path.

4. **Capacity overstated.** Reported as 3.82 bpp. Measured ceiling is 0.522 bpp
   (sd 0.202). The three planes do not multiply capacity.

5. **U-2-Net was never executed.** The `try` block in `sasg_shares.m` raised
   unconditionally, so the spectral-residual fallback always ran. All saliency
   results in the original submission were spectral residual.

6. **Share construction was self-contradictory.** Requiring `V1 XOR V2 = S`
   while setting `V1 = V2` inside the salient region forces the XOR to zero
   there; measured share recovery error was 5.00 %. Replaced by the carrier
   construction described above.

7. **Steganalysis results were unsupported.** No detector code existed in this
   repository. Those claims have been withdrawn from the manuscript.

The MATLAB sources from the previous release have been removed. Python is now
the reference implementation.

---

## Data

Covers are from BOSSBase v1.01 (Bas, Filler and Pevny, *Break Our Steganographic
System: The Ins and Outs of Organizing BOSS*, Information Hiding 2011), created
and publicly released as a benchmark for the steganography and steganalysis
research community. The four images used in Figure 4 are IDs 1, 3, 20 and 34.
None contains an identifiable person.

Full dataset: https://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip

---

## Citation

See `CITATION.cff`. Archived at Zenodo: DOI [INSERT DOI]

## License

MIT — see `LICENSE`.
