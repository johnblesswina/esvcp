#!/usr/bin/env python3
"""
verify.py — independent verification of the ESVCP recovery claim.

Run this first. It needs nothing but the files in this repository.

    python src/verify.py

For each deposited stego-image it:
  1. reconstructs the embedding index from the STEGO-IMAGE ALONE
     (no access to the cover is used at any point),
  2. extracts the payload,
  3. compares it against the known secret bitstream,
  4. independently recomputes PSNR and SSIM against the cover.

Expected output: BER = 0.000000 on all four images.
"""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import esvcp as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDS = ["1", "3", "20", "34"]
KEY = {"mlci": 0.4567}          # published seed, see README
SECRET_SEED = 12345             # rng seed for the payload, see README
BPP = 0.5


def main():
    print(__doc__.split("Expected")[0].strip())
    print()
    print(f"{'BOSSBase':>9} {'bits':>9} {'bpp':>7} {'BER':>10} "
          f"{'PSNR (dB)':>10} {'SSIM':>8}")
    print("-" * 58)

    failures = 0
    for name in IDS:
        stego = np.array(Image.open(f"{ROOT}/data/stego/{name}_stego.pgm"))
        cover = np.array(Image.open(f"{ROOT}/data/covers/{name}.pgm"))

        # how many bits were embedded: capped by the eligible set
        valid, _ = E.eligibility(stego)          # from the STEGO only
        n_eligible = int(valid.sum())
        n_bits = min(int(round(BPP * stego.size)), n_eligible)

        recovered = E.extract(stego, KEY, n_bits)

        expected = np.random.default_rng(SECRET_SEED).integers(
            0, 2, int(round(BPP * stego.size))).astype(np.uint8)[:n_bits]

        ber = float(np.mean(recovered != expected))
        psnr, ssim = E.metrics(cover, stego)

        flag = "" if ber == 0.0 else "   <-- MISMATCH"
        failures += (ber != 0.0)
        print(f"{name:>9} {n_bits:>9,} {n_bits/stego.size:>7.3f} "
              f"{ber:>10.6f} {psnr:>10.2f} {ssim:>8.4f}{flag}")

    print("-" * 58)
    if failures == 0:
        print("PASS — payload recovered exactly from every stego-image,")
        print("       using only the stego-image and the shared key.")
    else:
        print(f"FAIL — {failures} image(s) did not recover exactly.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
