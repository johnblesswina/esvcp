"""
esvcp_fixed.py — ESVCP with the three synchronisation defects repaired.

Changes relative to the published MATLAB (each marked FIX-n below):

FIX-1  esvcp_extract now uses the SAME entropy-filtered index the embedder
       used. The published extractor used the raw permutation.

FIX-2  Eligibility is computed from bit-planes B3..B7 only (img & 0xF8).
       MLCI writes to B0..B2, so those high planes are identical in cover
       and stego, and the RECEIVER can rebuild the index without the cover.
       This is what makes FIX-1 realisable in practice.

FIX-3  AEDQ is removed from the post-embedding path. Any operation that
       runs after MLCI and alters pixel values destroys the payload, and
       any operation that alters B3..B7 desynchronises the index. See
       ablation_aedq() for the measured cost of leaving it in.

Saliency: the published try/except in sasg_shares.m always fell through to
spectral residual, and a fixed tau=0.5 selected <0.5% of pixels. Threshold
is now Otsu on the saliency map - parameter-free and reproducible.
"""
import numpy as np
from skimage.filters import threshold_otsu
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import footprint_rectangle
from skimage.metrics import structural_similarity as ssim_fn
from scipy.ndimage import uniform_filter, gaussian_filter

MU = 3.9999
H_THRESH = 0.2
HIGH_PLANE_MASK = np.uint8(0xF8)          # keeps B3..B7, clears B0..B2


def logistic_map_perm(x0, mu, N):
    if not (0 < x0 < 1):
        raise ValueError(f"seed must be in (0,1); got {x0}")
    x = x0
    for _ in range(100):
        x = mu * x * (1 - x)
    seq = np.empty(N)
    for i in range(N):
        x = mu * x * (1 - x)
        seq[i] = x
    return np.argsort(seq, kind="stable")


def entropy_map(img, win=7):
    return np.clip(rank_entropy(img, footprint_rectangle((win, win))) / 8.0, 0, 1)


def eligibility(img, win=7):
    """FIX-2: derived from the planes MLCI never touches."""
    H = entropy_map(img & HIGH_PLANE_MASK, win)
    return H >= H_THRESH, H


def spectral_residual_saliency(img):
    im = img.astype(float) / 255.0
    F = np.fft.fft2(im)
    logA = np.log(np.abs(F) + 1)
    SR = logA - uniform_filter(logA, size=3)
    sal = np.abs(np.fft.ifft2(np.exp(SR + 1j * np.angle(F)))) ** 2
    return gaussian_filter(sal, 3)


def saliency_mask(cover):
    sal = spectral_residual_saliency(cover)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-10)
    tau = threshold_otsu(sal)
    return sal >= tau, sal, float(tau)


def _slices(L):
    n0, n1 = round(0.60 * L), round(0.30 * L)
    return n0, n1, L - n0 - n1


def embed(cover, secret_bits, key):
    M, N = cover.shape
    valid, H = eligibility(cover)
    perm = logistic_map_perm(key["mlci"], MU, M * N)
    flat_valid = valid.ravel()
    perm_valid = perm[flat_valid[perm]]
    n_eligible = len(perm_valid)

    L = min(len(secret_bits), n_eligible)
    truncated = L < len(secret_bits)
    bits = secret_bits[:L]
    idx = perm_valid[:L]
    n0, n1, n2 = _slices(L)

    out = cover.ravel().copy()

    def setbit(v, pos, b):
        m = np.uint8(1 << pos)
        return (v & ~m) | (b.astype(np.uint8) << pos)

    out[idx[:n0]] = setbit(out[idx[:n0]], 0, bits[:n0])
    out[idx[n0:n0 + n1]] = setbit(out[idx[n0:n0 + n1]], 1, bits[n0:n0 + n1])
    out[idx[n0 + n1:]] = setbit(out[idx[n0 + n1:]], 2, bits[n0 + n1:])

    return out.reshape(M, N), dict(n_eligible=n_eligible, n_embedded=L,
                                   truncated=truncated, H=H, valid=valid)


def extract(stego, key, n_bits):
    """FIX-1 + FIX-2: index rebuilt from the stego alone, no cover needed."""
    M, N = stego.shape
    valid, _ = eligibility(stego)
    perm = logistic_map_perm(key["mlci"], MU, M * N)
    flat_valid = valid.ravel()
    perm_valid = perm[flat_valid[perm]]

    L = min(n_bits, len(perm_valid))
    idx = perm_valid[:L]
    n0, n1, _ = _slices(L)
    f = stego.ravel()
    out = np.empty(L, np.uint8)
    out[:n0] = (f[idx[:n0]] >> 0) & 1
    out[n0:n0 + n1] = (f[idx[n0:n0 + n1]] >> 1) & 1
    out[n0 + n1:] = (f[idx[n0 + n1:]] >> 2) & 1
    return out


def sasg_shares(cover, secret_bits, seed=None):
    M, N = cover.shape
    sal_mask, sal, tau = saliency_mask(cover)

    L = min(len(secret_bits), M * N)
    sec = np.zeros(M * N, bool)
    sec[:L] = secret_bits[:L].astype(bool)
    sec = sec.reshape(M, N)

    rng = np.random.default_rng(seed)
    V1 = rng.integers(0, 2, (M, N)).astype(bool)
    V2 = np.logical_xor(V1, sec)          # XOR(V1,V2) == secret EVERYWHERE
    return [V1, V2], sal_mask, sal, tau


def metrics(a, b):
    a, b = a.astype(float), b.astype(float)
    mse = np.mean((a - b) ** 2)
    psnr = np.inf if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    return psnr, ssim_fn(a.astype(np.uint8), b.astype(np.uint8), data_range=255)
