"""Fractions Skill Score, as used by the competition: 1 mm/h threshold, 3x3 neighbourhood,
mean over the three lead times. Two aggregations are provided because the host's
exact pooling is not published: pooled over all samples, and per-sample mean."""
import numpy as np
from scipy.ndimage import uniform_filter

THR = 1.0
WIN = 3


def fractions(binary, win=WIN):
    # binary: (N, H, W) -> neighbourhood fraction of rainy pixels, zero padded at edges
    return uniform_filter(binary.astype(np.float32), size=(1, win, win), mode="constant", cval=0.0)


def fss_components(pred, obs, thr=THR, win=WIN):
    pf = fractions(pred > thr, win)
    po = fractions(obs > thr, win)
    num = ((pf - po) ** 2).sum(axis=(1, 2))
    den = (pf ** 2).sum(axis=(1, 2)) + (po ** 2).sum(axis=(1, 2))
    return num, den  # per sample


def fss_pooled(pred, obs, thr=THR, win=WIN):
    """pred, obs: (N, 3, H, W). FSS per lead time pooled over samples, then mean of 3."""
    scores = []
    for k in range(obs.shape[1]):
        num, den = fss_components(pred[:, k], obs[:, k], thr, win)
        scores.append(1.0 - num.sum() / max(den.sum(), 1e-12))
    return float(np.mean(scores)), scores


def fss_per_sample(pred, obs, thr=THR, win=WIN, empty=1.0):
    """Mean over samples of per-sample FSS; samples with no rain in either field score `empty`."""
    scores = []
    for k in range(obs.shape[1]):
        num, den = fss_components(pred[:, k], obs[:, k], thr, win)
        s = np.where(den > 0, 1.0 - num / np.maximum(den, 1e-12), empty)
        scores.append(float(s.mean()))
    return float(np.mean(scores)), scores


def report(name, pred, obs):
    p, pk = fss_pooled(pred, obs)
    s, sk = fss_per_sample(pred, obs)
    print(f"{name:28s} pooled={p:.4f} [{pk[0]:.3f} {pk[1]:.3f} {pk[2]:.3f}]  "
          f"per-sample={s:.4f} [{sk[0]:.3f} {sk[1]:.3f} {sk[2]:.3f}]", flush=True)
    return p
