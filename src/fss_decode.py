"""Turn rain probabilities into the binary mask that maximises the expected FSS.

Why: FSS = 2 * sum(f * o) / (sum(f^2) + sum(o^2)), where f and o are the 3x3 neighbourhood
fractions of the forecast mask and of the observed mask. A per-pixel threshold ignores the
neighbourhood: where the model says "40 % chance of rain" over a wide area, thresholding at 0.5
predicts all dry, while the metric would reward a mask covering about 40 % of the 3x3 windows.

How: with calibrated probabilities p, the expected observed fraction is box(p), so the expected
numerator is linear in the mask M: sum_y M(y) * q(y) with q = box(box(p)). The denominator grows
with sum(box(M)^2), which is larger for clustered masks. For a trade-off mu (tuned per lead on the
validation season) we maximise  sum(M * q) - mu * sum(box(M)^2)  by iterated conditional modes:
pixels whose 5x5 neighbourhoods cannot interact (same position on a 3x3 grid) are updated
together, each set to the better of rain / no rain given its neighbours. Each update can only
raise the objective. Only the sample's own probability map is used (rule 2.4).

Calibration: probabilities from a BCE loss with a positive class weight are inflated, so they are
first mapped through an isotonic (monotone) fit of observed rain frequency on the validation season.
"""
import numpy as np, torch, torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

def box(t):
    return F.avg_pool2d(t, 3, stride=1, padding=1)  # zero padding, divides by 9, same as the metric

def fit_calibration(p, obs, bins=400):
    """Monotone map from model probability to observed rain frequency, returned as a 1001-entry table."""
    idx = np.minimum((p.ravel() * bins).astype(np.int64), bins - 1)
    cnt = np.bincount(idx, minlength=bins).astype(np.float64)
    hit = np.bincount(idx, weights=obs.ravel().astype(np.float64), minlength=bins)
    keep = cnt > 0
    centers = (np.arange(bins) + 0.5) / bins
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(centers[keep], hit[keep] / cnt[keep], sample_weight=cnt[keep])
    return iso.predict(np.linspace(0, 1, 1001)).astype(np.float32)

def calibrate(p, table):
    return table[np.clip(np.rint(p * 1000), 0, 1000).astype(np.int64)]

def calibrate_wet(p, tables, edges):
    """p: (N, H, W). Calibrates each sample with the table of its mean-probability quartile (dry to wet situations);
    the quartile depends on the sample's own map only."""
    bins = np.digitize(p.mean(axis=(1, 2)), edges)
    out = np.empty(p.shape, np.float32)
    for j, t in enumerate(tables):
        if (bins == j).any(): out[bins == j] = calibrate(p[bins == j], np.asarray(t, np.float32))
    return out

def decode(p, mu, sweeps=6, init_thr=0.5, chunk=512, init=None):
    """p: calibrated probabilities (N, H, W). Returns a boolean mask (N, H, W). ICM starts from p > init_thr, or
    from the boolean masks init (N, H, W) when given."""
    N, H, W = p.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    classes = [torch.from_numpy((yy % 3 == i) & (xx % 3 == j))[None, None] for i in range(3) for j in range(3)]
    nb = box(torch.ones(1, 1, H, W)) * 9  # in-domain neighbours of each pixel (9 inside, 6 edge, 4 corner)
    out = np.zeros((N, H, W), bool)
    for s in range(0, N, chunk):
        pt = torch.from_numpy(np.ascontiguousarray(p[s:s + chunk], dtype=np.float32))[:, None]
        q = box(box(pt))
        M = (pt > init_thr).float() if init is None else torch.from_numpy(init[s:s + chunk].astype(np.float32))[:, None]
        for _ in range(sweeps):
            for c in classes:
                g = box(box(M)) - M * nb / 81  # neighbourhood sum without the pixel itself
                on = (q - mu * (2 * g + nb / 81) > 0).float()
                M = torch.where(c, on, M)
        out[s:s + chunk] = M[:, 0].numpy() > 0.5
    return out

def objective(M, p, mu, chunk=512):
    """Per-sample value of the decoder's objective sum(M q) - mu sum(box(M)^2) for boolean masks M (N, H, W)."""
    out = np.zeros(len(M))
    for s in range(0, len(M), chunk):
        Mt = torch.from_numpy(M[s:s + chunk].astype(np.float32))[:, None]
        pt = torch.from_numpy(np.ascontiguousarray(p[s:s + chunk], dtype=np.float32))[:, None]
        q = box(box(pt))
        out[s:s + chunk] = ((Mt * q).sum((1, 2, 3)) - mu * (box(Mt) ** 2).sum((1, 2, 3))).numpy()
    return out

def decode_best(p, mu, inits=(0.5, 0.65), anneal=True, sweeps=6):
    """decode() from several starts, keeping for each sample the mask with the highest objective: thresholds of p
    at each of inits, and (anneal) ICM at 0.7 mu continued at mu. ICM finds local optima, and the objective tracks
    the pooled FSS closely, so the extra starts gain a few 1e-4 of FSS per lead for three times the decoding time."""
    masks = [decode(p, mu, sweeps, thr) for thr in inits]
    if anneal: masks.append(decode(p, mu, sweeps + 2, init=decode(p, 0.7 * mu, 4)))
    J = np.stack([objective(M, p, mu) for M in masks])
    return np.stack(masks)[J.argmax(0), np.arange(len(p))]
