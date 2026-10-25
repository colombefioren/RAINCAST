"""Context calibration for the expected-FSS decoder (src/fss_decode.py).

The isotonic table of fss_decode.py maps each pixel's blend probability to a calibrated one, whatever surrounds
the pixel. The ensemble mean is smooth, so a pixel at the edge of a rain area and a pixel inside one can share a
probability while their outcomes differ. This module refines the calibrated probability with a logistic model on
features of the pixel's neighbourhood and situation, with all pairwise feature products (a quadratic surface):
the calibrated logit, the blend's 3x3, 9x9 and 25x25 mean logits and 3x3 maximum logit, the rain at the forecast
time and one hour before, the 9x9 rain share at the forecast time, the sample's mean blend logit, and the hour of
day of the lead time. Every feature of a sample comes from its own maps and timestamp, so test samples stay
independent of each other.

Fitted per lead time on the validation season (a pixel sample: pixels with some rain chance at rate 0.15, dry
pixels at rate 0.0075 weighted by 20), starting from the isotonic answer. On alternate validation days (fit on one
half, score on the other) it raised the decoded FSS of blend32cw2 by about 0.0010 (0.0010, 0.0013 and 0.0006 per
lead; 0.0007 with only the first seven features)."""
import numpy as np, torch
from scipy.ndimage import uniform_filter, maximum_filter
from fss_decode import calibrate

NF = 11

def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))

def features(c, P, now, prev, hours):
    """c: calibrated probabilities, P: blend probabilities, now / prev: rain maps (mm/h) at the forecast time and
    one hour before, all (n, H, W); hours: (n,) hour of day of the lead time. Returns (n*H*W, NF) float32."""
    ang = 2 * np.pi * np.asarray(hours, np.float32) / 24.0
    per_sample = lambda v: np.broadcast_to(np.asarray(v, np.float32)[:, None, None], c.shape)
    f = [logit(c), logit(uniform_filter(P, size=(1, 3, 3), mode="nearest")), logit(uniform_filter(P, size=(1, 9, 9), mode="nearest")),
         logit(uniform_filter(P, size=(1, 25, 25), mode="nearest")), logit(maximum_filter(P, size=(1, 3, 3), mode="nearest")),
         np.log1p(np.maximum(now, 0)), uniform_filter((now > 1.0).astype(np.float32), size=(1, 9, 9), mode="constant"),
         np.log1p(np.maximum(prev, 0)), per_sample(logit(P.mean(axis=(1, 2)))), per_sample(np.sin(ang)), per_sample(np.cos(ang))]
    return np.stack(f, -1).astype(np.float32).reshape(-1, NF)

def expand(Z):
    """Features plus all their pairwise products."""
    return torch.cat([Z, (Z[:, :, None] * Z[:, None, :]).flatten(1)], 1)

def fit(P, obs, table, now, prev, hours, chunk=100, seed=0):
    """P: (n, H, W) blend probabilities of one lead time, obs: (n, H, W) booleans (rain > 1 mm/h), table: its
    isotonic table. Returns the model as a dict of lists (JSON-ready)."""
    rng = np.random.default_rng(seed)
    Xs, Ys, Ws = [], [], []
    for s in range(0, len(P), chunk):
        r = slice(s, s + chunk); c = calibrate(P[r], table)
        F = features(c, P[r], now[r], prev[r], hours[r]); o = obs[r].reshape(-1); cc = c.reshape(-1)
        keep = np.where(cc > 0.01, rng.random(len(cc)) < 0.15, rng.random(len(cc)) < 0.0075)
        Xs.append(F[keep]); Ys.append(o[keep]); Ws.append(np.where(cc[keep] > 0.01, 1.0, 20.0))
    X = torch.from_numpy(np.concatenate(Xs)); Y = torch.from_numpy(np.concatenate(Ys).astype(np.float32))
    W = torch.from_numpy(np.concatenate(Ws).astype(np.float32)); del Xs, Ys, Ws
    mean, std = X.mean(0), X.std(0) + 1e-6
    Xe = expand((X - mean) / std); del X
    model = torch.nn.Linear(NF + NF * NF, 1)
    with torch.no_grad():  # start from the isotonic answer: the calibrated logit unchanged
        model.weight.zero_(); model.weight[0, 0] = std[0]; model.bias.fill_(float(mean[0]))
    opt = torch.optim.LBFGS(model.parameters(), max_iter=200, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(model(Xe)[:, 0], Y, reduction="none") * W).mean() \
               + 1e-5 * model.weight.pow(2).sum()
        loss.backward(); return loss
    opt.step(closure)
    return {"mean": mean.tolist(), "std": std.tolist(), "weight": model.weight[0].tolist(), "bias": float(model.bias.detach())}

def apply(P, table, model, now, prev, hours, chunk=100):
    """Context-calibrated probabilities (n, H, W) of blend probabilities P (n, H, W) of one lead time."""
    mean, std = torch.tensor(model["mean"]), torch.tensor(model["std"])
    wt, b = torch.tensor(model["weight"]), float(model["bias"])
    out = np.zeros(P.shape, np.float32)
    for s in range(0, len(P), chunk):
        r = slice(s, s + chunk); c = calibrate(P[r], table)
        with torch.no_grad():
            z = expand((torch.from_numpy(features(c, P[r], now[r], prev[r], hours[r])) - mean) / std) @ wt + b
        out[r] = torch.sigmoid(z).numpy().reshape(c.shape)
    return out
