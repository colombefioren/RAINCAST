"""Does the context calibration carry over to other models? The 32 members of blend32cw2 are split into two
16-member blends A and B (alternate members). Per lead, on alternate validation days: isotonic table and context
model fitted on blend A's probabilities of the fitting days, applied to blend B's probabilities of the scoring days
(cross-model), against the same fitted on blend B itself (same-model), both against the isotonic table alone fitted
on B. A loss of the gain across models would explain the public miss of submissions 33 and 34 (the test side is the
full-data twins, other models than the validation members). Usage (from src/): python context_transfer.py ../outputs/blend32cw2dec.json"""
import os, sys, json, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fss import fss_pooled, load_prob
from fss_decode import fit_calibration, calibrate, decode
import context_calib
torch.set_num_threads(6)
R = os.path.dirname(os.path.abspath(__file__)) + "/../"
cfg = json.load(open(sys.argv[1])); files = [f.replace("../", R) for f in cfg["probs"]]
w = np.asarray(cfg.get("weights") or [1.0] * len(files), np.float64)
def blend(idx):
    P = None; ws = w[idx] / w[idx].sum()
    for wi, i in zip(ws, idx):
        p = load_prob(files[i]).astype(np.float32); P = wi * p if P is None else P + wi * p
    return P
A, B = blend(np.arange(0, len(files), 2)), blend(np.arange(1, len(files), 2))
tr = np.load(f"{R}data/train.npz"); ts = pd.to_datetime(tr["times"]); val = np.asarray((ts >= "2022-12-01") & (ts < "2023-04-01"))
Xv = tr["X"][val]; now, prev = Xv[:, 2].copy(), Xv[:, 1].copy(); del Xv, tr
yv = np.load(f"{R}data/yval.npy") > 1.0; tv = ts[val]; fold = tv.dayofyear.values % 2
MU = cfg["decode"]["mu"]
for k in range(3):
    hrs = (tv.hour.values + k + 1) % 24; obs = yv[:, k]; res = {"isotonic B": 0, "context same-model (B)": 0, "context cross-model (A to B)": 0}
    dec = {n: np.zeros_like(obs) for n in res}
    for f in (0, 1):
        fit, sc = np.where(fold != f)[0], np.where(fold == f)[0]
        tB = fit_calibration(B[fit, k], obs[fit]); tA = fit_calibration(A[fit, k], obs[fit])
        dec["isotonic B"][sc] = decode(calibrate(B[sc, k], tB), MU[k])
        cB = context_calib.fit(B[fit, k], obs[fit], tB, now[fit], prev[fit], hrs[fit])
        dec["context same-model (B)"][sc] = decode(context_calib.apply(B[sc, k], tB, cB, now[sc], prev[sc], hrs[sc]), MU[k])
        cA = context_calib.fit(A[fit, k], obs[fit], tA, now[fit], prev[fit], hrs[fit])
        dec["context cross-model (A to B)"][sc] = decode(context_calib.apply(B[sc, k], tA, cA, now[sc], prev[sc], hrs[sc]), MU[k])
    o = (obs[:, None] * 10.0).astype(np.float32)
    print(f"lead {k+1}: " + "  ".join(f"{n} {fss_pooled(m[:, None] * 10.0, o)[0]:.5f}" for n, m in dec.items()), flush=True)
