"""Evaluate blends of saved validation probability maps and tune per-lead thresholds.
Usage: python src/blend.py --data data --out outputs/blend.json [--dilate 0] prob1_valprob.npy prob2_valprob.npy ...
Each *_valprob.npy must have a sibling *_testprob.npy; the blended test prediction is written next to --out."""
import argparse, json, numpy as np, pandas as pd
from scipy.ndimage import uniform_filter
from fss import fss_pooled, report

p = argparse.ArgumentParser()
p.add_argument("--data", required=True); p.add_argument("--out", required=True)
p.add_argument("--smooth", type=int, default=0, help="average probabilities over k x k window before thresholding")
p.add_argument("--dilate", type=int, default=0, help="1: also sweep a 3x3 dilation per lead")
p.add_argument("--decode", type=int, default=0, help="1: also fit calibration and tune the expected-FSS decoder (src/fss_decode.py) per lead")
p.add_argument("--mu", default="", help="comma-separated decoder mu per lead; skips the mu search (faster)")
p.add_argument("--weights", default="", help="comma-separated blend weights, default equal")
p.add_argument("probs", nargs="+")
a = p.parse_args()

import os
YV = f"{a.data}/yval.npy"  # cached validation targets (Dec 2022 - Mar 2023), avoids reloading 4 GB each time
if os.path.exists(YV):
    yv = np.load(YV)
else:
    tr = np.load(f"{a.data}/train.npz"); ts = pd.to_datetime(tr["times"])
    val = np.asarray((ts >= "2022-12-01") & (ts < "2023-04-01")); yv = (tr["y"][val] > 1.0).astype(np.float16) * 10
    np.save(YV, yv)
w = np.array([float(x) for x in a.weights.split(",")]) if a.weights else np.ones(len(a.probs))
w = w / w.sum()
for f in a.probs:
    pv = np.load(f).astype(np.float32); report(f.split("/")[-1][:28], (pv > 0.5) * 10.0, yv)
PV = sum(wi * np.load(f).astype(np.float32) for wi, f in zip(w, a.probs))
PT = sum(wi * np.load(f.replace("_valprob", "_testprob")).astype(np.float32) for wi, f in zip(w, a.probs))
if a.smooth > 1:
    PV = uniform_filter(PV, size=(1, 1, a.smooth, a.smooth), mode="nearest")
    PT = uniform_filter(PT, size=(1, 1, a.smooth, a.smooth), mode="nearest")
thr, best = [], []
for k in range(3):
    cands = np.arange(0.15, 0.76, 0.025)
    sc = [fss_pooled((PV[:, k:k+1] > t) * 10.0, yv[:, k:k+1])[0] for t in cands]
    thr.append(float(cands[int(np.argmax(sc))])); best.append(float(max(sc)))
    print(f"lead {k+1}: thr {thr[-1]:.3f} fss {best[-1]:.4f}")
pred = np.stack([(PV[:, k] > thr[k]) * 10.0 for k in range(3)], 1)
score = report("blend tuned", pred, yv)
from scipy.ndimage import maximum_filter
dil = [1, 1, 1]
if a.dilate:
    # per lead: re-tune threshold jointly with a k x k max-pool dilation of the binary mask
    for k in range(3):
        best_s, best_t, best_d = -1, thr[k], 1
        for d in (1, 3):
            for t in np.arange(0.3, 0.9, 0.025):
                m = (PV[:, k] > t).astype(np.float32)
                if d > 1: m = maximum_filter(m, size=(1, d, d))
                sc = fss_pooled(m[:, None] * 10.0, yv[:, k:k+1])[0]
                if sc > best_s: best_s, best_t, best_d = sc, float(t), d
        thr[k], dil[k] = best_t, best_d
        print(f"lead {k+1}: dilate {best_d} thr {best_t:.3f} fss {best_s:.4f}")
    pred = np.stack([maximum_filter((PV[:, k] > thr[k]).astype(np.float32), size=(1, dil[k], dil[k])) * 10.0 if dil[k] > 1 else (PV[:, k] > thr[k]) * 10.0 for k in range(3)], 1)
    score = report("blend tuned+dilate", pred, yv)
cfg = {"thr": thr, "dilate": dil, "val_fss": score, "per_lead": best, "probs": a.probs, "weights": w.tolist(), "smooth": a.smooth}
if a.decode:
    # calibration and mu are fitted on the whole validation season (in-sample); src/eval_decode.py gives the
    # honest out-of-sample estimate. mu is searched coarse (step 0.05) then fine (step 0.025) on every 2nd sample.
    import torch
    from fss_decode import fit_calibration, calibrate, decode
    torch.set_num_threads(os.cpu_count())
    tables, mus, dec = [], [], []
    for k in range(3):
        table = fit_calibration(PV[:, k], yv[:, k] > 1.0)
        c_half, y_half = calibrate(PV[::2, k], table), yv[::2, k:k+1]
        if a.mu:
            mu = float(a.mu.split(",")[k])
        else:
            sc = lambda m: fss_pooled(decode(c_half, m)[:, None] * 10.0, y_half)[0]
            tried = {round(m, 3): sc(m) for m in np.arange(0.25, 0.501, 0.05)}
            m0 = max(tried, key=tried.get)
            for m in (m0 - 0.025, m0 + 0.025): tried[round(m, 3)] = sc(m)
            mu = max(tried, key=tried.get)
        full = fss_pooled(decode(calibrate(PV[:, k], table), mu)[:, None] * 10.0, yv[:, k:k+1])[0]
        print(f"lead {k+1}: decoder mu {mu:.3f} fss {full:.4f} (threshold {best[k]:.4f})", flush=True)
        tables.append(table.tolist()); mus.append(float(mu)); dec.append(float(full))
    cfg["decode"] = {"mu": mus, "tables": tables}
    cfg["val_fss_decoded"] = float(np.mean(dec)); cfg["per_lead_decoded"] = dec
    print("decoded validation FSS", round(cfg["val_fss_decoded"], 4))
json.dump(cfg, open(a.out, "w"), indent=1)
np.save(a.out.replace(".json", "_testprob.npy"), PT.astype(np.float32))
print("saved", a.out, "and blended test probs")
