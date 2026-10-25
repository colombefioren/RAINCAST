"""Blend test probability maps, threshold per lead time (or decode, if the config has a "decode" entry),
write submission.csv in the host's format."""
import sys, json, numpy as np, pandas as pd
out, thr_json, probs = sys.argv[1], sys.argv[2], sys.argv[3:]
cfg = json.load(open(thr_json)); thr = cfg["thr"]; dil = cfg.get("dilate", [1, 1, 1])
from scipy.ndimage import maximum_filter
from fss import load_prob
w = np.asarray(cfg.get("weights") or [1.0] * len(probs), np.float64)  # the blend's member weights (equal by default)
assert len(w) == len(probs), f"{len(probs)} probability files for {len(w)} blend weights"
P = sum(wi * load_prob(p) for wi, p in zip(w / w.sum(), probs)).astype(np.float32)  # (200,3,128,128), one file in memory at a time
pred = np.stack([maximum_filter((P[:, k] > thr[k]).astype(np.float32), size=(1, dil[k], dil[k])) * 10.0 if dil[k] > 1 else (P[:, k] > thr[k]) * 10.0 for k in range(3)], 1).astype(np.float32)
if "decode" in cfg:  # expected-FSS decoder instead of thresholds, see src/fss_decode.py
    from fss_decode import calibrate, calibrate_wet, decode_best
    dec, wet, ctx = cfg["decode"], cfg["decode"].get("wet"), cfg["decode"].get("context")
    if ctx:  # neighbourhood-aware calibration (src/context_calib.py): reads each test sample's own rain maps and hour
        import context_calib
        te = np.load(f"{cfg.get('data', '../data')}/test.npz"); Xt = te["X"]; hrs = pd.to_datetime(te["times"]).hour.values
        assert len(Xt) == len(P)
        cal = lambda k: context_calib.apply(P[:, k], np.array(dec["tables"][k], np.float32), ctx[k], Xt[:, 2], Xt[:, 1], (hrs + k + 1) % 24)
    elif wet:
        cal = lambda k: calibrate_wet(P[:, k], wet["tables"][k], np.array(wet["edges"][k]))
    else:
        cal = lambda k: calibrate(P[:, k], np.array(dec["tables"][k], np.float32))
    pred = np.stack([decode_best(cal(k), dec["mu"][k]) * 10.0 for k in range(3)], 1).astype(np.float32)
    if ctx: print("calibrated with the neighbourhood-aware model")
    if wet and not ctx: print("calibrated per dry-to-wet quartile")
    print("decoded with mu", cfg["decode"]["mu"])
assert pred.shape == (200, 3, 128, 128) and np.isfinite(pred).all()
flat = pred.transpose(0, 2, 3, 1).reshape(-1, 3)
sub = pd.DataFrame({"id": np.arange(len(flat)), "t1": flat[:, 0], "t2": flat[:, 1], "t3": flat[:, 2]})
sub.to_csv(out, index=False)
print("wrote", out, sub.shape, "rain frac per lead", (flat > 1).mean(axis=0).round(4))
