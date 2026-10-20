"""Blend test probability maps, threshold per lead time, write submission.csv in the host's format."""
import sys, json, numpy as np, pandas as pd
out, thr_json, probs = sys.argv[1], sys.argv[2], sys.argv[3:]
thr = json.load(open(thr_json))["thr"]
P = np.mean([np.load(p) for p in probs], axis=0)  # (200,3,128,128)
pred = np.stack([(P[:, k] > thr[k]) * 10.0 for k in range(3)], 1).astype(np.float32)
assert pred.shape == (200, 3, 128, 128) and np.isfinite(pred).all()
flat = pred.transpose(0, 2, 3, 1).reshape(-1, 3)
sub = pd.DataFrame({"id": np.arange(len(flat)), "t1": flat[:, 0], "t2": flat[:, 1], "t3": flat[:, 2]})
sub.to_csv(out, index=False)
print("wrote", out, sub.shape, "rain frac per lead", (flat > 1).mean(axis=0).round(4))
