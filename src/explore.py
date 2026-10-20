import sys, numpy as np, pandas as pd
from fss import report
D = sys.argv[1]
tr = np.load(f"{D}/train.npz")
X, y, times = tr["X"], tr["y"], tr["times"]
print(X.shape, y.shape, X.dtype)
ts = pd.to_datetime(times)
print("time range", ts.min(), ts.max())
print(pd.Series(ts.year * 100 + ts.month).value_counts().sort_index())
print("rain frac >1mm/h: X", (X > 1).mean(), "y", (y > 1).mean())
print("y max", y.max(), "quantiles", np.quantile(y[::50], [0.5, 0.9, 0.99, 0.999]))
print("frac of samples with any rain>1 in y[:,0]:", ((y[:, 0] > 1).sum(axis=(1, 2)) > 0).mean())
d = np.diff(ts.values).astype("timedelta64[h]").astype(int)
print("train gaps(h): min/median/max", d.min(), np.median(d), d.max(), " count==1:", (d == 1).sum())
# validation = last season (Dec 2022 - Mar 2023)
val = np.asarray((ts >= "2022-12-01") & (ts < "2023-04-01"))
print("val samples", val.sum(), "train samples", (~val).sum())
yv, Xv = y[val], X[val]
# baselines
pers = np.repeat(Xv[:, 2:3], 3, axis=1)
report("persistence(t)", pers, yv)
report("zeros", np.zeros_like(yv), yv)
report("ones", np.full_like(yv, 5.0), yv)
mean3 = np.repeat(Xv.mean(axis=1, keepdims=True), 3, axis=1)
report("mean of 3 inputs", mean3, yv)
from scipy.ndimage import maximum_filter, uniform_filter
for s in (3, 5):
    dil = maximum_filter(Xv[:, 2], size=(1, s, s))
    report(f"persistence dilated {s}x{s}", np.repeat(dil[:, None], 3, axis=1), yv)
    sm = uniform_filter(Xv[:, 2], size=(1, s, s), mode="constant")
    for thr in (0.3, 0.5, 0.7, 1.0):
        report(f"pers smooth{s} thr{thr}", np.repeat((sm[:, None] > thr) * 10.0, 3, axis=1), yv)
