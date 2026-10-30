"""Draw one validation case: the three input rain maps, the observed rain masks at +1h, +2h and +3h, and the
masks forecast by one network, saved as docs/forecast.png.

Usage (from the repository root, with data/train.npz downloaded and a validation run's probabilities in outputs/):
  python tools/plot_forecast.py outputs/<session>/<run>_last6_valprob.npy [sample index in the validation season]

Without an index, the sample with the most rain in its +1h target among those where the inputs hold some rain
is drawn. Thresholds come from the run's summary json (per lead time), as src/blend.py would apply them."""
import json, os, sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

prob_file = sys.argv[1]
run_json = prob_file.split("_last")[0] + ".json"
thr = json.load(open(run_json))["epoch_blends"]["last6"]["thr"] if os.path.exists(run_json) else [0.5, 0.5, 0.5]
tr = np.load("data/train.npz"); ts = pd.to_datetime(tr["times"])
val = np.flatnonzero((ts >= "2022-12-01") & (ts < "2023-04-01"))
yv = np.load("data/yval.npy") if os.path.exists("data/yval.npy") else (tr["y"][val] > 1.0) * 10.0
prob = np.load(prob_file).astype(np.float32) / 255
rain = (yv[:, 0] > 1).mean(axis=(1, 2))
i = int(sys.argv[2]) if len(sys.argv) > 2 else int(np.argmax(rain))
X = tr["X"][val[i]].astype(np.float32); when = ts[val[i]]
obs = yv[i] > 1; fc = np.stack([prob[i, k] > thr[k] for k in range(3)])

fig, ax = plt.subplots(3, 3, figsize=(9.6, 9.9), dpi=110)
fig.patch.set_facecolor("#0f1115")
for a in ax.ravel(): a.set_facecolor("#0f1115"); a.set_xticks([]); a.set_yticks([])
for k in range(3):
    ax[0, k].imshow(np.maximum(X[k], 0.05), cmap="Blues", norm=LogNorm(vmin=0.05, vmax=30), interpolation="nearest")
    ax[0, k].set_title(f"input  t{['-2h', '-1h', ''][k]}", color="#e6e6e6", fontsize=11)
    ax[1, k].imshow(obs[k], cmap="Blues", vmin=0, vmax=1.25, interpolation="nearest")
    ax[1, k].set_title(f"observed rain > 1 mm/h  t+{k + 1}h", color="#e6e6e6", fontsize=11)
    ax[2, k].imshow(fc[k], cmap="Oranges", vmin=0, vmax=1.25, interpolation="nearest")
    ax[2, k].set_title(f"forecast  t+{k + 1}h", color="#e6e6e6", fontsize=11)
fig.suptitle(f"{when:%d %B %Y, %H:%M} UTC", color="#e6e6e6", fontsize=13, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))
os.makedirs("docs", exist_ok=True)
fig.savefig("docs/forecast.png", facecolor=fig.get_facecolor())
print(f"docs/forecast.png: validation sample {i} ({when}), thresholds {thr}")
