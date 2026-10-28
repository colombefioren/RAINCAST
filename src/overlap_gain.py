"""How much would the forbidden test overlap raise the score? Simulated on the validation season only:
the per-lead coverage comes from the test timestamps (how many test targets are another test sample's
input frame), and that many validation samples get a perfect copy of their target. No test target is used,
and nothing here feeds a submission: it only sizes the jump the leak would give, for docs/JOURNAL.md.
Usage: cd src && python overlap_gain.py ../outputs/<member>_valprob.npy"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/cocofioren/HACK/INDABAX-RAINFALL/src")
from fss import fss_components, load_prob
R = "/home/cocofioren/HACK/INDABAX-RAINFALL"
te = np.load(f"{R}/data/test.npz", allow_pickle=True)
tkey = [k for k in te.files if te[k].ndim == 1][0]
t = pd.to_datetime(te[tkey]); n = len(t); tset = {}
for i, ti in enumerate(t): tset.setdefault(ti, []).append(i)
cover = np.zeros((n, 3), bool)
for i, ti in enumerate(t):
    for k in (1, 2, 3):
        target = ti + pd.Timedelta(hours=k)
        # frame `target` is an input of sample j when t_j in {target, target+1h, target+2h}
        if any((target + pd.Timedelta(hours=d)) in tset for d in (0, 1, 2)): cover[i, k - 1] = True
print("test samples:", n, "| samples with any covered lead:", int(cover.any(1).sum()),
      "| covered per lead:", cover.sum(0).tolist(), "| all three leads:", int(cover.all(1).sum()))
frac = cover.mean(0)
yv = np.load(f"{R}/data/yval.npy").astype(np.float32)   # (N,3,H,W), 10 where rain > 1 mm/h
pv = load_prob(sys.argv[1]); N = len(yv)
# baseline: per-lead threshold like src/blend.py
pred = np.zeros_like(yv); base_k = []
for k in range(3):
    best = (-1, 0.5)
    for thr in np.arange(0.2, 0.71, 0.05):
        num, den = fss_components((pv[:, k] > thr) * 10.0, yv[:, k]); s = 1 - num.sum() / den.sum()
        if s > best[0]: best = (s, thr)
    pred[:, k] = (pv[:, k] > best[1]) * 10.0; base_k.append(best[0])
base = float(np.mean(base_k)); print(f"baseline validation FSS (this member): {base:.4f}  per lead {np.round(base_k, 4).tolist()}")
num = np.zeros((N, 3)); den = np.zeros((N, 3)); den_perfect = np.zeros((N, 3))
for k in range(3):
    num[:, k], den[:, k] = fss_components(pred[:, k], yv[:, k])
    _, den_perfect[:, k] = fss_components(yv[:, k], yv[:, k])
def score(mask):
    """pooled FSS when the masked (sample, lead) entries are replaced by a perfect copy of the target"""
    s = []
    for k in range(3):
        m = mask[:, k]; s.append(1 - (num[~m, k].sum()) / (den[~m, k].sum() + den_perfect[m, k].sum()))
    return float(np.mean(s))
rng = np.random.default_rng(0); draws = []
for _ in range(200):
    mask = np.zeros((N, 3), bool)
    for k in range(3): mask[rng.choice(N, int(round(frac[k] * N)), replace=False), k] = True
    draws.append(score(mask))
draws = np.array(draws)
print(f"with the overlap copied, random samples: mean {draws.mean():.4f} (min {draws.min():.4f}, max {draws.max():.4f}) -> gain +{draws.mean()-base:.4f}")
# the covered test samples could be rainier than average: take the rainiest samples instead
rain = yv.reshape(N, -1).sum(1); order = np.argsort(-rain); mask = np.zeros((N, 3), bool)
for k in range(3): mask[order[: int(round(frac[k] * N))], k] = True
s = score(mask); print(f"with the overlap copied, rainiest samples: {s:.4f} -> gain +{s-base:.4f}")
# what public score that means, with the +0.014 offset validation -> public seen on submission 40
print(f"same gains on the public board from 0.7838: random {0.7838 + draws.mean()-base:.4f}, rainiest {0.7838 + s - base:.4f}")
