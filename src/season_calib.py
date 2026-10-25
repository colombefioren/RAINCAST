"""Fit and check the decoder's calibration across rain seasons.

The decoder (src/fss_decode.py) is calibrated on one held-out season (December 2022 to March 2023). Fold models
trained with --val_season give out-of-fold probabilities for the other training seasons, so the calibration can be
fitted on several seasons and checked season by season: for each season, the isotonic table and the context model
(src/context_calib.py) are fitted on the other seasons and scored on that one (decoded pooled FSS), next to the
isotonic table fitted on the season itself (the current practice, in-sample).

Usage: python src/season_calib.py --data data --out outputs/season_calib.json --members2022 outputs/blend32cw2dec.json
         --folds "outputs/molab-set8/unetplusplus_v{season}_resnet34_s*_last*_valprob.npy" [--seasons 2019,2020,2021,2022]
The season blends are equal-weight means of their members (the 2022 blend keeps its configured weights). The output
config holds, per lead, the isotonic table and the context model fitted on every listed season, with the mu given
by --mu, in the format make_submission.py reads."""
import argparse, glob, json, os, numpy as np, pandas as pd, torch
from fss import fss_pooled, load_prob
from fss_decode import fit_calibration, calibrate, decode
import context_calib

p = argparse.ArgumentParser()
p.add_argument("--data", required=True); p.add_argument("--out", required=True)
p.add_argument("--members2022", required=True, help="blend .json whose members are the 2022 season's validation runs")
p.add_argument("--folds", required=True, help="glob with {season} for the other seasons' fold members")
p.add_argument("--seasons", default="2019,2020,2021,2022")
p.add_argument("--mu", default="0.425,0.375,0.35")
p.add_argument("--check", type=int, default=1, help="0: only fit the final config on every season (faster)")
a = p.parse_args()
torch.set_num_threads(os.cpu_count())
seasons = [int(s) for s in a.seasons.split(",")]; MU = [float(m) for m in a.mu.split(",")]
tr = np.load(f"{a.data}/train.npz"); ts = pd.to_datetime(tr["times"])
X = tr["X"][:, 1:3].astype(np.float16)  # rain one hour before and at the forecast time, half the size
Y = tr["y"] > 1.0; del tr  # the whole arrays would hold 4 GB next to the per-season blends

def season_rows(s): return np.where(np.asarray((ts >= f"{s}-12-01") & (ts < f"{s + 1}-04-01")))[0]

def members(s):
    if s == 2022:
        cfg = json.load(open(a.members2022)); w = np.asarray(cfg.get("weights") or [1.0] * len(cfg["probs"]), np.float64)
        return cfg["probs"], w / w.sum()  # paths as blend.py saw them (run this script from the same folder, src/)
    files = sorted(glob.glob(a.folds.format(season=s))); assert files, f"no fold members for {s}"
    return files, np.full(len(files), 1.0 / len(files))

S = {}  # per season: blend probabilities (n, 3, H, W) float16, truth (n, 3, H, W) bool, rain now / before, hours
for s in seasons:
    files, w = members(s); rows = season_rows(s)
    PV = None
    for wi, f in zip(w, files):
        pr = load_prob(f)
        assert len(pr) == len(rows), f"{f}: {len(pr)} maps for {len(rows)} samples of the {s} season"
        PV = wi * pr if PV is None else PV + wi * pr
    Xs = X[rows]
    S[s] = dict(P=PV.astype(np.float16), y=Y[rows], now=Xs[:, 1].copy(), prev=Xs[:, 0].copy(), hours=ts[rows].hour.values)
    print(f"season {s}: {len(rows)} samples, {len(files)} members", flush=True)
    del Xs
del X, Y

def fit_lead(train, k):
    """Isotonic table and context model of lead k fitted on the given seasons."""
    P = np.concatenate([S[s]["P"][:, k].astype(np.float32) for s in train]); o = np.concatenate([S[s]["y"][:, k] for s in train])
    table = fit_calibration(P, o)
    ctx = context_calib.fit(P, o, table, np.concatenate([S[s]["now"] for s in train]), np.concatenate([S[s]["prev"] for s in train]),
                            np.concatenate([(S[s]["hours"] + k + 1) % 24 for s in train]), rate=1.0 / len(train))
    return table, ctx

def score(s, k, table, ctx=None):
    P = S[s]["P"][:, k].astype(np.float32); o = (S[s]["y"][:, k:k+1] * 10.0).astype(np.float32)
    c = calibrate(P, table) if ctx is None else context_calib.apply(P, table, ctx, S[s]["now"], S[s]["prev"], (S[s]["hours"] + k + 1) % 24)
    return fss_pooled(decode(c, MU[k])[:, None] * 10.0, o)[0]

result = {"seasons": seasons, "mu": MU, "check": {}}
if a.check and len(seasons) > 1:
    for s in seasons:
        others = [t for t in seasons if t != s]; row = {}
        for k in range(3):
            own = fit_calibration(S[s]["P"][:, k].astype(np.float32), S[s]["y"][:, k])
            table, ctx = fit_lead(others, k)
            row[f"lead{k+1}"] = {"isotonic in-season": score(s, k, own), "isotonic other seasons": score(s, k, table),
                                 "context other seasons": score(s, k, table, ctx)}
        result["check"][s] = row
        print(f"held-out season {s}: " + " | ".join(f"lead {k+1} " + " ".join(f"{n.split()[0]}{'' if 'in-' in n else '*'} {v:.5f}" for n, v in row[f'lead{k+1}'].items()) for k in range(3)), flush=True)
    print("(* fitted on the other seasons)", flush=True)
tables, context = [], []
for k in range(3):
    table, ctx = fit_lead(seasons, k); tables.append(table.tolist()); context.append(ctx)
cfg2022 = json.load(open(a.members2022))  # the test side blends the 2022 members' full-data twins with these weights
result.update(thr=cfg2022["thr"], dilate=[1, 1, 1], probs=cfg2022["probs"], weights=cfg2022["weights"], data=a.data,
              decode={"mu": MU, "tables": tables, "context": context})
json.dump(result, open(a.out, "w"))
print("wrote", a.out, flush=True)
