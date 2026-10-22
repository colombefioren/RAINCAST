"""Download what the analysis needs from a run-queue session's output (TPU or GPU), run by run, so the local
disk stays small.

A full TPU session leaves about 10 GB on Kaggle (see build_kernel_queue.py). Kept locally:
- for every validation run: its summary, its best last-k validation blend, stored as uint8 (probability
  x 255: half the size of float16 and finer than the decoder's 1001-step calibration table), and the mean
  test probability of the same k epochs, both under the blend's name (e.g. <tag>_last3_valprob.npy and
  <tag>_last3_testprob.npy, the pair src/blend.py expects);
- for every full-data run: the mean test probability of its last k epochs, with k taken from its validation
  twin (the run with the same settings apart from the seed and --full), or 4 without a twin.
Per-epoch files are deleted once averaged.

Usage: python src/fetch_tpu_runs.py <owner>/<kernel-slug> <output folder>
The first argument may also be a local folder holding a downloaded output (used to test this script).
"""
import json, os, re, shutil, subprocess, sys, numpy as np

source, out = sys.argv[1], sys.argv[2]
KAGGLE = os.environ.get("KAGGLE", "kaggle")
os.makedirs(out, exist_ok=True)

def fetch(pattern):
    """Bring the output files whose names match the regular expression into the output folder."""
    if os.path.isdir(source):
        for f in os.listdir(source):
            if re.search(pattern, f): shutil.copy(f"{source}/{f}", out)
    else:
        subprocess.run([KAGGLE, "kernels", "output", source, "-p", out, "--file-pattern", pattern, "-q", "-o"], check=True)

def twin_key(args):
    """A run's settings without seed, --full and paths: a validation run and its full-data twin share it."""
    a, key, i = args.split(), [], 0
    while i < len(a):
        if a[i] in ("--seed", "--out", "--data"): i += 2
        elif a[i] == "--full": i += 1
        else: key.append(a[i]); i += 1
    return " ".join(key)

def mean_test(tag, k, epochs):
    """Average the test probabilities of the given epochs into <tag>_<k>_testprob.npy."""
    fetch(rf"^{re.escape(tag)}_ep({'|'.join(map(str, epochs))})_testprob\.npy$")
    files = [f"{out}/{tag}_ep{e}_testprob.npy" for e in epochs]
    missing = [f for f in files if not os.path.exists(f)]
    if missing: print(f"{tag}: missing {missing}, skipped"); return False
    np.save(f"{out}/{tag}_{k}_testprob.npy", np.mean([np.load(f).astype(np.float32) for f in files], 0).astype(np.float16))
    for f in files: os.remove(f)
    return True

fetch(r"\.json$")
# session.json, or tpu_session.json for the sessions built before the GPU queue existed
log = next(f"{out}/{f}" for f in ["session.json", "tpu_session.json"] if os.path.exists(f"{out}/{f}"))
runs = [r for r in json.load(open(log))["runs"] if r["trained"]]
best = {}  # twin key -> (k, epochs) of the validation run
for r in runs:
    summary = f"{out}/{r['tag']}.json"
    if "--full" in r["args"].split() or not os.path.exists(summary): continue
    blends = json.load(open(summary)).get("epoch_blends", {})
    if not blends: print(f"{r['tag']}: no epoch blends, skipped"); continue
    k = max(blends, key=lambda b: blends[b]["val_fss"]); epochs = blends[k]["epochs"]
    fetch(rf"^{re.escape(r['tag'])}_{k}_valprob\.npy$")
    vp = f"{out}/{r['tag']}_{k}_valprob.npy"
    if not os.path.exists(vp): print(f"{r['tag']}: no {k} validation blend, skipped"); continue
    np.save(vp, np.rint(np.load(vp).astype(np.float32) * 255).astype(np.uint8))
    if mean_test(r["tag"], k, epochs):
        best[twin_key(r["args"])] = (k, epochs)
        print(f"{r['tag']}: {k} (epochs {epochs}) validation {blends[k]['val_fss']:.4f}")
for r in runs:
    if "--full" not in r["args"].split(): continue
    a = r["args"].split(); n_ep = int(a[a.index("--epochs") + 1]) if "--epochs" in a else 8
    twin = twin_key(r["args"]) in best
    k, epochs = best[twin_key(r["args"])] if twin else ("last4", list(range(n_ep - 4, n_ep)))
    if mean_test(r["tag"], k, epochs): print(f"{r['tag']}: {k} (epochs {epochs})", "from its validation twin" if twin else "by default, no validation twin")
