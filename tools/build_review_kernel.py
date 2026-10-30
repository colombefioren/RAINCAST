"""Build the code-review package asked of the top ten participants: a Kaggle notebook that reproduces the final
submission from the published code and the saved probability maps, and the two datasets it reads.

Usage (from the repository root, with the final submission's files under outputs/):
  python tools/build_review_kernel.py

Writes:
  outputs/review/code/      the tracked files of the repository at HEAD (dataset cocofioren/indabax-rainfall-code)
  outputs/review/members/   the 58 members' validation and test probabilities and their twins' test probabilities,
                            compressed, with index.json and the submitted file (dataset cocofioren/indabax-rainfall-members)
  kernel/indabax-rainfall-reproduction/   the notebook and its kernel-metadata.json

Publish with:
  kaggle datasets create -p outputs/review/code        (or: kaggle datasets version -p outputs/review/code -m "<note>")
  kaggle datasets create -p outputs/review/members -r skip
  kaggle kernels push -p kernel/indabax-rainfall-reproduction

The notebook has three parts: training one member and its twin from random weights with the exact command of the
run lists (optional, about three hours on two T4 GPUs), the blend and decoder fit on the held-out season, and the
decoding of the twins' test probabilities into the submission file, checked against the submitted one."""
import json, os, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OWNER = "cocofioren"
CODE, MEMBERS, KERNEL = "indabax-rainfall-code", "indabax-rainfall-members", "indabax-rainfall-reproduction"
COMPETITION = "hackathon-IndabaX-MDG-2026"
SUBMITTED = "outputs/sub40_blend41csi17_twins_decoded.csv"  # submission 40, public leaderboard 0.78384
LIVE_MEMBER = ("runs/cloud-scratch-set11b.txt", 1685)  # the member trained live in the notebook, with its twin 1785

final = json.load(open("final/submission.json"))

# 1. the code: every tracked file at HEAD
code_dir = "outputs/review/code"
shutil.rmtree(code_dir, ignore_errors=True); os.makedirs(code_dir)
subprocess.run(f"git archive HEAD | tar -x -C {code_dir}", shell=True, check=True)
json.dump({"title": "RAINCAST code", "id": f"{OWNER}/{CODE}", "licenses": [{"name": "MIT"}],
           "subtitle": "Rainfall nowcasting from satellite rain maps: the RAINCAST repository",
           "description": "The RAINCAST repository (https://github.com/colombefioren/RAINCAST) at the commit of the final "
                          "submission: training (`src/train_unet.py`), blend and decoder (`src/blend.py`, `src/fss_decode.py`), "
                          "submission step (`src/twins.py`, `src/make_submission.py`), the run list of every training session "
                          "(`runs/`), the members of the final submission (`final/`), the design notes and the rules checklist "
                          "(`docs/`). Read by the notebook `indabax-rainfall-reproduction`, which rebuilds the final submission "
                          "from the saved probability maps of the dataset `indabax-rainfall-members`."},
          open(f"{code_dir}/dataset-metadata.json", "w"), indent=1)

# 2. the members: compressed probability maps (written beforehand by outputs/scratch or by this script), index, submission
mem_dir = "outputs/review/members"
os.makedirs(mem_dir, exist_ok=True)
twins = subprocess.run([sys.executable, "twins.py"] + ["../" + m for m in final["members"]],
                       capture_output=True, text=True, cwd="src", check=True).stdout.split()
index = {"members": [], "members_test": [], "twins": [], "weights": final["weights"], "mu": final["decode"]["mu"]}
import numpy as np
for m, t in zip(final["members"], twins):
    for key, f in (("members", m), ("members_test", m.replace("_valprob", "_testprob")), ("twins", t[3:])):
        name = os.path.basename(f)[:-4]
        dst = f"{mem_dir}/{name}.npz"
        if not os.path.exists(dst): np.savez_compressed(dst, prob=np.load(f))
        index[key].append(name + ".npz")
json.dump(index, open(f"{mem_dir}/index.json", "w"), indent=1)
shutil.copy(SUBMITTED, f"{mem_dir}/submission.csv")
shutil.copy("final/members.md", f"{mem_dir}/members.md")
json.dump({"title": "RAINCAST members", "id": f"{OWNER}/{MEMBERS}", "licenses": [{"name": "CC0-1.0"}],
           "subtitle": "Probability maps of the 58 networks behind the final RAINCAST forecast",
           "description": "For each of the 58 members of the final submission: its validation probability map (mean of its "
                          "last epochs on the held-out season, December 2022 to March 2023), its test probability map, and the "
                          "test probability map of its full-data twin (same settings, seed + 100, trained on every season), as "
                          "written by `src/train_unet.py` and compacted by `src/fetch_runs.py` of the RAINCAST repository "
                          "(dataset `indabax-rainfall-code`). Files are `.npz` with one array `prob` (uint8 probability x 255 "
                          "for validation maps, float16 for test maps). `index.json` lists the members, their twins, the blend "
                          "weights and the decoder mu; `members.md` describes each member; `submission.csv` is the submitted "
                          "file. The notebook `indabax-rainfall-reproduction` rebuilds the submission from these files."},
          open(f"{mem_dir}/dataset-metadata.json", "w"), indent=1)

# 3. the notebook
run_list, seed = LIVE_MEMBER
lines = [l.strip() for l in open(run_list) if l.strip() and not l.startswith("#")]
base = next(l for l in lines if l.startswith("base:"))[5:].strip()
own = next(l for l in lines if l.endswith(f"--seed {seed}"))
twin_own = f"{own} --full".replace(f"--seed {seed}", f"--seed {seed + 100}")
member_tag = f"unetplusplus_clim_oof_resnet34_s{seed}"
member_json = json.load(open(f"outputs/cloud-set11b/{member_tag}.json"))

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}
py = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}
cells = [
md(f"""
# RAINCAST: reproduction of the final submission

Code review package for *Hackathon - IndabaX Madagascar 2026*, participant Coco Fioren. The final submission is a
blend of 58 U-Net++ networks trained from random weights on the competition data only, calibrated and decoded into
the rain mask with the highest expected FSS. Public leaderboard {final['public_leaderboard']}, held-out season
{final['validation_fss_decoded']:.6f}.

Inputs of this notebook:

- `{OWNER}/{CODE}`: the repository (training, blend, decoder, submission scripts; `README.md`, `docs/APPROACH.md`,
  `docs/RULES.md`, `final/members.md`, the run lists of every training session in `runs/`).
- `{OWNER}/{MEMBERS}`: for each of the 58 members, its validation and test probability maps (mean of its last
  epochs; the validation season is December 2022 to March 2023) and the test probability map of its full-data
  twin (same settings, seed + 100, trained on every season), as written by `src/train_unet.py` and compacted by
  `src/fetch_runs.py`; `submission.csv` is the file that was submitted.
- the competition data.

The three parts below follow `README.md`, section *Reproducing the final submission*:

1. **Training.** The 58 members and their 58 twins are 116 runs of `src/train_unet.py` (about 55 GPU hours on an
   RTX Pro 6000; the run lists in `runs/` name every setting and seed). Part 1 trains one member and its twin here
   with their exact command and compares the result with the saved member. Set `TRAIN_LIVE = False` to skip it.
2. **Blend and decoder** on the held-out season: `src/blend.py` with the final weights and mu.
3. **Submission**: `src/make_submission.py` decodes the twins' blended test probabilities; the file must equal
   `submission.csv`.

Every test sample is processed on its own from its three rain maps and its own timestamp; no external data, no
pretrained weights, no look-up across samples (`docs/RULES.md`).
"""),
py(f"""
import glob, json, os, shutil, subprocess, sys
def find(slug):
    hits = [d for d in (f"/kaggle/input/datasets/{OWNER}/{{slug}}", f"/kaggle/input/{{slug}}") if os.path.isdir(d)]
    assert hits, slug
    return hits[0]
CODE, MEMBERS = find("{CODE}"), find("{MEMBERS}")
DATA = "/kaggle/input/competitions/{COMPETITION}"
assert os.path.exists(f"{{DATA}}/train.npz") and os.path.exists(f"{{DATA}}/test.npz")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"], check=True)
shutil.rmtree("/kaggle/working/code", ignore_errors=True); shutil.copytree(CODE, "/kaggle/working/code")
os.chdir("/kaggle/working/code/src")
print(open("../README.md").read()[:600])
"""),
md(f"""
## 1. Training one member and its twin

Member `{member_tag}` (session `{run_list}`, weight 3 in the blend, validation FSS
{member_json['epoch_blends']['last6']['val_fss']:.4f} for the mean of its last 6 epochs) and its twin, seed {seed + 100}
with `--full`. The command is the session's base line plus the run's own line. The two runs train side by side,
one per GPU (about three hours on two T4 GPUs); with one GPU they run one after the other.
"""),
py(f"""
TRAIN_LIVE = True
BASE = "{base}".replace("/kaggle/working", "/kaggle/working/live")
RUNS = ["{own}", "{twin_own}"]
if TRAIN_LIVE:
    import torch
    os.makedirs("/kaggle/working/live", exist_ok=True)
    n_gpu = max(1, torch.cuda.device_count())
    procs = []
    for i, own in enumerate(RUNS):
        cmd = [sys.executable, "-u", "train_unet.py"] + BASE.split() + own.split()
        log = open(f"/kaggle/working/live/run{{i}}.log", "w")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % n_gpu))))
        if n_gpu == 1: procs[-1].wait()
    for p in procs: p.wait()
    for i in range(len(RUNS)): print(open(f"/kaggle/working/live/run{{i}}.log").read()[-1500:])
"""),
py(f"""
if TRAIN_LIVE:
    import numpy as np
    from fss import fss_pooled, load_prob
    live = json.load(open("/kaggle/working/live/{member_tag}.json"))
    saved = {json.dumps({k: round(v["val_fss"], 4) for k, v in member_json["epoch_blends"].items()})}
    print("validation FSS of the last-k epoch means, live:", {{k: round(v["val_fss"], 4) for k, v in live["epoch_blends"].items()}})
    print("validation FSS of the last-k epoch means, saved:", saved)
    pv_live = np.mean([load_prob(f"/kaggle/working/live/{member_tag}_ep{{e}}_valprob.npy") for e in live["epoch_blends"]["last6"]["epochs"]], 0)
    pv_saved = np.load(f"{{MEMBERS}}/{member_tag}_last6_valprob.npz")["prob"].astype(np.float32) / 255
    print("correlation between the live and the saved probability maps:", round(float(np.corrcoef(pv_live.ravel(), pv_saved.ravel())[0, 1]), 4))
    pt_live = np.mean([load_prob(f"/kaggle/working/live/{member_tag.replace(f's{seed}', f's{seed + 100}')}_full_ep{{e}}_testprob.npy") for e in live["epoch_blends"]["last6"]["epochs"]], 0)
    pt_saved = np.load(f"{{MEMBERS}}/{member_tag.replace(f's{seed}', f's{seed + 100}')}_full_last6_testprob.npz")["prob"].astype(np.float32)
    print("correlation between the live and the saved twin test maps:", round(float(np.corrcoef(pt_live.ravel(), pt_saved.ravel())[0, 1]), 4))
"""),
md("""
## 2. Blend and decoder on the held-out season

The 58 validation maps are averaged with weights 1, 2 or 3 by family (`final/members.md`), an isotonic fit per
lead time calibrates the blend, and the expected-FSS decoder runs with mu 0.425, 0.375, 0.35
(`src/fss_decode.py`). `final.json` then holds the calibration tables and mu, as `final/submission.json` does.
"""),
py("""
import numpy as np
index = json.load(open(f"{MEMBERS}/index.json"))
os.makedirs("/kaggle/tmp/members", exist_ok=True)
for name in index["members"] + index["members_test"] + index["twins"]:
    np.save(f"/kaggle/tmp/members/{name[:-4]}.npy", np.load(f"{MEMBERS}/{name}")["prob"])
members = [f"/kaggle/tmp/members/{m[:-4]}.npy" for m in index["members"]]
twins = [f"/kaggle/tmp/members/{t[:-4]}.npy" for t in index["twins"]]
os.makedirs("/kaggle/tmp/data", exist_ok=True)
for f in ("train.npz", "test.npz"):
    if not os.path.exists(f"/kaggle/tmp/data/{f}"): os.symlink(f"{DATA}/{f}", f"/kaggle/tmp/data/{f}")
cmd = [sys.executable, "blend.py", "--data", "/kaggle/tmp/data", "--out", "/kaggle/working/final.json", "--decode", "1",
       "--mu", ",".join(map(str, index["mu"])), "--weights", ",".join(map(str, index["weights"]))] + members
subprocess.run(cmd, check=True)
final = json.load(open("/kaggle/working/final.json")); shipped = json.load(open("../final/submission.json"))
print("decoded validation FSS", round(final["val_fss_decoded"], 6), "| final/submission.json", shipped["validation_fss_decoded"])
"""),
md("""
## 3. The submission

The twins' test maps are blended with the same weights, calibrated with the tables of `final.json` and decoded
with the same mu; the result is written in the competition's format and compared with the submitted file.
"""),
py("""
subprocess.run([sys.executable, "make_submission.py", "/kaggle/working/submission.csv", "/kaggle/working/final.json"] + twins, check=True)
import pandas as pd
ours, shipped = pd.read_csv("/kaggle/working/submission.csv"), pd.read_csv(f"{MEMBERS}/submission.csv")
print("rows", len(ours), "| identical to the submitted file:", ours.equals(shipped), "| cells that differ:", int((ours.values != shipped.values).sum()))
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(f"kernel/{KERNEL}", exist_ok=True)
json.dump(nb, open(f"kernel/{KERNEL}/{KERNEL}.ipynb", "w"), indent=1)
json.dump({"id": f"{OWNER}/{KERNEL}", "title": KERNEL, "code_file": f"{KERNEL}.ipynb", "language": "python",
           "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
           "machine_shape": "NvidiaTeslaT4", "enable_internet": "true",
           "dataset_sources": [f"{OWNER}/{CODE}", f"{OWNER}/{MEMBERS}"], "competition_sources": [COMPETITION],
           "kernel_sources": []}, open(f"kernel/{KERNEL}/kernel-metadata.json", "w"), indent=1)
for c in cells:
    if c["cell_type"] == "code": compile(c["source"], KERNEL, "exec")
print(f"kernel/{KERNEL}: {len(cells)} cells; {code_dir}: {sum(len(f) for _, _, f in os.walk(code_dir))} files; "
      f"{mem_dir}: {len(os.listdir(mem_dir))} files")
