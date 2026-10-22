"""Bundle a list of training runs into one Kaggle TPU v5e-8 kernel that trains them eight at a time, one run
per TPU chip, for as long as the session's time budget allows.

The TPU quota (20 hours a week) is separate from the GPU quota and counts session hours, so eight runs side by
side cost the quota of one. A TPU session also waits in Kaggle's queue before it starts (often 12 hours or
more in October 2026), so one long session that trains many runs is worth more than several short ones.

How the session runs:
- Each run is an independent process that owns one chip: the TPU_* environment variables make every process
  a one-chip TPU of its own, so a run that ends early does not disturb the others.
- Eight workers, one per chip, take the next run from the list as soon as their chip is free. A worker only
  starts a run that should end within the time budget, judged from finished runs of the same model.
- The competition arrays are unpacked once into .npy files that every run memory-maps, so the runs share one
  copy of the training data in RAM.
- segmentation-models-pytorch is installed with the TPU stack (torch, torch_xla, ...) pinned as pip
  constraints, so it cannot replace the torch build that torch_xla needs.
- As soon as a run ends, the files the analysis needs are copied to /kaggle/working (the kernel output) and
  its scratch files are deleted.
- If the one-chip processes fail before training (three quick failures while nothing trained) or nothing
  trains within 45 minutes, the remaining runs go one after another on the default device.
- tpu_session.json in the output records every run: chip, start, duration, exit code and files kept.

Usage: python build_kernel_tpu.py <kernel-slug> <runs-file> [budget-hours, default 7.75]
The runs file holds a "base:" line with the train_unet.py arguments every run shares, then one run per line
(its own arguments, in priority order); # starts a comment. Push with -t 32400, Kaggle's 9-hour TPU limit.
"""
import json, subprocess, sys

slug, runs_file = sys.argv[1], sys.argv[2]
budget_h = float(sys.argv[3]) if len(sys.argv) > 3 else 7.75
base, runs = [], []
for line in open(runs_file):
    line = line.split("#")[0].strip()
    if line.startswith("base:"): base = line[len("base:"):].split()
    elif line: runs.append(base + line.split())
assert base and runs, __doc__
args = [[x.replace("/kaggle/working", "/kaggle/tmp/out") for x in r] + ["--xla", "1"] for r in runs]
assert all(r[r.index("--out") + 1] == "/kaggle/tmp/out" for r in args), "every run needs --out /kaggle/working"

HELPERS = r'''opt = lambda run, k, d: run[run.index(k) + 1] if k in run else d
def tag_of(run):
    """The file prefix train_unet.py gives a run (arch, attention, encoder, seed, full)."""
    arch, att = opt(run, "--arch", "Unet"), opt(run, "--attention", "")
    return (("" if arch == "Unet" else arch.lower() + "_") + (att + "_" if att else "") + opt(run, "--encoder", "resnet18")
            + "_s" + opt(run, "--seed", "42") + ("_full" if "--full" in run else ""))
'''
exec(HELPERS)  # the builder checks file names with the same helpers the launcher uses

LAUNCHER = r'''import glob, json, os, re, shutil, subprocess, sys, threading, time
from importlib import metadata
T0 = time.time()
BUDGET, HARD = __BUDGET__, __HARD__  # seconds from now: runs must be expected to end before BUDGET; at HARD every run stops
RUNS = __RUNS__
OUT, DATA, WORK = "/kaggle/tmp/out", "/kaggle/tmp/data", "/kaggle/working"
for d in (OUT, DATA): os.makedirs(d, exist_ok=True)
def log(*msg):  # one write per line, so lines from different threads do not mix
    print(f"[{(time.time() - T0) / 60:5.1f} min] " + " ".join(map(str, msg)) + "\n", end="", flush=True)

# what the machine offers, for the log (commands run without a shell)
for cmd in (["nproc"], ["free", "-g"], ["df", "-h", "/kaggle/tmp", "/kaggle/working"], [sys.executable, "--version"]):
    try: subprocess.run(cmd)
    except OSError as e: print(cmd[0], "unavailable:", e, flush=True)
installed = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True).stdout.splitlines()
print("\n".join(l for l in installed if re.match(r"(torch|torchvision|torch[-_]xla|libtpu(-nightly)?|numpy) ", l, re.I)), flush=True)

# segmentation-models-pytorch must not replace the torch build that torch_xla was compiled against, so the
# installed versions of the TPU stack are pip constraints
pins = []
for name in ["torch", "torchvision", "torchaudio", "torch_xla", "libtpu", "libtpu-nightly", "numpy"]:
    try: pins.append(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError: pass
open("/kaggle/tmp/pins.txt", "w").write("\n".join(pins) + "\n")
log("pip constraints:", pins)
imports = lambda: subprocess.run([sys.executable, "-c", "import torch_xla, segmentation_models_pytorch"]).returncode == 0
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-c", "/kaggle/tmp/pins.txt", "segmentation-models-pytorch"])
if not imports():  # the pinned install could not resolve: install the package code alone
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "segmentation-models-pytorch", "timm"])
if not imports(): sys.exit("torch_xla or segmentation_models_pytorch does not import, see the errors above")
open("/kaggle/tmp/train.py", "w").write(__CODE__)

# the competition arrays, unpacked once into .npy files that every run memory-maps: the runs then share one
# copy of the 4.3 GB of training data in RAM instead of loading one each
import numpy as np
found = glob.glob("/kaggle/input/**/train.npz", recursive=True)
if found:
    src = os.path.dirname(found[0])
    for split in ["train", "test"]:
        z = np.load(f"{src}/{split}.npz")
        for k in z.files: np.save(f"{DATA}/{split}_{k}.npy", z[k])
    for r in RUNS: r[r.index("--data") + 1] = DATA
    log("data unpacked into", DATA, sorted(os.listdir(DATA)))
else:
    log("no train.npz under /kaggle/input: every run reads its own --data path")

__HELPERS__
def model_of(run):
    """Runs of the same model take the same time per epoch."""
    return " ".join(opt(run, k, "") for k in ["--arch", "--attention", "--encoder", "--upscale", "--bs"])
def work(run):
    """Epochs, a full-data epoch counting 1.2: a third more training samples but no validation pass."""
    return int(opt(run, "--epochs", "8")) * (1.2 if "--full" in run else 1.0)

def collect(run):
    """Copy what the analysis needs to the kernel output: summary, weights, test probabilities (final and last
    epochs) and the best last-k validation blend, or every validation file of a run that stopped before its
    summary. Then free the scratch disk."""
    tag = tag_of(run); summary = f"{OUT}/{tag}.json"
    keep = glob.glob(summary) + glob.glob(f"{OUT}/{tag}.pt") + glob.glob(f"{OUT}/{tag}_*testprob.npy")
    if os.path.exists(summary):
        blends = json.load(open(summary)).get("epoch_blends", {})
        k = max(blends, key=lambda b: blends[b]["val_fss"]) if blends else None
        keep.append(f"{OUT}/{tag}_{k}_valprob.npy" if k else f"{OUT}/{tag}_valprob.npy")
        if k: log(f"[{tag}] best blend {k} {blends[k]['val_fss']:.4f}")
    else:
        keep += glob.glob(f"{OUT}/{tag}_*valprob.npy")
    kept = []
    for f in keep:
        if os.path.exists(f): shutil.copy(f, WORK); kept.append(os.path.basename(f))
    for f in glob.glob(f"{OUT}/{tag}.*") + glob.glob(f"{OUT}/{tag}_*"): os.remove(f)
    return kept

lock = threading.Lock()
todo = list(range(len(RUNS)))  # runs not started yet, in priority order
rate = {}                      # model -> seconds per unit of work, measured on finished runs
procs, report, quick_failures, retried, trained_runs = {}, [], [], set(), set()
trained_any, pinning_failed, stop = threading.Event(), threading.Event(), threading.Event()

def run_one(i, chip):
    """Train RUNS[i] on one chip (None: the default device), relay its log, then collect its files."""
    env = dict(os.environ, PJRT_DEVICE="TPU", PYTHONWARNINGS="ignore")
    if chip is not None:  # a one-chip TPU of its own
        port = 8476 + chip
        env.update(TPU_VISIBLE_CHIPS=str(chip), TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1", TPU_PROCESS_BOUNDS="1,1,1",
                   TPU_PROCESS_PORT=str(port), TPU_PROCESS_ADDRESSES=f"localhost:{port}", CLOUD_TPU_TASK_ID="0",
                   ALLOW_MULTIPLE_LIBTPU_LOAD="1")
    t, trained = time.time(), False
    log(f"[run{i + 1}] starts on chip {chip}: {tag_of(RUNS[i])}")
    p = subprocess.Popen([sys.executable, "-u", "/kaggle/tmp/train.py"] + RUNS[i], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with lock: procs[i] = p
    for line in p.stdout:  # each line gets the run number, so the kernel log tells the runs apart
        if " train_loss " in line and not trained: trained = True; trained_any.set()
        if re.match(r"[IW]\d{4} ", line): continue  # info and warning lines of the TPU runtime
        print(f"[run{i + 1}] " + line, end="", flush=True)
    p.wait(); took = time.time() - t
    if trained: trained_runs.add(i)
    files = collect(RUNS[i])
    log(f"[run{i + 1}] exit code {p.returncode} after {took / 60:.1f} min, {len(files)} files kept")
    with lock:
        procs.pop(i, None)
        if p.returncode == 0:
            m = model_of(RUNS[i]); rate[m] = max(rate.get(m, 0.0), took / work(RUNS[i]))
        report.append(dict(run=i + 1, tag=tag_of(RUNS[i]), args=" ".join(RUNS[i]), chip=chip, start_min=round((t - T0) / 60, 1),
                           minutes=round(took / 60, 1), exit_code=p.returncode, trained=trained, files=files))
        json.dump(dict(seconds_per_epoch=rate, runs=report, not_started=[j + 1 for j in todo]),
                  open(f"{WORK}/tpu_session.json", "w"), indent=1)
    return p.returncode, trained

def estimate(run):
    """Expected seconds for a run: from finished runs of the same model, else twice the slowest model so far."""
    if model_of(run) in rate: return rate[model_of(run)] * work(run)
    return 2 * max(rate.values()) * work(run) if rate else 0.0

def worker(chip):
    fails = 0  # runs in a row that failed on this chip before training
    while not stop.is_set():
        with lock:
            now = time.time() - T0
            i = next((j for j in todo if now + 1.1 * estimate(RUNS[j]) < BUDGET), None)
            if i is None: return
            todo.remove(i)
        code, trained = run_one(i, chip)
        if code == 0 or trained: fails = 0; continue
        fails += 1
        with lock:
            quick_failures.append(i)
            if i not in retried: retried.add(i); todo.append(i)  # one more try at the end, maybe on another chip
            if len(quick_failures) >= 3 and not trained_any.is_set(): pinning_failed.set(); stop.set()
        if fails >= 3: return log(f"chip {chip} retired: three runs in a row failed on it before training")

def stop_all(reason):
    stop.set(); log("stopping every run:", reason)
    with lock: live = list(procs.values())
    for p in live: p.terminate()
    time.sleep(30)
    for p in live:
        if p.poll() is None: p.kill()

pinned = True
def watchdog():
    """Stops every run at HARD; when nothing has trained 45 minutes into the one-chip phase, gives up on it."""
    while True:
        time.sleep(20); el = time.time() - T0
        if el > HARD: return stop_all("time limit")
        if pinned and el > 45 * 60 and not trained_any.is_set() and not stop.is_set():
            pinning_failed.set(); stop_all("nothing trained in 45 minutes with one chip per run")
threading.Thread(target=watchdog, daemon=True).start()

workers = [threading.Thread(target=worker, args=(c,)) for c in range(8)]
for w in workers: w.start()
for w in workers: w.join()
pinned = False
if pinning_failed.is_set() and time.time() - T0 < HARD:
    with lock: rest = [j for j in range(len(RUNS)) if j not in trained_runs]; todo.clear()
    log("one chip per run did not work: runs", [j + 1 for j in rest], "go one after another on the default device")
    stop.clear()
    for i in rest:
        if stop.is_set() or time.time() - T0 + 1.1 * estimate(RUNS[i]) > BUDGET: continue
        run_one(i, None)
log("done:", sum(r["exit_code"] == 0 for r in report), "of", len(RUNS), "runs finished,", len(todo), "never started")
'''
# build_kernel.sh bundles fss.py, advect.py and train_unet.py into one file; reuse it, then make that file
# read its arguments from the command line instead of the KERNEL_ARGS line
subprocess.run(["./build_kernel.sh", " ".join(runs[0]), slug], check=True)
path = f"kernel/{slug}/{slug}.py"
lines = open(path).read().split("\n")
assert lines[0].startswith("KERNEL_ARGS = ") and "pip" in lines[1]
code = "\n".join(["import sys; KERNEL_ARGS = sys.argv[1:]"] + lines[2:])

# runs share one output folder, so no run's file prefix may start another run's files
tags = [tag_of(r) for r in args]
clash = [(a, b) for a in tags for b in tags if a != b and b.startswith(a + "_")]
assert len(set(tags)) == len(tags) and not clash, f"run file names collide: {clash or tags}"

budget = int(budget_h * 3600)
launcher = (LAUNCHER.replace("__BUDGET__", str(budget)).replace("__HARD__", str(budget + 1500))
            .replace("__HELPERS__", HELPERS).replace("__RUNS__", repr(args)).replace("__CODE__", repr(code)))
open(path, "w").write(launcher)
compile(launcher, path, "exec")
meta = f"kernel/{slug}/kernel-metadata.json"
m = json.load(open(meta))
m.update({"enable_gpu": "false", "enable_tpu": "true", "machine_shape": "TpuV5E8"})
json.dump(m, open(meta, "w"), indent=1)
print(path, f"trains {len(runs)} runs on 8 TPU chips within {budget_h} h:", ", ".join(tags))
