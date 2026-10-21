"""Bundle up to eight training runs into one Kaggle TPU v5e-8 kernel that trains them at the same time,
one run per TPU chip.

The TPU quota (20 hours a week) is separate from the GPU quota and counts session hours, so eight runs in
one session cost the quota of one. Each run is an independent process that owns one chip: the TPU_*
environment variables make every process a one-chip TPU of its own, so a run that finishes early does not
disturb the others. If that fails, the runs that never trained go one after another on the default device.
Every run gets --xla 1 and writes to /kaggle/tmp/out (the scratch disk); as soon as a run ends, the files
the analysis needs are copied to /kaggle/working, the kernel output, so a session cut short keeps the
finished runs.

Usage: python build_kernel_tpu.py <kernel-slug> "<args for run 1>" ["<args for run 2>" ...]
Runs must differ in seed or model, because they share one output folder.
"""
import json, subprocess, sys

slug, runs = sys.argv[1], sys.argv[2:]
assert 1 <= len(runs) <= 8, __doc__
# build_kernel.sh bundles fss.py + train_unet.py into one file; reuse it, then make that file read its
# arguments from the command line instead of the KERNEL_ARGS line
subprocess.run(["./build_kernel.sh", runs[0], slug], check=True)
path = f"kernel/{slug}/{slug}.py"
lines = open(path).read().split("\n")
assert lines[0].startswith("KERNEL_ARGS = ") and "pip" in lines[1]
code = "\n".join(["import sys; KERNEL_ARGS = sys.argv[1:]"] + lines[2:])
args = [r.replace("--out /kaggle/working", "--out /kaggle/tmp/out").split() + ["--xla", "1"] for r in runs]
assert all("/kaggle/tmp/out" in r for r in args), "every run needs --out /kaggle/working"

launcher = f'''import glob, json, os, shutil, subprocess, sys, threading, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"], check=False)
for c in ["nproc", "free -g | head -2", "df -h /kaggle/tmp | tail -1",  # what the machine offers, for the log
          "pip list 2>/dev/null | grep -i -E '^(torch|torch-xla|torch_xla|libtpu|libtpu-nightly) '"]:
    subprocess.run(c, shell=True)
RUNS = {args!r}
OUT = "/kaggle/tmp/out"
os.makedirs(OUT, exist_ok=True)
open("/kaggle/tmp/train.py", "w").write({code!r})

def tag_of(run):
    """The file prefix train_unet.py gives a run (arch, attention, encoder, seed, full)."""
    v = lambda k, d: run[run.index(k) + 1] if k in run else d
    arch, att = v("--arch", "Unet"), v("--attention", "")
    return (("" if arch == "Unet" else arch.lower() + "_") + (att + "_" if att else "") + v("--encoder", "resnet18")
            + "_s" + v("--seed", "42") + ("_full" if "--full" in run else ""))

def collect(run):
    """Copy what the analysis needs to /kaggle/working (the kernel output, 20 GB at most) as soon as a run ends:
    summary, weights, test probabilities of every epoch, and only the best last-k validation blend (about
    280 MB each). A run that stopped before its summary keeps every validation file it saved."""
    tag = tag_of(run); keep = glob.glob(f"{{OUT}}/{{tag}}.json") + glob.glob(f"{{OUT}}/{{tag}}.pt") + glob.glob(f"{{OUT}}/{{tag}}_*testprob.npy")
    if os.path.exists(f"{{OUT}}/{{tag}}.json"):
        blends = json.load(open(f"{{OUT}}/{{tag}}.json")).get("epoch_blends", {{}})
        if blends:
            k = max(blends, key=lambda b: blends[b]["val_fss"]); keep.append(f"{{OUT}}/{{tag}}_{{k}}_valprob.npy")
            print(f"[{{tag}}] best blend {{k}} {{blends[k]['val_fss']:.4f}}", flush=True)
    else:
        keep += glob.glob(f"{{OUT}}/{{tag}}_*valprob.npy")
    for f in keep:
        if os.path.exists(f): shutil.copy(f, "/kaggle/working")
    print(f"[{{tag}}] {{len(keep)}} files copied to /kaggle/working", flush=True)

def start(i, pinned):
    env = dict(os.environ, PJRT_DEVICE="TPU")
    if pinned:  # every process is a one-chip TPU of its own
        port = 8476 + i
        env.update(TPU_VISIBLE_CHIPS=str(i), TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1", TPU_PROCESS_BOUNDS="1,1,1",
                   TPU_PROCESS_PORT=str(port), TPU_PROCESS_ADDRESSES=f"localhost:{{port}}", CLOUD_TPU_TASK_ID="0",
                   ALLOW_MULTIPLE_LIBTPU_LOAD="1")
    p = subprocess.Popen([sys.executable, "-u", "/kaggle/tmp/train.py"] + RUNS[i], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    def relay():  # prefix each line with the run number so the kernel log tells the runs apart
        for line in p.stdout:
            if "Warning" not in line: print(f"[run{{i + 1}}] " + line, end="", flush=True)
        p.wait(); print(f"[run{{i + 1}}] exit code {{p.returncode}}", flush=True); collect(RUNS[i])
    t = threading.Thread(target=relay); t.start()
    return p, t

t0 = time.time()
jobs = [start(i, True) for i in range(len(RUNS))]
codes = [p.wait() for p, t in jobs]; [t.join() for p, t in jobs]
print("exit codes", codes, f"after {{time.time() - t0:.0f}}s", flush=True)
# fallback: if pinning chips failed (runs died before finishing their first epoch), train those runs one
# after another on the default device, in the order given
first_epoch = lambda run: glob.glob(f"{{OUT}}/{{tag_of(run)}}_ep0_*") or glob.glob(f"{{OUT}}/{{tag_of(run)}}.pt")
retry = [i for i, c in enumerate(codes) if c != 0 and not first_epoch(RUNS[i])]
if retry:
    print("runs that never trained with one chip each:", [i + 1 for i in retry], "- training them one after another", flush=True)
    for i in retry:
        p, t = start(i, False); p.wait(); t.join()
'''
open(path, "w").write(launcher)
compile(launcher, path, "exec")
meta = f"kernel/{slug}/kernel-metadata.json"
m = json.load(open(meta))
m.update({"enable_gpu": "false", "enable_tpu": "true", "machine_shape": "TpuV5E8"})
json.dump(m, open(meta, "w"), indent=1)
print(path, "is a", len(runs), "run TPU launcher,", len(launcher.splitlines()), "lines")
