"""Bundle two training runs into one Kaggle kernel that trains them at the same time, one per GPU.

Kaggle's T4 machine has two GPUs and the weekly quota counts session hours, so two runs in one session
cost the same quota as one run. With a single GPU the runs go one after the other.

Usage: python build_kernel_pair.py <kernel-slug> "<args for run 1>" "<args for run 2>"
The two runs must differ in seed or model, because both write to /kaggle/working.
"""
import json, os, subprocess, sys

slug, runs = sys.argv[1], sys.argv[2:4]
assert len(runs) == 2, __doc__
# build_kernel.sh bundles fss.py + train_unet.py into one file; reuse it, then make that file read its
# arguments from the command line instead of the KERNEL_ARGS line
subprocess.run(["./build_kernel.sh", runs[0], slug], check=True)
path = f"kernel/{slug}/{slug}.py"
lines = open(path).read().split("\n")
assert lines[0].startswith("KERNEL_ARGS = ") and "pip" in lines[1]
code = "\n".join(["import sys; KERNEL_ARGS = sys.argv[1:]"] + lines[2:])

launcher = f'''import os, subprocess, sys, threading, torch
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"], check=False)
RUNS = {[r.split() for r in runs]!r}
os.makedirs("/kaggle/tmp", exist_ok=True)
open("/kaggle/tmp/train.py", "w").write({code!r})
n_gpu = torch.cuda.device_count()
print("GPUs:", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)], flush=True)

def start(i, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    p = subprocess.Popen([sys.executable, "-u", "/kaggle/tmp/train.py"] + RUNS[i], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    def relay():  # prefix each line with the run number so the kernel log shows both runs
        for line in p.stdout:
            if "Warning" not in line: print(f"[run{{i + 1}}] " + line, end="", flush=True)
    t = threading.Thread(target=relay); t.start()
    return p, t

if n_gpu >= 2:
    jobs = [start(0, 0), start(1, 1)]
    codes = [p.wait() for p, t in jobs]; [t.join() for p, t in jobs]
else:
    print("only one GPU: the two runs go one after the other", flush=True)
    codes = []
    for i in range(2):
        p, t = start(i, 0); codes.append(p.wait()); t.join()
print("exit codes", codes, flush=True)
'''
open(path, "w").write(launcher)
compile(launcher, path, "exec")
print(path, "is a two-run launcher,", len(launcher.splitlines()), "lines")
