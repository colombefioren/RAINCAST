"""Run a run-queue kernel script (build_kernel_queue.py --gpu) on a cloud notebook sandbox whose
RTX PRO 6000 Blackwell GPU (96 GB) trains several runs side by side, outside Kaggle's
GPU quota.

Usage: python tools/build_cloud_job.py <kernel-slug> <results-dataset-name> [jobs-dataset-name]
e.g.   python tools/build_kernel_queue.py indabax-cloud-set1 runs/cloud-scratch-set1.txt 10.25 --gpu --runs-per-gpu 3
       python tools/build_cloud_job.py indabax-cloud-set1 indabax-cloud-set1-results

It publishes the private Kaggle dataset cocofioren/indabax-cloud-jobs (the kernel script as job.py, job.json,
cloud_worker.py and src/fetch_runs.py) and writes the sandbox notebook cloud/indabax-cloud-worker.py. In the sandbox:
create a notebook with its three cells, choose the GPU (resources menu > GPUs > RTX Pro 6000, Save and
restart), paste the Kaggle API token into the password box and click Start training. The token stays in the
session: it is never saved in the notebook. The notebook downloads the four files one by one: since 25 October
Kaggle answers 404 to whole-archive downloads of most new versions of this private dataset. cloud_worker.py then
- downloads train.npz and test.npz to /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 (the sandbox runs
  notebooks as root, so the Kaggle paths of the kernel script work unchanged);
- runs job.py, relaying its log without the per-iteration lines;
- after each finished run (at most every 30 minutes) and at the end, compacts /kaggle/working with
  src/fetch_runs.py (each validation run's best last-k blend as uint8 with the mean test probabilities of the
  same epochs, and each full-data twin's mean test probabilities) and uploads it as a new version of the
  private dataset cocofioren/<results-dataset-name>.
Fetch the results with: kaggle datasets download -d cocofioren/<results-dataset-name> -p outputs/<dir> --unzip

Sandbox limits (October 2026): a notebook runs at most 12 hours, hence a budget of about 10 hours, and stops
after 90 idle minutes, so its browser tab stays open. Anyone who has the sandbox notebook's URL can open it (it is
not listed anywhere), so the notebook holds no competition code: the code comes from the private dataset.
"""
import json, os, shutil, subprocess, sys, tempfile

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
JOBS = "cocofioren/" + (sys.argv[3] if len(sys.argv) > 3 else "indabax-cloud-jobs")
KAGGLE = os.environ.get("KAGGLE", "kaggle")

WORKER = r'''"""Sandbox side of build_cloud_job.py: download the data, run job.py, upload compact results after each run."""
import glob, json, os, re, shutil, signal, subprocess, sys, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
JOB = json.load(open(f"{HERE}/job.json"))
RESULTS = f"cocofioren/{JOB['results']}"
DATA, WORK, COMPACT = "/kaggle/input/competitions/hackathon-IndabaX-MDG-2026", "/kaggle/working", "/kaggle/compact"
os.makedirs(DATA, exist_ok=True); os.makedirs(WORK, exist_ok=True)

def exists(dataset):
    r = subprocess.run(["kaggle", "datasets", "status", dataset], capture_output=True, text=True)
    return "ready" in r.stdout.lower()

def publish(folder, results, note):
    """Compact a job's output folder with fetch_runs.py and publish it as a version of its results dataset."""
    compact = folder + "-compact"; shutil.rmtree(compact, ignore_errors=True)
    r = subprocess.run([sys.executable, f"{HERE}/fetch_runs.py", folder, compact], capture_output=True, text=True)
    print(r.stdout + r.stderr, end="", flush=True)
    if r.returncode != 0: return print("compaction failed, nothing uploaded", flush=True)
    json.dump({"title": results.split("/")[1], "id": results, "licenses": [{"name": "CC0-1.0"}]}, open(f"{compact}/dataset-metadata.json", "w"))
    cmd = ["version", "-p", compact, "-m", note] if exists(results) else ["create", "-p", compact]
    rc = subprocess.run(["kaggle", "datasets"] + cmd + ["-q"]).returncode
    print(f"upload ({note}):", "done" if rc == 0 else f"failed, exit code {rc}", sorted(os.listdir(compact)), flush=True)

def stop_previous():
    """Start training again in the same cloud session (after interrupting the cell) replaces the job still running
    there: its worker, queue script and training processes are killed, its files are moved out of /kaggle/working,
    so this job's uploads hold only its own runs, and the runs it finished after its last upload are published to
    its own results dataset (named in the job.json the worker leaves in /kaggle/working)."""
    keep = {os.getpid(), os.getppid()}
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try: cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="ignore")
        except OSError: continue
        if int(pid) not in keep and any(k in cmd for k in ("cloud_worker.py", "/kaggle/jobs/job.py", "/kaggle/tmp/train.py")):
            try: os.kill(int(pid), signal.SIGKILL); print("stopped process", pid, "of the previous job", flush=True)
            except OSError: pass
    if os.listdir(WORK):
        old = f"/kaggle/previous-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.move(WORK, old); os.makedirs(WORK)
        print("previous job's files moved to", old, flush=True)
        if os.path.exists(f"{old}/job.json") and os.path.exists(f"{old}/session.json"):
            publish(old, f"cocofioren/{json.load(open(f'{old}/job.json'))['results']}", "runs finished before a restart")
stop_previous()
shutil.copy(f"{HERE}/job.json", WORK)  # tells a later restart which results dataset this job's leftovers belong to
subprocess.run(["nvidia-smi", "-L"])
for f in ["train.npz", "test.npz"]:
    if not os.path.exists(f"{DATA}/{f}"):
        subprocess.run(["kaggle", "competitions", "download", "-c", "hackathon-IndabaX-MDG-2026", "-f", f, "-p", DATA, "-q"], check=True)
for z in glob.glob(f"{DATA}/*.zip"):
    shutil.unpack_archive(z, DATA); os.remove(z)
print("data:", sorted(os.listdir(DATA)), "job:", JOB, flush=True)

lock = threading.Lock()
def upload(note):
    """Compact the kernel output with fetch_runs.py, then publish it as the results dataset (private)."""
    with lock:
        if os.path.exists(f"{WORK}/session.json"): publish(WORK, RESULTS, note)

def watch(p):
    """Upload after each finished run, at most every 30 minutes: a stopped session keeps the runs already done."""
    seen, last = 0.0, 0.0
    while p.poll() is None:
        time.sleep(60)
        log = f"{WORK}/session.json"
        if os.path.exists(log) and os.path.getmtime(log) > seen and time.time() - last > 1800:
            seen, last = os.path.getmtime(log), time.time()
            runs = json.load(open(log))["runs"]
            upload(f"{sum(r['exit_code'] == 0 for r in runs)} runs finished")

p = subprocess.Popen([sys.executable, "-u", f"{HERE}/job.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
threading.Thread(target=watch, args=(p,), daemon=True).start()
for line in p.stdout:
    if not re.search(r"\] ep\d+ it\d+/", line): print(line, end="", flush=True)
p.wait()
upload("final")
print("worker done, exit code", p.returncode, flush=True)
'''

NOTEBOOK_TEMPLATE = r'''import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    token = mo.ui.text(kind="password", label="Kaggle API token", full_width=True)
    start = mo.ui.run_button(label="Start training")
    mo.vstack([mo.md("Paste your Kaggle API token, then click **Start training**. The token stays in this session: it is never saved in the notebook."), token, start])
    return start, token


@app.cell
def _(mo, start, token):
    import os, pathlib, subprocess
    mo.stop(not (start.value and token.value.strip()), mo.md("Waiting for the token and the Start training button."))
    kaggle_dir = pathlib.Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    (kaggle_dir / "access_token").write_text(token.value.strip())
    os.chmod(kaggle_dir / "access_token", 0o600)
    cmd = "python -m pip -q install -U kaggle && mkdir -p /kaggle/jobs && for f in job.py job.json cloud_worker.py fetch_runs.py; do kaggle datasets download -d JOBS_DATASET -f $f -p /kaggle/jobs -o -q || exit 1; done && python -u /kaggle/jobs/cloud_worker.py"
    proc = subprocess.Popen(cmd, shell=True, env=dict(os.environ, KAGGLE_API_TOKEN=token.value.strip()), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
    return


if __name__ == "__main__":
    app.run()
'''

slug, results = sys.argv[1], sys.argv[2]
script = f"kernel/{slug}/{slug}.py"
src = open(script).read()
assert "\nGPU = True" in src and "RUNS_PER_GPU" in src, f"{script}: build it with tools/build_kernel_queue.py --gpu first"
NOTEBOOK = NOTEBOOK_TEMPLATE.replace("JOBS_DATASET", JOBS)
compile(WORKER, "cloud_worker.py", "exec"); compile(NOTEBOOK, "indabax-cloud-worker.py", "exec")
os.makedirs("cloud", exist_ok=True)
open("cloud/indabax-cloud-worker.py", "w").write(NOTEBOOK)
with tempfile.TemporaryDirectory() as d:
    shutil.copy(script, f"{d}/job.py"); shutil.copy("src/fetch_runs.py", d)
    open(f"{d}/cloud_worker.py", "w").write(WORKER)
    json.dump({"slug": slug, "results": results}, open(f"{d}/job.json", "w"))
    json.dump({"title": JOBS.split("/")[1], "id": JOBS, "licenses": [{"name": "CC0-1.0"}]}, open(f"{d}/dataset-metadata.json", "w"))
    exists = "ready" in subprocess.run([KAGGLE, "datasets", "status", JOBS], capture_output=True, text=True).stdout.lower()
    cmd = ["version", "-p", d, "-m", f"{slug} to {results}"] if exists else ["create", "-p", d]  # create: private by default
    subprocess.run([KAGGLE, "datasets"] + cmd + ["-q"], check=True)
print(f"{JOBS} now runs {slug}, results to cocofioren/{results}; notebook: cloud/indabax-cloud-worker.py")
