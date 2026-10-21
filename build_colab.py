"""Run Kaggle kernel scripts on Google Colab's free GPU to save Kaggle GPU quota.

Three commands:
  python build_colab.py notebook <kernel-slug>     one notebook for one run: colab/<slug>.ipynb
  python build_colab.py worker                     the reusable worker notebook: colab/indabax-colab-worker.ipynb
  python build_colab.py jobs <slug> [<slug> ...]   publish the worker's job queue (private Kaggle dataset)

How the worker works: the job queue is a private Kaggle dataset, cocofioren/indabax-colab-jobs, holding one
training script per job (the bundled kernel script with Colab paths) and jobs.json (the order). The worker
notebook downloads the queue, runs every job whose results are not on Kaggle yet, and uploads each job's
results as a private Kaggle dataset cocofioren/indabax-colab-<slug>. Fetch them locally with
`kaggle datasets download -d cocofioren/indabax-colab-<slug> -p outputs/<dir> --unzip`. Datasets use no GPU quota.

The queue also holds worker.py, the same worker as plain Python, for any Linux machine with a GPU and the
environment variable KAGGLE_API_TOKEN, such as a Saturn Cloud job. Its whole start command is SATURN_COMMAND.

Credentials: the Kaggle API token is read from a Colab secret named KAGGLE_API_TOKEN, which the account owner
adds once (key icon in the left sidebar), or on Saturn Cloud from a secret attached to the job as the
environment variable KAGGLE_API_TOKEN. It is never written into a notebook, a command or a dataset.
"""
import json, os, subprocess, sys, tempfile

KAGGLE_DATA = "--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working"
COLAB_DATA = "--data /content/data --out /content/out"
JOBS = "cocofioren/indabax-colab-jobs"


def colab_script(slug):
    """The bundled kernel script of kernel/<slug>/ with its data and output paths moved to Colab."""
    src = open(f"kernel/{slug}/{slug}.py").read()
    assert KAGGLE_DATA in src.split("\n", 1)[0], f"{slug}: unexpected kernel arguments"
    return src.replace(KAGGLE_DATA, COLAB_DATA, 1)


def results_name(slug):
    return ("indabax-colab-" + slug.removeprefix("indabax-"))[:50]


def md(text): return {"cell_type": "markdown", "metadata": {}, "source": text}
def code(text): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


SETUP = [
    code("!nvidia-smi -L\n!pip -q install -U kaggle segmentation-models-pytorch"),
    code("import os, pathlib\nfrom google.colab import userdata\nROOT = '/content'\n"
         "tok = userdata.get('KAGGLE_API_TOKEN').strip()\n"
         "d = pathlib.Path.home() / '.kaggle'; d.mkdir(exist_ok=True)\n"
         "(d / 'access_token').write_text(tok); os.chmod(d / 'access_token', 0o600)\n"
         "os.environ['KAGGLE_API_TOKEN'] = tok\n"
         "print('token loaded from Colab secrets')"),
    code("!mkdir -p /content/data /content/out\n"
         "!kaggle competitions download -c hackathon-IndabaX-MDG-2026 -f train.npz -p /content/data\n"
         "!kaggle competitions download -c hackathon-IndabaX-MDG-2026 -f test.npz -p /content/data\n"
         "!cd /content/data && for z in *.zip; do [ -e \"$z\" ] && unzip -o -q \"$z\" && rm \"$z\"; done; ls -la /content/data"),
]

# worker.py (Saturn Cloud or any Linux GPU machine): the token comes from the environment and everything
# lives under ~/indabax. The start command installs the packages and fetches the queue that holds worker.py.
SATURN_COMMAND = ("pip -q install -U kaggle segmentation-models-pytorch && mkdir -p ~/.kaggle ~/indabax && "
                  "printf %s \"$KAGGLE_API_TOKEN\" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token && "
                  f"kaggle datasets download -d {JOBS} -p ~/indabax/queue --unzip -o -q && python -u ~/indabax/queue/worker.py")
SETUP_PY = '''import glob, os, subprocess, zipfile
ROOT = os.path.expanduser("~/indabax")
assert os.environ.get("KAGGLE_API_TOKEN", "").strip(), "the Kaggle API token must be in the environment variable KAGGLE_API_TOKEN"
os.makedirs(f"{ROOT}/data", exist_ok=True); os.makedirs(f"{ROOT}/out", exist_ok=True)
for f in ["train.npz", "test.npz"]:
    if not os.path.exists(f"{ROOT}/data/{f}"):
        subprocess.run(f"kaggle competitions download -c hackathon-IndabaX-MDG-2026 -f {f} -p {ROOT}/data", shell=True, check=True)
for z in glob.glob(f"{ROOT}/data/*.zip"):
    zipfile.ZipFile(z).extractall(f"{ROOT}/data"); os.remove(z)
print(sorted(os.listdir(f"{ROOT}/data")), flush=True)
'''

# Shared by the notebooks and worker.py: copy a finished run's test probabilities, its json and its best last-k
# validation file into a folder, then create (or version) the private results dataset.
UPLOAD_FN = '''import glob, json, os, shutil, subprocess
def upload(out, name):
    up = f"{ROOT}/upload/{name}"; shutil.rmtree(up, ignore_errors=True); os.makedirs(up)
    for f in glob.glob(f"{out}/*_testprob.npy") + glob.glob(f"{out}/*.json"):
        shutil.copy(f, up)
    jsons = glob.glob(f"{out}/*.json")
    for jf in jsons:
        blends = json.load(open(jf)).get("epoch_blends", {})
        if blends:  # keep only the best last-k validation file (each is about 280 MB)
            k = max(blends, key=lambda b: blends[b]["val_fss"]); tag = os.path.basename(jf)[:-5]
            shutil.copy(f"{out}/{tag}_{k}_valprob.npy", up); print("best epoch blend", k, blends[k]["val_fss"])
    if not jsons:  # the run stopped before its summary: keep every validation file and the weights
        for f in glob.glob(f"{out}/*_valprob.npy") + glob.glob(f"{out}/*.pt"):
            shutil.copy(f, up)
    json.dump({"title": name, "id": f"cocofioren/{name}", "licenses": [{"name": "CC0-1.0"}]}, open(f"{up}/dataset-metadata.json", "w"))
    print(sorted(os.listdir(up)), flush=True)
    if subprocess.run(f"kaggle datasets create -p {up} -q", shell=True).returncode != 0:  # private by default
        subprocess.run(f"kaggle datasets version -p {up} -m rerun -q", shell=True)
    shutil.rmtree(up, ignore_errors=True)
'''

WORKER_LOOP = '''import torch
assert torch.cuda.is_available(), "No GPU: on Colab, Runtime > Change runtime type > T4 GPU; on Saturn, a T4 instance size"
JOBS = "%s"
def results_name(slug): return ("indabax-colab-" + slug.removeprefix("indabax-"))[:50]
def uploaded(slug):
    r = subprocess.run(f"kaggle datasets status cocofioren/{results_name(slug)}", shell=True, capture_output=True, text=True)
    return "ready" in r.stdout.lower()
done, failed = set(), set()
while True:
    shutil.rmtree(f"{ROOT}/jobs", ignore_errors=True)
    if subprocess.run(f"kaggle datasets download -d {JOBS} -p {ROOT}/jobs --unzip -o -q", shell=True).returncode != 0:
        print("could not download the job queue"); break
    queue = json.load(open(f"{ROOT}/jobs/jobs.json"))
    todo = [s for s in queue if s not in done and s not in failed and not uploaded(s)]
    print("jobs left:", todo, flush=True)
    if not todo:
        print("all jobs done: stop the runtime (Colab: Runtime > Disconnect and delete runtime; Saturn: Stop)"); break
    slug = todo[0]; out = f"{ROOT}/out/{slug}"; shutil.rmtree(out, ignore_errors=True)
    script = open(f"{ROOT}/jobs/{slug}.py").read()  # queued scripts use Colab paths
    script = script.replace("--data /content/data", f"--data {ROOT}/data", 1).replace("--out /content/out", f"--out {out}", 1)
    open(f"{ROOT}/job.py", "w").write(script)
    print("=" * 20, "running", slug, flush=True)
    rc = subprocess.run(f"set -o pipefail; python -u {ROOT}/job.py 2>&1 | grep --line-buffered -v Warning",
                        shell=True, executable="bash").returncode
    if not glob.glob(f"{out}/*_testprob.npy"):
        print("job failed:", slug, "exit code", rc, flush=True); failed.add(slug); continue
    if rc != 0: print("job stopped early:", slug, "exit code", rc, "- uploading what it saved", flush=True)
    upload(out, results_name(slug)); done.add(slug); shutil.rmtree(out, ignore_errors=True)
''' % JOBS


def write_nb(path, cells):
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                        "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 0}
    for c in nb["cells"]:  # notebook format stores source as a list of lines
        c["source"] = c["source"].splitlines(keepends=True)
    json.dump(nb, open(path, "w"), indent=1)
    print(path, "written")


def notebook(slug):
    name = results_name(slug)
    write_nb(f"colab/{slug}.ipynb", [
        md(f"# {slug} on Colab\n\n"
           "Same training script as the Kaggle kernel of the same name, run on Colab's GPU to save Kaggle quota.\n\n"
           "1. Runtime > Change runtime type > **T4 GPU**.\n"
           "2. One time only: add a secret named `KAGGLE_API_TOKEN` (key icon, left sidebar) holding your Kaggle API token, "
           "and allow this notebook to read it.\n"
           f"3. Runtime > **Run all**. Results go to the private Kaggle dataset `cocofioren/{name}`.\n"),
        *SETUP,
        code("%%writefile /content/train.py\n" + colab_script(slug)),
        code("!set -o pipefail; python -u /content/train.py 2>&1 | grep --line-buffered -v Warning"),
        code(UPLOAD_FN + f"upload('/content/out', '{name}')"),
    ])


def worker():
    write_nb("colab/indabax-colab-worker.ipynb", [
        md("# IndabaX Colab worker\n\n"
           "Trains the queued jobs one after the other on Colab's GPU, to save Kaggle GPU quota. "
           f"The queue is the private Kaggle dataset `{JOBS}`; each job's results go to a private dataset "
           "`cocofioren/indabax-colab-<job>`. Jobs whose results already exist are skipped, so running it again "
           "after a disconnect resumes the queue.\n\n"
           "1. Runtime > Change runtime type > **T4 GPU**.\n"
           "2. The `KAGGLE_API_TOKEN` secret (key icon, left sidebar) must allow this notebook to read it.\n"
           "3. Runtime > **Run all**, and keep the tab open.\n"),
        *SETUP,
        code(UPLOAD_FN + "\n" + WORKER_LOOP),
    ])


def jobs(slugs):
    with tempfile.TemporaryDirectory() as d:
        for s in slugs:
            open(f"{d}/{s}.py", "w").write(colab_script(s))
        json.dump(slugs, open(f"{d}/jobs.json", "w"))
        open(f"{d}/worker.py", "w").write(SETUP_PY + UPLOAD_FN + WORKER_LOOP)
        json.dump({"title": "indabax-colab-jobs", "id": JOBS, "licenses": [{"name": "CC0-1.0"}]},
                  open(f"{d}/dataset-metadata.json", "w"))
        exists = subprocess.run(["kaggle", "datasets", "status", JOBS], capture_output=True, text=True)
        cmd = (["kaggle", "datasets", "version", "-p", d, "-m", "queue: " + " ".join(slugs)[:200]]
               if "ready" in exists.stdout.lower() else ["kaggle", "datasets", "create", "-p", d])
        subprocess.run(cmd + ["-q"], check=True)
    print("job queue published:", slugs)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "notebook": notebook(args[0])
    elif cmd == "worker": worker()
    elif cmd == "jobs": jobs(args)
    else: sys.exit(__doc__)
