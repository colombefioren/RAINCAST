"""Turn a built Kaggle kernel (kernel/<slug>/<slug>.py) into a Google Colab notebook colab/<slug>.ipynb.

Why: Colab's free GPU saves Kaggle GPU quota. The notebook runs the exact same training script.
- Credentials: the Kaggle API token is read from a Colab secret named KAGGLE_API_TOKEN, which the
  account owner adds once (key icon in the left sidebar). It is never written into the notebook.
- Data: downloaded from the competition with the Kaggle CLI.
- Results: uploaded as a PRIVATE Kaggle dataset cocofioren/indabax-colab-<slug>, so they can be fetched
  locally with `kaggle datasets download` (datasets use no GPU quota).

Usage: python build_colab.py indabax-unetpp-r34-up3-long-s41
"""
import json, sys, re

slug = sys.argv[1]
src = open(f"kernel/{slug}/{slug}.py").read()
src = src.replace("--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working",
                  "--data /content/data --out /content/out")
assert "/kaggle/" not in src.split("\n", 1)[0], "kernel args still point at /kaggle"
dataset = ("indabax-colab-" + slug.removeprefix("indabax-"))[:50]

def md(text): return {"cell_type": "markdown", "metadata": {}, "source": text}
def code(text): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}

cells = [
    md(f"# {slug} on Colab\n\n"
       "Same training script as the Kaggle kernel of the same name, run on Colab's GPU to save Kaggle quota.\n\n"
       "1. Runtime > Change runtime type > **T4 GPU**.\n"
       "2. One time only: add a secret named `KAGGLE_API_TOKEN` (key icon, left sidebar) holding your Kaggle API token, "
       "and allow this notebook to read it.\n"
       "3. Runtime > **Run all**. Results go to the private Kaggle dataset "
       f"`cocofioren/{dataset}`.\n"),
    code("!nvidia-smi -L\n!pip -q install -U kaggle segmentation-models-pytorch"),
    code("import os, pathlib\nfrom google.colab import userdata\n"
         "tok = userdata.get('KAGGLE_API_TOKEN').strip()\n"
         "d = pathlib.Path.home() / '.kaggle'; d.mkdir(exist_ok=True)\n"
         "(d / 'access_token').write_text(tok); os.chmod(d / 'access_token', 0o600)\n"
         "os.environ['KAGGLE_API_TOKEN'] = tok\n"
         "print('token loaded from Colab secrets')"),
    code("!mkdir -p /content/data /content/out\n"
         "!kaggle competitions download -c hackathon-IndabaX-MDG-2026 -f train.npz -p /content/data\n"
         "!kaggle competitions download -c hackathon-IndabaX-MDG-2026 -f test.npz -p /content/data\n"
         "!cd /content/data && for z in *.zip; do [ -e \"$z\" ] && unzip -o -q \"$z\" && rm \"$z\"; done; ls -la /content/data"),
    code("%%writefile /content/train.py\n" + src),
    code("!python -u /content/train.py 2>&1 | grep -v Warning"),
    code("import glob, json, os, shutil\n"
         "up = '/content/upload'; os.makedirs(up, exist_ok=True)\n"
         "for f in glob.glob('/content/out/*_ep*_testprob.npy') + glob.glob('/content/out/*_testprob.npy') + glob.glob('/content/out/*.json'):\n"
         "    shutil.copy(f, up)\n"
         "for jf in glob.glob('/content/out/*.json'):\n"
         "    blends = json.load(open(jf)).get('epoch_blends', {})\n"
         "    if blends:  # keep only the best last-k validation file (each is about 280 MB)\n"
         "        k = max(blends, key=lambda b: blends[b]['val_fss'])\n"
         "        tag = os.path.basename(jf)[:-5]\n"
         "        shutil.copy(f'/content/out/{tag}_{k}_valprob.npy', up); print('best epoch blend', k, blends[k]['val_fss'])\n"
         f"json.dump({{'title': '{dataset}', 'id': 'cocofioren/{dataset}', 'licenses': [{{'name': 'CC0-1.0'}}]}}, open(f'{{up}}/dataset-metadata.json', 'w'))\n"
         "print(sorted(os.listdir(up)))\n"
         "r = os.system(f'kaggle datasets create -p {up} -q')  # private by default\n"
         "if r != 0: os.system(f'kaggle datasets version -p {up} -m rerun -q')"),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                    "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:  # notebook format stores source as a list of lines
    c["source"] = c["source"].splitlines(keepends=True)
json.dump(nb, open(f"colab/{slug}.ipynb", "w"), indent=1)
print(f"colab/{slug}.ipynb written, results dataset cocofioren/{dataset}")
