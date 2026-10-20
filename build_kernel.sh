#!/usr/bin/env bash
# Usage: ./build_kernel.sh "<args for train_unet.py>" [kernel-slug]
# Bundles src/fss.py + src/train_unet.py into kernel/<slug>/<slug>.py with kernel-metadata.json.
set -e
cd "$(dirname "$0")"
ARGS="$1"; SLUG="${2:-indabax-rain-unet}"
mkdir -p "kernel/$SLUG"
python3 - "$ARGS" "$SLUG" <<'PY'
import ast, sys
args, slug = sys.argv[1], sys.argv[2]
def strip_doc(src):
    tree = ast.parse(src)
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        return "".join(src.splitlines(keepends=True)[tree.body[0].end_lineno:])
    return src
fss = strip_doc(open("src/fss.py").read())
tr = strip_doc(open("src/train_unet.py").read())
tr = tr.replace("from fss import fss_pooled, fss_per_sample, report\n", "")
tr = tr.replace("a = p.parse_args()\n", "a = p.parse_args(KERNEL_ARGS)\n")
out = (f"KERNEL_ARGS = '''{args}'''.split()\n"
       "import subprocess, sys; subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch'], check=False)\n"
       + fss + "\n" + tr)
open(f"kernel/{slug}/{slug}.py", "w").write(out)
ast.parse(out); print(f"kernel/{slug}/{slug}.py parses ok,", len(out.splitlines()), "lines")
import json
json.dump({"id": f"cocofioren/{slug}", "title": slug, "code_file": f"{slug}.py", "language": "python",
           "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
           "dataset_sources": [], "competition_sources": ["hackathon-IndabaX-MDG-2026"], "kernel_sources": []},
          open(f"kernel/{slug}/kernel-metadata.json", "w"), indent=1)
PY
