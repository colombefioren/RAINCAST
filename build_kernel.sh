#!/usr/bin/env bash
# Usage: ./build_kernel.sh "<args for train_unet.py>"  -> kernel/indabax-rain-unet.py
set -e
cd "$(dirname "$0")"
ARGS="$1"
python3 - "$ARGS" <<'PY'
import ast, sys
args = sys.argv[1]
def strip_doc(src):
    tree = ast.parse(src)
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        lines = src.splitlines(keepends=True)
        return "".join(lines[tree.body[0].end_lineno:])
    return src
fss = strip_doc(open("src/fss.py").read())
tr = strip_doc(open("src/train_unet.py").read())
tr = tr.replace("from fss import fss_pooled, fss_per_sample, report\n", "")
tr = tr.replace("a = p.parse_args()\n", "a = p.parse_args(KERNEL_ARGS)\n")
out = (f"KERNEL_ARGS = '''{args}'''.split()\n"
       "import subprocess, sys; subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch'], check=False)\n"
       + fss + "\n" + tr)
open("kernel/indabax-rain-unet.py", "w").write(out)
ast.parse(out); print("kernel script parses ok,", len(out.splitlines()), "lines")
PY
cat > kernel/kernel-metadata.json <<JSON
{
  "id": "cocofioren/indabax-rain-unet",
  "title": "indabax-rain-unet",
  "code_file": "indabax-rain-unet.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "dataset_sources": [],
  "competition_sources": ["hackathon-IndabaX-MDG-2026"],
  "kernel_sources": []
}
JSON
