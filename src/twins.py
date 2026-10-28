"""Print the test-probability file of the full-data twin of each validation member given on the command line.

A validation member is trained with the last season held out; its twin has the same settings, seed + 100 and
--full (every season, December 2023 included), so it is the model that predicts the test set. The twin's file
is looked up in every output folder next to the member's (twins trained in a later session land in that
session's folder); when a twin was fetched twice, the newer file wins.

Usage (from src/): python twins.py ../outputs/<folder>/<run>_s<seed>_last<k>_valprob.npy ...
Output: the matching ../outputs/<folder>/<run>_s<seed + 100>_full_last<k>_testprob.npy files, in the same order,
ready for make_submission.py."""
import glob, os, re, sys

out = []
for v in sys.argv[1:]:
    m = re.match(r"(.*)/([^/]+)_s(\d+)_last\d_valprob\.npy$", v)
    if not m: sys.exit(f"not a validation member: {v}")
    folder, name, seed = m.group(1), m.group(2), int(m.group(3))
    hits = glob.glob(f"{os.path.dirname(folder)}/*/{name}_s{seed + 100}_full_last[0-9]_testprob.npy")
    if not hits: sys.exit(f"no twin for {v}")
    out.append(max(hits, key=os.path.getmtime))
print(" ".join(out))
