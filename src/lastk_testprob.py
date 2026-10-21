"""Average the test probabilities of the last k epochs of a run, matching its *_last{k}_valprob.npy.

Usage: python src/lastk_testprob.py outputs/kup3/resnet34_s32 5
Reads outputs/kup3/resnet34_s32_ep*_testprob.npy, writes outputs/kup3/resnet34_s32_last5_testprob.npy.
"""
import sys, glob, re, numpy as np

prefix, k = sys.argv[1], int(sys.argv[2])
files = sorted(glob.glob(f"{prefix}_ep*_testprob.npy"), key=lambda f: int(re.search(r"_ep(\d+)_", f).group(1)))
assert len(files) >= k, f"only {len(files)} epoch files for {prefix}"
mean = np.mean([np.load(f).astype(np.float32) for f in files[-k:]], axis=0)
np.save(f"{prefix}_last{k}_testprob.npy", mean.astype(np.float16))
print(f"averaged epochs {[re.search(r'_ep(\d+)_', f).group(1) for f in files[-k:]]} into {prefix}_last{k}_testprob.npy")
