# INDABAX-RAINFALL

Rainfall nowcasting over Madagascar for the IndabaX Madagascar 2026 hackathon on Kaggle
(`hackathon-IndabaX-MDG-2026`). Given three hourly rain maps (t-2h, t-1h, t), predict the rain maps
at t+1h, t+2h and t+3h. Evaluated with the Fractions Skill Score (FSS) at a 1 mm/h threshold and a
3x3 pixel neighbourhood, averaged over the three lead times.

## Layout

| Path | What it is |
| --- | --- |
| `src/fss.py` | Reimplementation of the competition metric (pooled and per-sample variants). |
| `src/explore.py` | Data statistics and non-learned baselines (persistence, smoothing, dilation). |
| `src/advect.py` | Optical-flow advection baseline and optional advected input channels. |
| `src/train_unet.py` | U-Net training: inputs, losses, augmentation, input upscaling, validation, per-epoch probabilities, test inference. |
| `src/lastk_testprob.py` | Average the test probabilities of a run's last k epochs. |
| `src/blend.py` | Blend validation probabilities of several runs, tune thresholds, and with `--decode 1` fit the decoder. |
| `src/fss_decode.py` | Expected-FSS decoder: calibrated probabilities to the rain mask that maximises the expected FSS. |
| `src/eval_decode.py` | Out-of-sample check of the decoder against thresholds (alternate weeks). |
| `src/eval_tta.py` | Check of shift test-time augmentation on a saved checkpoint. |
| `src/make_submission.py` | Blend test probabilities, threshold or decode, write `submission.csv`. |
| `build_kernel.sh` | Bundle `fss.py` + `train_unet.py` into one file for a Kaggle GPU kernel. |
| `kernel/` | Kaggle kernel script and metadata, pushed with `kaggle kernels push`. |
| `docs/APPROACH.md` | Why the model is built this way, in plain words. |
| `docs/JOURNAL.md` | Experiment log: every run, its validation score and leaderboard score. |
| `docs/RULES.md` | Competition rules that constrain the work, and how the code respects them. |
| `docs/RESEARCH.md` | Literature review of techniques for this task, ranked, with what was implemented. |
| `docs/HANDBOOK.md` | Machine learning, Kaggle and this project explained for the participant, with talking points for verification. |
| `data/` | Competition files (git-ignored): `train.npz`, `test.npz`, `sample_submission.csv`. |
| `outputs/` | Model weights, probability maps, submissions (git-ignored). |

## Setup

```bash
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
kaggle competitions download -c hackathon-IndabaX-MDG-2026 -f train.npz -p data
kaggle competitions download -c hackathon-IndabaX-MDG-2026 -f test.npz -p data
kaggle competitions download -c hackathon-IndabaX-MDG-2026 -f sample_submission.csv -p data
```

The Kaggle CLI reads its token from `~/.kaggle/access_token` (Kaggle settings, API Tokens).

## Reproduce the best submission (public 0.77372, submission 11)

Eight U-Nets are trained on Kaggle GPUs with the 2022-2023 season held out. Each run saves the test
probabilities of every epoch and the validation probabilities of its last-k epoch averages.

| Kernel | Arguments after the common ones | Last k | Validation FSS |
| --- | --- | --- | --- |
| `indabax-unet-r34-up3` | `--encoder resnet34 --upscale 3 --seed 32` | 5 | 0.7081 |
| `indabax-unet-r34-up3-s35` | `--encoder resnet34 --upscale 3 --seed 35 --tta 1` | 4 | 0.7066 |
| `indabax-unet-r34-up4` | `--encoder resnet34 --upscale 4 --seed 34 --tta 1` | 4 | 0.7073 |
| `indabax-unet-r34-up2` | `--encoder resnet34 --upscale 2 --seed 31` | 5 | 0.7032 |
| `indabax-unet-r50-up2` | `--encoder resnet50 --upscale 2 --seed 51` | 4 | 0.7040 |
| `indabax-unet-r34-up3-loss` | `--encoder resnet34 --upscale 3 --csi_w 1 --ms_fss 1 --pos_w 2 --mixup 0.5 --wet_sampling 1 --seed 36 --tta 1` | 6 | 0.7060 |
| `indabax-unet-r50-up3` | `--encoder resnet50 --upscale 3 --seed 52 --tta 1` | 4 | 0.7082 |
| `indabax-unet-effb3-up3` | `--encoder efficientnet-b3 --upscale 3 --bs 16 --seed 23 --tta 1` | 5 | 0.7057 |

Common arguments: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working
--threads 4 --save_each --epochs 6 --lr 5e-4 --aug 2 --dropout 0.2`.

```bash
# 1. build and run one kernel per row (2 GPU kernels at a time on Kaggle)
./build_kernel.sh "<common arguments> <row arguments>" indabax-unet-r34-up3
kaggle kernels push -p kernel/indabax-unet-r34-up3
kaggle kernels status cocofioren/indabax-unet-r34-up3
kaggle kernels output cocofioren/indabax-unet-r34-up3 -p outputs/kup3

# 2. average the last k epochs of test probabilities, matching the validation file
python src/lastk_testprob.py outputs/kup3/resnet34_s32 5

# 3. blend the eight runs, calibrate and fit the decoder on the held-out season (run from src/)
cd src
python blend.py --data ../data --out ../outputs/blend18dec.json --decode 1 --mu 0.425,0.375,0.35 \
  ../outputs/kup3/resnet34_s32_last5_valprob.npy ../outputs/kup3s35/resnet34_s35_last4_valprob.npy \
  ../outputs/kup4/resnet34_s34_last4_valprob.npy ../outputs/kup2/resnet34_s31_last5_valprob.npy \
  ../outputs/kr50up2/resnet50_s51_last4_valprob.npy ../outputs/kup3loss/resnet34_s36_last6_valprob.npy \
  ../outputs/kr50up3/resnet50_s52_last4_valprob.npy ../outputs/keffb3up3/efficientnet-b3_s23_last5_valprob.npy

# 4. decode the blended test probabilities and write the submission
python make_submission.py ../outputs/sub11.csv ../outputs/blend18dec.json ../outputs/blend18dec_testprob.npy
kaggle competitions submit -c hackathon-IndabaX-MDG-2026 -f ../outputs/sub11.csv -m "sub11"
```

Without `--mu`, `blend.py` searches mu per lead time (slower, same values). The runs `r34-up2` and
`r34-up3` were built before `--tta` existed and use single predictions.

## Results

See `docs/JOURNAL.md`. Persistence (copy the last input map) scores 0.604 FSS on the held-out season.
The best single U-Net scores 0.708 with thresholds; eight U-Nets blended and decoded score 0.7655 on
the held-out season and 0.77372 on the public leaderboard.
