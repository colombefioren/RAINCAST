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
| `src/train_unet.py` | U-Net training: inputs, loss, validation split, threshold tuning, test inference. |
| `src/make_submission.py` | Blend test probability maps, threshold, write `submission.csv`. |
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

## Reproduce a submission

```bash
# 1. baselines and data checks (CPU, ~2 min)
python src/explore.py data

# 2. train with validation on the 2022-2023 season, tune thresholds, predict test (GPU recommended)
python src/train_unet.py --data data --out outputs --encoder resnet34 --epochs 12 --seed 42

# 3. write the submission from one or more probability maps
python src/make_submission.py outputs/submission.csv outputs/resnet34_s42.json outputs/resnet34_s42_testprob.npy

# 4. submit
kaggle competitions submit -c hackathon-IndabaX-MDG-2026 -f outputs/submission.csv -m "resnet34 unet, seed 42"
```

### On Kaggle GPU

```bash
./build_kernel.sh "--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --encoder resnet34 --epochs 12 --threads 4"
cd kernel && kaggle kernels push -p .
kaggle kernels status cocofioren/indabax-rain-unet
kaggle kernels output cocofioren/indabax-rain-unet -p ../outputs
```

## Results

See `docs/JOURNAL.md`. Persistence (copy the last input map) scores 0.604 FSS on the held-out season.
