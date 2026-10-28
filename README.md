# INDABAX-RAINFALL

Rainfall nowcasting over Madagascar for the Kaggle hackathon *Hackathon - IndabaX Madagascar 2026*
(`hackathon-IndabaX-MDG-2026`). Given three hourly rain maps (t-2h, t-1h, t) of 128x128 pixels, predict the
rain maps at t+1h, t+2h and t+3h. The score is the Fractions Skill Score (FSS) at a 1 mm/h threshold with a
3x3 pixel neighbourhood, averaged over the three lead times.

The final submission blends 58 U-Net++ networks trained from random weights on the competition data only,
then calibrates the blended probabilities and decodes them into the rain mask with the highest expected FSS.
It scores 0.7699 on the held-out season (December 2022 to March 2023) and 0.78384 on the public leaderboard.

## Method

**The metric sets the task.** FSS only asks whether each pixel is above 1 mm/h, then compares the fraction
of rainy pixels in every 3x3 window. So the model predicts, for each lead time, the probability that a pixel
exceeds 1 mm/h, and the submission writes 10 mm/h where the decoded mask says rain and 0 elsewhere.

**One network** (`src/train_unet.py`):

- U-Net++ decoder with a ResNet-34 encoder (`segmentation-models-pytorch`), random initialisation.
- Inputs: the three rain maps as log1p(rain)/2, two fixed coordinate channels (standardised row and column
  index), sine and cosine of the hour, the position in the rainy season. Most members add three climatology
  channels: how often each pixel had rain at the hour of each lead time, counted on the training rows and,
  for the training samples, without the sample's own season (out of fold).
- The input is enlarged 3x (bilinear) before the encoder and the logits are averaged back, so the encoder's
  deep layers keep spatial detail.
- Loss: binary cross-entropy with positive weight 8 plus a soft FSS term (the metric on probabilities); the
  strongest members add a soft CSI (intersection over union) term.
- AdamW, learning rate 1e-3 with a one-cycle schedule, batch 16, 14 epochs, dropout 0.2 before the output
  layer, random translations of up to 16 pixels, mixed precision.
- The probabilities of the last 6 epochs are averaged: each epoch makes slightly different mistakes.

**Validation.** The 2022-2023 season (2,887 hourly samples) is held out of training; everything that is
tuned (thresholds, calibration, decoder) is tuned there. Consecutive samples overlap in time, so a random
split would leak.

**Blend and decoder** (`src/blend.py`, `src/fss_decode.py`):

1. The validation probabilities of the 58 members are averaged with weights 1, 2 or 3 by family
   (`final/members.md`).
2. An isotonic fit per lead time maps the blended probability to the observed rain frequency.
3. With calibrated probabilities p, the expected FSS of a mask M is a ratio whose numerator is linear in M,
   with q = box(box(p)). The decoder maximises sum(M q) - mu sum(box(M)^2) by iterated conditional modes
   (mu = 0.425, 0.375, 0.35 per lead time), which puts partial masks where rain is uncertain instead of a
   hard threshold. It reads only the sample's own probability map.

**Test side.** Every validation member has a full-data twin: the same settings and seed + 100, trained on
all seasons including December 2023, the closest data to the 2024 test period. The twins' test probabilities
are blended with the same weights and decoded with the calibration and mu fitted on the validation members.

## Results on the held-out season

| Forecast | FSS |
| --- | --- |
| Persistence (copy the map at t) | 0.604 |
| Best single network, thresholds per lead time | 0.7154 |
| 58-member blend, thresholds per lead time | 0.7187 |
| 58-member blend, calibrated and decoded (final) | 0.7699 |

Public leaderboard: 0.78384 (200 test samples, scored on a public subset).

## Rules

Each test sample is processed on its own from its three rain maps and its own timestamp; no external data,
no pretrained weights, no look-up across samples. `docs/RULES.md` goes through the competition rules one by
one and says how the code follows each, including the software used and the AI coding assistance during
development.

## Layout

| Path | Content |
| --- | --- |
| `src/train_unet.py` | Training, validation, per-epoch probabilities and test inference of one network. |
| `src/fss.py` | The competition metric (pooled over samples, and per sample). |
| `src/advect.py` | Optical-flow baseline; optional advected input channels (not used by the final members). |
| `src/blend.py` | Weighted blend of validation probabilities, thresholds, calibration and decoder fit. |
| `src/fss_decode.py` | Isotonic calibration and the expected-FSS decoder. |
| `src/context_calib.py` | Neighbourhood-aware calibration, an option of `blend.py` not used by the final submission. |
| `src/eval_decode.py` | Out-of-sample check of the decoder against thresholds (alternate weeks). |
| `src/twins.py` | Pairs each validation member with its full-data twin's test probabilities. |
| `src/make_submission.py` | Blends test probabilities, decodes them and writes the submission file. |
| `src/fetch_runs.py` | Compacts a training session's output: best last-k blend of each run, mean test probabilities. |
| `src/lastk_testprob.py` | Averages the test probabilities of a run's last k epochs. |
| `tools/build_kernel.sh` | Bundles the training code into one Kaggle kernel script. |
| `tools/build_kernel_queue.py` | Turns a run list into one kernel that trains the runs side by side on GPUs or TPU chips. |
| `tools/build_kernel_pair.py` | Two runs in one Kaggle kernel, one per GPU. |
| `tools/build_molab.py` | Runs a run-queue kernel on molab (marimo's cloud notebooks) with results uploaded to a private Kaggle dataset. |
| `runs/` | The run lists of every training session, with the reason for each group of runs. |
| `final/members.md` | The 58 members of the final submission: settings, sessions, scores. |
| `final/submission.json` | Member files, weights, calibration tables and mu of the final submission. |
| `docs/APPROACH.md` | Why the model is built this way. |
| `docs/RULES.md` | Competition rules and how the code follows them; software and references. |
| `data/`, `outputs/`, `kernel/` | Competition files, model outputs and generated kernels (git-ignored). |

## Setup

```bash
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
kaggle competitions download -c hackathon-IndabaX-MDG-2026 -f train.npz -p data
kaggle competitions download -c hackathon-IndabaX-MDG-2026 -f test.npz -p data
```

The Kaggle CLI reads its token from `~/.kaggle/access_token`.

## Reproducing the final submission

1. **Train the members and their twins.** Each session is a run list in `runs/` (the sessions of the final
   members are named in `final/members.md`; twins are the `--full` lines with seed + 100). A list becomes one
   kernel script that trains its runs side by side within a time budget:

   ```bash
   python tools/build_kernel_queue.py indabax-molab-set9 runs/molab-scratch-set9.txt 10.25 --gpu --runs-per-gpu 2
   ```

   On Kaggle: `kaggle kernels push -p kernel/indabax-molab-set9 -t 43200`, then
   `python src/fetch_runs.py cocofioren/indabax-molab-set9 outputs/molab-set9` once the session ends.
   On molab: `python tools/build_molab.py indabax-molab-set9 indabax-molab-set9-results` publishes the job
   to a private Kaggle dataset; the notebook it writes runs the job and uploads the compacted results, fetched
   with `kaggle datasets download -d cocofioren/indabax-molab-set9-results -p outputs/molab-set9 --unzip`.
   A single run trains with `python src/train_unet.py <base line> <own settings>`.

2. **Blend and fit the decoder** on the held-out season (from `src/`):

   ```bash
   cd src
   M=$(python -c "import json; c = json.load(open('../final/submission.json')); print(' '.join('../' + m for m in c['members']))")
   W=$(python -c "import json; print(','.join(map(str, json.load(open('../final/submission.json'))['weights'])))")
   python blend.py --data ../data --out ../outputs/final.json --decode 1 --mu 0.425,0.375,0.35 --weights $W $M
   ```

   `../outputs/final.json` then holds the thresholds, calibration tables and mu of `final/submission.json`.

3. **Decode the twins' test probabilities** and write the submission:

   ```bash
   python make_submission.py ../outputs/submission.csv ../outputs/final.json $(python twins.py $M)
   kaggle competitions submit -c hackathon-IndabaX-MDG-2026 -f ../outputs/submission.csv -m "final"
   ```

Seeds are fixed; GPU kernels and mixed precision keep the runs reproducible to a few 1e-4 of FSS.

## References

- Roberts, N. M., and Lean, H. W. (2008). Scale-selective verification of rainfall accumulations from
  high-resolution forecasts of convective events. Monthly Weather Review, 136(1), 78-97.
- Amell, A., Hee, L., Pfreundschuh, S., and Eriksson, P. (2025). Probabilistic near-real-time retrievals of
  Rain over Africa using deep learning. Journal of Geophysical Research: Atmospheres, 130, e2025JD044595.
- Zhou, Z., et al. (2018). UNet++: A nested U-Net architecture for medical image segmentation. DLMIA 2018.
- Iakubovskii, P. (2019). Segmentation Models Pytorch. https://github.com/qubvel/segmentation_models.pytorch
