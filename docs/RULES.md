# Rules that shape this work

Source: the competition pages on Kaggle (Rules, Overview, Data), read again in full on 2026-10-21. The
full text lives on Kaggle. The organisers changed the rules between 20 and 21 October; the section below
explains what changed and what this repository does about it.

## Rule change on external data and pretrained weights

On 20 October, rule 2.5 allowed publicly available external data and pretrained models, provided they
were documented. When the rules were read again on 21 October, rule 2.4 limited training, validation and
prediction to the supplied competition data, and rule 2.5 forbade external datasets and any model or
weights pretrained on data outside the competition data. No announcement was posted in the competition
Discussion.

What this repository does about it:

- Submissions 1 to 15 (20 and 21 October) used encoders initialised from ImageNet weights, which the
  rules allowed at that time. Under the current rules they are not eligible as final submissions, and they
  will not be selected.
- From 21 October, `src/train_unet.py` initialises every network from random weights (`--pretrained 0`,
  the default). `--pretrained 1` is kept only to reproduce the early runs.
- The ImageNet-based submission files prepared for 22 October (`sub16` to `sub19`) were not submitted.
- The final submissions are selected by hand on Kaggle, from models trained from scratch. Leaving the
  choice to Kaggle would pick the best public scores, which are ImageNet-based.

## Rules and how the code follows them

| Rule | How this repository follows it |
| --- | --- |
| 2.1 Individual participation, one Kaggle account | All submissions come from the `cocofioren` account. No team. |
| 2.2 At most 5 submissions per day | `docs/JOURNAL.md` tracks the daily count. A submission is made only when validation says it can beat the best score or answers a useful question. |
| 2.3 Deadline 30 October 2026, 09:00 UTC (12:00 Madagascar); private leaderboard released 31 October | The final submissions are selected by hand before the deadline. |
| 2.4 Only the supplied data may be used for training, validation and prediction; models must be trained on the supplied training data | Training uses `train.npz` only. Validation is the December 2022 to March 2023 season of `train.npz`. Nothing else is downloaded or read. |
| 2.4 and Overview: each test sample is processed independently, using only its three input rain maps; its timestamps may be used as input features | Each forecast reads the sample's three rain maps and features of its own timestamp (sine and cosine of the hour, day of the season). The network also gets two fixed coordinate channels generated in code (standardised row and column index, the CoordConv technique), identical for every sample; the latitude and longitude arrays of the data files are not read. Test-time augmentation shifts only the sample's own maps. The decoder reads only the sample's own probability map, with calibration tables fitted on training data. |
| 2.4 No attempt to obtain the hidden targets or to exploit leakage (temporal overlaps, timestamps, matches between samples, or any other source) | 31 test samples overlap in time with other test samples, so one sample's input contains another sample's target. The code never looks across test samples. No observation for any test target time is looked up anywhere: the public Rain over Africa archive is never accessed. |
| 2.5 No external datasets, whether public, private, free or paid | None used. |
| 2.5 No models or weights pretrained outside the competition data | Since 21 October every network starts from random weights (see above). |
| 2.5 Open-source architectures, software and tools allowed if they bring in no external data or pretrained weights; software and significant tools must be documented | Listed in the table below. |
| 2.6 No private sharing of solutions, code, models or predictions; each submission must be the participant's own work | The repository and the Kaggle datasets that carry Colab results are private and shared with nobody. Colombe runs, understands and can explain every script; the AI coding assistant used during development is disclosed below, as rule 2.11 asks. |
| 2.7 Reproducibility: the approach must be documented; potential winners provide full code on request | Fixed seeds, `requirements.txt`, one command per step in `README.md`, every run in `docs/JOURNAL.md`, the reasoning in `docs/APPROACH.md`. |
| 2.8 Winners release their code under an Open Source Initiative licence | `LICENSE` (MIT). |
| 2.9 Metric: FSS, 1 mm/h, 3x3 neighbourhood, mean of +1h, +2h, +3h; the private leaderboard decides | `src/fss.py` mirrors the metric. Model selection uses the held-out season, not the public leaderboard. |
| 2.10 Code review of the top participants before prizes, including the data-use restrictions and the independent processing of test samples | This repository, with the rows above as the checklist. |
| 2.11 Scientific integrity, responsible use of AI, attribution of data, methods, software and tools | Data: Amell et al. (2025), Rain over Africa. Metric: Roberts and Lean (2008). Methods: `docs/RESEARCH.md`. Software and AI assistance: table below. |
| Foundational rule 4b: no hand labelling or human prediction of validation or test records | None. |
| Foundational rule 6c: open-source code only under licences that allow commercial use | All libraries below qualify. |

## Software and tools used to generate submissions

| Tool | Role | Licence |
| --- | --- | --- |
| Python 3.12, NumPy, SciPy, pandas | Data handling, metric, submission file | PSF, BSD-3-Clause |
| PyTorch | Training and inference | BSD-3-Clause |
| segmentation-models-pytorch (with timm) | U-Net and U-Net++ architectures, built with random weights | MIT, Apache-2.0 |
| scikit-learn | Isotonic calibration in the decoder | BSD-3-Clause |
| Kaggle Notebooks, Google Colab and Saturn Cloud | Free T4 GPUs for training | service terms |
| Kaggle API client | Data download, kernel runs, submissions | Apache-2.0 |
| Claude Code (Anthropic) | AI coding assistant used during development | service terms |

## References

- Amell, A., Hee, L., Pfreundschuh, S., and Eriksson, P. (2025). Probabilistic near-real-time retrievals
  of Rain over Africa using deep learning. Journal of Geophysical Research: Atmospheres, 130,
  e2025JD044595. https://doi.org/10.1029/2025JD044595
- Roberts, N. M., and Lean, H. W. (2008). Scale-selective verification of rainfall accumulations from
  high-resolution forecasts of convective events. Monthly Weather Review, 136(1), 78-97.
  https://doi.org/10.1175/2007MWR2123.1
