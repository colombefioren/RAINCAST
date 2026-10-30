# Rules that shape this work

Source: the competition pages on Kaggle (Rules, Overview, Data), read again in full on 2026-10-21. The
full text lives on Kaggle. The organisers changed the rules between 20 and 21 October; the section below
explains what changed and what this repository does about it.

## Rule change on external data and pretrained weights

**What the rules said on 20 October.** When the competition opened, rule 2.5 allowed publicly available
external data and pretrained models, provided they were documented. The first fifteen submissions (20 and
21 October) followed that rule: their encoders started from ImageNet weights, the usual starting point for
a segmentation network, and the best of them scored 0.78152 on the public leaderboard (submission 15,
21 October, 07:48 Madagascar time).

**What changed.** When the rules were read again in full on the morning of 21 October, rules 2.4 and 2.5
had been rewritten. Rule 2.4 now limits training, validation and prediction to the supplied competition
data and requires each test sample to be processed on its own from its three rain maps and its timestamp.
Rule 2.5 now forbids external datasets of any kind and any model or weights pretrained on data outside the
competition data. No announcement was posted in the competition Discussion; the change was found by
re-reading the Rules page.

**What was done about it, the same day.**

1. Every training run started after 21 October initialises its network from random weights.
   `src/train_unet.py` builds the encoder with `encoder_weights=None` unless `--pretrained 1` is given;
   the default is 0 and no run list in `runs/` sets it. The flag is kept only so that the early submissions
   can be reproduced.
2. The ImageNet-based submission files prepared for 22 October (`sub16` to `sub19`) were not submitted,
   and the training queue of ImageNet-based runs was emptied. The first from-scratch submission
   (submission 20, 22 October) scored 0.77024, below the ImageNet-based best; the from-scratch blends
   passed it on 23 October (submission 23, 0.78235) and reached 0.78384 on 27 October (submission 40).
3. The coordinate channels of the network, previously read from the latitude and longitude arrays of the
   data files, are generated in code (standardised row and column index, identical values), so that the
   model reads nothing but the three rain maps and the timestamp of a sample.
4. Rain over Africa's public archive, which holds the same product for 2004 to 2019, was never accessed:
   it would be external data under the new rule 2.5. Under the old rule it was the obvious way to a
   higher score, and the decision to leave it alone is recorded in the project journal.
5. Submissions 1 to 15 stay on the leaderboard, since Kaggle keeps every submission, but they are not
   eligible under the current rules and were never candidates for the final selection.
6. The two final submissions were selected by hand on Kaggle on 30 October: submission 40 (public 0.78384)
   and submission 43 (public 0.78357). Both blend only networks trained from random weights after
   21 October (`final/members.md` names each member's training session and seed; the earliest session,
   `molab-set1`, started on 23 October). Leaving the choice to Kaggle would have been wrong on the other
   side: Kaggle picks the best public scores, and the ImageNet-based submissions are among them.

**How to verify.** `final/submission.json` lists the 58 member files; every member comes from a session
in `runs/molab-scratch-set*.txt` or `runs/gpu-*.txt`, whose base lines never set `--pretrained`.
The reproduction notebook trains one member and its twin with the same command and gets the same scores.

## Rules and how the code follows them

| Rule | How this repository follows it |
| --- | --- |
| 2.1 Individual participation, one Kaggle account | All submissions come from the `cocofioren` account. No team. |
| 2.2 At most 5 submissions per day | A submission was made only when the held-out season predicted a gain or answered a question about the leaderboard. |
| 2.3 Deadline 30 October 2026, 09:00 UTC (12:00 Madagascar); private leaderboard released 31 October | The final submissions are selected by hand before the deadline. |
| 2.4 Only the supplied data may be used for training, validation and prediction; models must be trained on the supplied training data | Training uses `train.npz` only. Validation is the December 2022 to March 2023 season of `train.npz`. Nothing else is downloaded or read. |
| 2.4 and Overview: each test sample is processed independently, using only its three input rain maps; its timestamps may be used as input features | Each forecast reads the sample's three rain maps and features of its own timestamp (sine and cosine of the hour, day of the season). The network also gets two fixed coordinate channels generated in code (standardised row and column index, the CoordConv technique), identical for every sample; the latitude and longitude arrays of the data files are not read. Test-time augmentation shifts only the sample's own maps. The decoder reads only the sample's own probability map, with calibration tables fitted on training data. |
| 2.4 No attempt to obtain the hidden targets or to exploit leakage (temporal overlaps, timestamps, matches between samples, or any other source) | 47 of the 200 test samples have a target hour that is an input hour of another test sample (31 at +1h, 24 at +2h, 25 at +3h), so one sample's input contains another sample's target. The code never looks across test samples: each forecast reads one sample's three maps and timestamp, and the decoder reads one sample's probability map. No observation for any test target time is looked up anywhere: the public Rain over Africa archive is never accessed. |
| 2.5 No external datasets, whether public, private, free or paid | None used. |
| 2.5 No models or weights pretrained outside the competition data | Since 21 October every network starts from random weights (see the section above); the final submissions use only such networks. |
| 2.5 Open-source architectures, software and tools allowed if they bring in no external data or pretrained weights; software and significant tools must be documented | Listed in the table below. |
| 2.6 No private sharing of solutions, code, models or predictions; each submission must be the participant's own work | The repository and the Kaggle datasets that carry training results stayed private and shared with nobody until the deadline. The code-review package (the reproduction notebook `indabax-rainfall-reproduction` and the datasets `indabax-rainfall-code` and `indabax-rainfall-members`) is published in the competition's Code section after the deadline, as the organisers asked of the top ten. The AI coding assistant used during development is disclosed below, as rule 2.11 asks. |
| 2.7 Reproducibility: the approach must be documented; potential winners provide full code on request | Fixed seeds, `requirements.txt`, one command per step in `README.md`, every training session in `runs/`, the members of the final submission in `final/members.md`, the reasoning in `docs/APPROACH.md`. The Kaggle notebook built by `tools/build_review_kernel.py` reruns the blend, the decoder fit and the submission step from the saved probability maps and checks the result against the submitted file; it also trains one member and its twin with their exact command. |
| 2.8 Winners release their code under an Open Source Initiative licence | `LICENSE` (MIT). |
| 2.9 Metric: FSS, 1 mm/h, 3x3 neighbourhood, mean of +1h, +2h, +3h; the private leaderboard decides | `src/fss.py` mirrors the metric. Model selection uses the held-out season, not the public leaderboard. |
| 2.10 Code review of the top participants before prizes, including the data-use restrictions and the independent processing of test samples | This repository and the reproduction notebook, with the rows above as the checklist. |
| 2.11 Scientific integrity, responsible use of AI, attribution of data, methods, software and tools | Data: Amell et al. (2025), Rain over Africa. Metric: Roberts and Lean (2008). Methods: references below. Software and AI assistance: table below. |
| Foundational rule 4b: no hand labelling or human prediction of validation or test records | None. |
| Foundational rule 6c: open-source code only under licences that allow commercial use | All libraries below qualify. |

## Software and tools used to generate submissions

| Tool | Role | Licence |
| --- | --- | --- |
| Python 3.12, NumPy, SciPy, pandas | Data handling, metric, submission file | PSF, BSD-3-Clause |
| PyTorch | Training and inference | BSD-3-Clause |
| segmentation-models-pytorch (with timm) | U-Net and U-Net++ architectures, built with random weights | MIT, Apache-2.0 |
| scikit-learn | Isotonic calibration in the decoder | BSD-3-Clause |
| Kaggle Notebooks, molab (marimo) | GPU sessions for training | service terms |
| Kaggle API client | Data download, kernel runs, submissions | Apache-2.0 |
| Claude (Anthropic) | AI assistant used during development; every script was reviewed and run by the participant | service terms |

## References

- Amell, A., Hee, L., Pfreundschuh, S., and Eriksson, P. (2025). Probabilistic near-real-time retrievals
  of Rain over Africa using deep learning. Journal of Geophysical Research: Atmospheres, 130,
  e2025JD044595. https://doi.org/10.1029/2025JD044595
- Roberts, N. M., and Lean, H. W. (2008). Scale-selective verification of rainfall accumulations from
  high-resolution forecasts of convective events. Monthly Weather Review, 136(1), 78-97.
  https://doi.org/10.1175/2007MWR2123.1
