# Rules that shape this work

Source: the competition Rules page, read on 2026-10-20. Full text lives on Kaggle.

| Rule | Consequence for this repo |
| --- | --- |
| Individual participation, one Kaggle account | All submissions come from the `cocofioren` account. No team. |
| Max 5 submissions per day | `docs/JOURNAL.md` tracks the daily count. Submit only after a validation score justifies it. |
| Deadline 30 October 2026, 09:00 UTC (12:00 Madagascar) | Final two submissions selected on Kaggle before that time. |
| Do not reconstruct, infer or exploit the hidden test targets | 31 test samples overlap in time with other test samples, so one sample's input contains another sample's target. The code never looks across test samples: each prediction uses only that sample's own three input maps. |
| External data, pretrained models and open-source software allowed if freely accessible and documented | ImageNet-pretrained encoders from `segmentation-models-pytorch` (MIT licence). No external datasets. Documented in `docs/APPROACH.md`. |
| No private sharing of code or predictions | This repository is private and shared with nobody. |
| Each submission must be the participant's own work | Colombe runs, understands and can explain every script. An AI coding assistant (Claude) was used as a tool during development; this is stated in the write-up for transparency under rule 2.11 on responsible AI use. |
| Reproducibility: organisers may request full code, architecture, hyperparameters, dependencies | Fixed seeds, `requirements.txt`, one command per step in `README.md`, every run logged in `docs/JOURNAL.md`. |
| Winner licence: Open Source Initiative approved | Code can be released under MIT if required. |
| Metric: FSS, 1 mm/h, 3x3, mean of +1h, +2h, +3h; private leaderboard decides | `src/fss.py` mirrors the metric. Model selection uses the held-out season, not the public leaderboard. |
