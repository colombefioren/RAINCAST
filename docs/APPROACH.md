# Approach

## The metric decides the model

FSS only looks at whether each pixel is above 1 mm/h, then compares the fraction of rainy pixels in
every 3x3 window between forecast and observation. Two consequences:

1. Predicting exact rain amounts is wasted effort. The task is binary segmentation: rain / no rain at
   each of three lead times. The submission writes 10 mm/h where the model says rain and 0 elsewhere.
2. FSS rewards forecasts whose rainy area is about the same size as the observed area. A model that
   outputs probabilities needs a decision threshold per lead time, tuned on validation data to maximise
   FSS. The best threshold is usually below 0.5 because rain covers only about 5 % of pixels.

## Data

- 10,862 training samples, one per hour, December to March, seasons 2019-20 through 2023-24 (the
  last season only has December 2023). Consecutive samples overlap: sample i's targets are sample
  i+1's inputs. Random splits would leak, so validation is a whole season (December 2022 to March 2023,
  2,887 samples).
- 200 test samples from January to March 2024, at random hours.
- Rain fraction above 1 mm/h: about 4.9 % of pixels.

## Baselines (held-out season, pooled FSS)

| Forecast | FSS |
| --- | --- |
| Persistence: copy the map at t to all three lead times | 0.604 |
| Mean of the three input maps | 0.520 |
| All zeros | 0.000 |
| All rain | 0.099 |

Persistence is strong at +1h (0.76) and weak at +3h (0.46). The model has to learn motion and
growth/decay of rain cells.

## Model

- U-Net (`segmentation-models-pytorch`) with an ImageNet-pretrained ResNet encoder.
- Inputs, 8 channels: the three rain maps as log1p(rain)/2, latitude and longitude grids
  (standardised), sin and cos of the hour, and the position within the rainy season.
- Outputs, 3 channels: probability of rain above 1 mm/h at +1h, +2h, +3h.
- Loss: binary cross-entropy with positive weight 8 (rain is rare) plus a soft FSS loss, which is the
  metric itself computed on probabilities instead of binary masks, so the network optimises what is
  scored.
- Optimiser AdamW, one-cycle learning rate schedule, batch 32.
- Validation each epoch; the checkpoint with the best FSS is kept. Thresholds are then tuned per lead
  time on the validation probabilities.
- For the final submission the same recipe is retrained on all seasons including December 2023, the
  closest data to the test period, and several seeds are averaged before thresholding.

## What was tried and rejected

Recorded in `docs/JOURNAL.md` as it happens.

## Tools

Python, NumPy, pandas, SciPy, PyTorch, segmentation-models-pytorch, timm, Kaggle CLI. Development was
done with the help of an AI coding assistant (Claude); every script was reviewed and run by the
participant.
