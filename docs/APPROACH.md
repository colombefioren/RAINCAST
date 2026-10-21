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
- Optimiser AdamW (learning rate 5e-4, weight decay 1e-4), one-cycle learning rate schedule, batch 32,
  6 epochs.
- Regularisation: dropout 0.2 before the output layer and random translations of up to 16 pixels.
  Rotations and flips were tried and hurt, because rain over Madagascar mostly moves in one direction
  and a rotated map teaches the network wrong motion.
- Input upscaling (best single change so far, about +0.012 FSS on validation): the 8-channel input is
  bilinearly enlarged from 128x128 to 256x256 before the encoder, and the 3 output maps are averaged
  back down to 128x128. A ResNet shrinks its input 32 times, so a 128-pixel map becomes a 4x4 grid at
  the bottom of the U-Net; at 256 pixels the grid is 8x8 and the first layers see rain cells that span
  more pixels, closer to the object sizes the ImageNet weights were trained on.
- Epoch averaging: the test probabilities of the last 5 epochs are averaged, which beats the single
  best checkpoint by about +0.004 because each epoch makes slightly different mistakes.
- Thresholds are tuned per lead time on the held-out season (Dec 2022 to Mar 2023), from the averaged
  probabilities.
- For the final submissions the same recipe is retrained on all seasons including December 2023, the
  closest data to the test period, with the thresholds from the matching validation run. Several
  seeds and recipes are averaged before thresholding. With thresholds, training on all seasons was worth only
  about +0.002 on the public leaderboard (submissions 4 and 5). With the decoder it is worth about
  +0.006 (submissions 12 and 13, same model and calibration, only the training seasons differ). The
  full-data models are decoded with the calibration table and mu fitted on their validation twins.

## From probabilities to a rain mask

A single threshold per lead time is not what the metric wants. FSS compares 3x3 neighbourhood
fractions, so in an area where rain is uncertain the best answer is a partial mask whose local density
matches the expected rain coverage. `src/fss_decode.py` does this:

1. Calibrate the blended probabilities with an isotonic fit of observed rain frequency on the
   validation season (one table per lead time).
2. With calibrated p, the expected observed fraction is box(p), and the expected FSS of a mask M is
   2 * sum(M * q) / (sum(box(M)^2) + C), with q = box(box(p)) and C independent of M.
3. Maximise sum(M * q) - mu * sum(box(M)^2) with iterated conditional modes. Pixels on the same
   position of a 3x3 grid cannot interact, so each of the 9 groups is updated in one vectorised step.
   mu is tuned per lead time on validation (theory: half the achievable FSS, about 0.33 to 0.43).

Out-of-sample check (calibration and mu fitted on alternate weeks, scored on the other weeks):
+0.04 at +1h, +0.05 at +2h, +0.065 at +3h. The gain grows with lead time because uncertainty grows.

## What was tried and rejected

Recorded in `docs/JOURNAL.md` as it happens.

## Tools

Python, NumPy, pandas, SciPy, PyTorch, segmentation-models-pytorch, timm, Kaggle CLI. Development was
done with the help of an AI coding assistant (Claude); every script was reviewed and run by the
participant.
