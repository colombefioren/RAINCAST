# Approach

## The metric decides the model

FSS only looks at whether each pixel is above 1 mm/h, then compares the fraction of rainy pixels in
every 3x3 window between forecast and observation. Two consequences:

1. Predicting exact rain amounts is wasted effort. The task is binary segmentation: rain / no rain at
   each of three lead times. The submission writes 10 mm/h where the model says rain and 0 elsewhere.
2. FSS rewards forecasts whose rainy area is about the same size as the observed area, and in uncertain
   areas it rewards a partial mask whose local density matches the chance of rain. A model that outputs
   probabilities therefore needs a decoder tuned on validation data, not a fixed threshold.

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
growth or decay of rain cells.

## One network

- U-Net++ (`segmentation-models-pytorch`) with a ResNet-34 encoder, trained from random weights. The rules
  forbid weights pretrained outside the competition data; the submissions made before 21 October used
  ImageNet initialisation, which the rules allowed at that time (see `RULES.md`), and were not selected.
- Inputs, 8 channels: the three rain maps as log1p(rain)/2, two fixed coordinate channels (standardised
  row and column index, the CoordConv technique; they equal the standardised latitude and longitude
  grids), sin and cos of the hour, and the position within the rainy season. The out-of-fold climatology
  members add three channels: how often each pixel had rain above 1 mm/h at the hour of day of each lead
  time, counted over the training rows and smoothed over one hour and 5x5 pixels. Training samples read a
  table counted without their own season, as validation and test samples do; otherwise the table contains
  the sample's own target and the network trusts it more than it deserves on a new season.
- Outputs, 3 channels: probability of rain above 1 mm/h at +1h, +2h, +3h.
- Loss: binary cross-entropy with positive weight 8 (rain is rare) plus a soft FSS loss, which is the
  metric itself computed on probabilities instead of binary masks, so the network optimises what is
  scored. The strongest members add a soft CSI (intersection over union) term with weight 1 to 3: it
  raised a single run from about 0.7125 to about 0.7145 on the held-out season.
- Optimiser AdamW (learning rate 1e-3, weight decay 1e-4), one-cycle schedule, batch 16, 14 epochs
  (12 to 24 for the earliest members), mixed precision.
- Regularisation: dropout 0.2 before the output layer and random translations of up to 16 pixels.
  Rotations and flips hurt, because rain over Madagascar mostly moves in one direction and a rotated map
  teaches the network wrong motion.
- Input upscaling (about +0.012 FSS for a single run): the input is bilinearly enlarged from 128x128 to
  384x384 before the encoder, and the output maps are averaged back down. A ResNet shrinks its input 32
  times, so a 128-pixel map becomes a 4x4 grid at the bottom of the network; at 384 pixels the grid is
  12x12 and the deep layers keep spatial detail.
- Epoch averaging: the probabilities of the last 6 epochs are averaged, which beats the single best
  checkpoint by about +0.004 because each epoch makes slightly different mistakes.

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
   mu is tuned per lead time on validation (theory: half the achievable FSS; 0.425, 0.375 and 0.35).

Out-of-sample check (calibration and mu fitted on alternate weeks, scored on the other weeks):
+0.04 at +1h, +0.05 at +2h, +0.065 at +3h over tuned thresholds. The gain grows with lead time because
uncertainty grows.

## Blend

Fifty-eight validation runs are averaged with weights by family: 1 for the plain and climatology members
of the first sessions, 2 for the out-of-fold climatology members, 3 for the CSI-loss members
(`../final/members.md`). The blend gains about +0.05 over the best single run before decoding; with
decoding it reaches 0.7699 on the held-out season. Beyond about forty members, more seeds of the same
family tie; the last gains came from new families (climatology inputs, then the CSI loss).

## Test side

For the test set, each validation member is replaced by its full-data twin: the same settings, seed + 100,
trained on all seasons including December 2023, the closest data to the 2024 test period. The twins' test
probabilities are blended with the same weights and decoded with the calibration and mu fitted on the
validation members. Training on all seasons was worth about +0.006 on the public leaderboard with the
decoder (two submissions that differed only in the training seasons).

## Tried and not kept

Rotation and flip augmentation, zoom augmentation, input-frame dropout, temporal mixup, wet-sample
oversampling, multi-scale FSS terms, an auxiliary regression head, advected input channels, test-time
augmentation by shifts, dilation of the decoded masks, calibration per dry-to-wet quartile, a
neighbourhood-aware calibration (`src/context_calib.py`, +0.001 on validation but no gain on the public
board), encoders other than ResNet-34 (ResNet-18, ResNet-50, EfficientNet-B2 tie or lose), weight decay
0.05 and dropout 0.4 on long schedules. Everything was measured on the held-out season; the public
leaderboard served only to confirm gains.
