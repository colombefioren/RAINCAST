# Experiment journal

One line per run. Validation = pooled FSS on the December 2022 to March 2023 season.
Public leaderboard = score shown on Kaggle. Private leaderboard decides the final rank.

| Date | Run | Validation FSS | Public LB | Notes |
| --- | --- | --- | --- | --- |
| 2026-10-20 | persistence baseline | 0.604 | not submitted | +1h 0.76, +2h 0.59, +3h 0.46 |
| 2026-10-20 | persistence, 3x3 smoothing, threshold 1.0 | 0.602 | not submitted | smoothing does not help |
| 2026-10-20 | resnet34 U-Net, 12 epochs, lr 1e-3, no aug (kernel v1) | 0.685 (best epoch 1, thr 0.525 all leads) | 0.69473 | overfits: val 0.685 at epoch 1, falls to 0.640 by epoch 11 while train loss keeps dropping. +1h 0.79, +2h 0.67, +3h 0.59 |
| 2026-10-20 | resnet34, 6 epochs, lr 5e-4, rot90+flip aug, dropout 0.2 (kernel v2) | 0.672 (epoch 6, still rising) | | no overfitting any more, but rotations slow learning: worse than run 1 |
| 2026-10-20 | resnet18, same + stride 2 (kernel -b v1) | 0.662 (epoch 5 of 6) | | rotations + half data: slow, underfits |
| 2026-10-20 | resnet18, 4 epochs, rot90+flip aug, dropout 0.2, advection inputs, stride 2, laptop CPU | 0.664 (tuned thr 0.625/0.575/0.550) | | worse than run 1: rotation augmentation likely breaks the geographic motion prior; stride 2 halves data |
| 2026-10-20 | blend 2:1 of run 1 and the CPU run | 0.686 | submitted | +0.001 only |
| 2026-10-20 | resnet34, 3 epochs, no aug, dropout 0.2, advection inputs, laptop CPU | 0.653 after epoch 1, process died in epoch 2 | | killed (memory) while other jobs ran; the GPU kernel `indabax-unet-r34-advect` covers the same question |
| 2026-10-20 | resnet34, 5 epochs, lr 5e-4, no aug, dropout 0.2, advection (`indabax-unet-r34-advect`) | 0.680 single, 0.688 last-4-epoch average | | advection inputs: no measurable gain over the next row |
| 2026-10-20 | resnet34, 5 epochs, lr 5e-4, no aug, dropout 0.2 (`indabax-unet-r34-dropout`) | 0.678 single, 0.686 last-4-epoch average | | dropout alone removes the overfitting; peak now at epoch 4 |
| 2026-10-20 | blend of run 1 + r34-advect last4 + r34-dropout last4 + kernel v2 epoch 6 | 0.694 (thr 0.575/0.55/0.55) | 0.70364 | +0.009 over best single |
| 2026-10-20 | 4-run blend + 3x3 or 5x5 probability smoothing before threshold | 0.692 / 0.688 | | smoothing hurts, dropped |
| 2026-10-20 | resnet34, translation aug, dropout, advection, 6 epochs (`indabax-unet-r34-shift`) | 0.686 single, 0.691 last-2 average | | best single recipe so far; translation augmentation gives about +0.003 |
| 2026-10-20 | 5-run blend (adds efficientnet-b3) | 0.695 | not submitted | +0.001, within noise |
| 2026-10-20 | resnet34, translation aug, 2x input upscaling (`indabax-unet-r34-up2`) | 0.6996 single (epoch 3), 0.7032 last-5 average (thr 0.55/0.525/0.50) | | best recipe so far: +0.012 over the shift recipe; the pretrained encoder sees finer structure when the 128x128 map is bilinearly upscaled to 256x256 and the output pooled back |
| 2026-10-20 | efficientnet-b3, translation aug, dropout, advection, 6 epochs (`indabax-unet-effb3-shift`) | 0.684 single, 0.687 last-4 average | | same plateau as resnet34; useful as blend diversity |
| 2026-10-20 | 4-run blend + per-lead 3x3 dilation sweep | 0.694 | | dilation never selected, no gain |
| 2026-10-20 | resnet34, soft CSI + multi-scale FSS loss, pos_w 2, mixup 0.5, wet sampling, translation aug (`indabax-unet-r34-loss`) | 0.679 single, 0.691 last-5 average | | same plateau as the shift recipe; thresholds drop to 0.45/0.40/0.375 (better calibrated), adds blend diversity |
| 2026-10-20 | 6-run blend (adds r34-shift) | 0.6955 | not submitted | plateau: every recipe lands at 0.69 single, 0.695 blended |
| 2026-10-20 | 7-run blend (adds r34-loss) | 0.6966 | not submitted | |
| 2026-10-20 | resnet34 shift recipe trained on ALL seasons incl. Dec 2023, 5 epochs, last-3 average (`indabax-unet-r34-shift-full-s101`) | no validation possible | 0.70337 | tests whether full-data training carries to the test period; thresholds 0.575/0.55/0.55 from the validation shift run |
| 2026-10-20 | loss recipe trained on all seasons, 6 epochs (`indabax-unet-r34-loss-full-s102`) | no validation possible | | second full-data model; test probabilities kept per epoch for blending |
| 2026-10-20 | 8-run blend, upscale run weighted 3 | 0.7012 | not submitted | old runs drag the upscale run down |
| 2026-10-20 | shift last3 + loss last5 + upscale last5, weights 1:1:4 / 1:1:8 | 0.7037 / 0.7039 | not submitted | barely above the upscale run alone (0.7032): the older recipes add almost nothing now, so more upscale-style models are the way forward |
| 2026-10-20 | full-data shift s101 last3 + full-data loss s102 last5, thresholds 0.525/0.475/0.45 from the matching validation pair (0.6957) | no validation possible | 0.70517 (new best) | submission 5: checks whether blending two full-data models carries to the public leaderboard before the final full-data upscale ensemble |
| 2026-10-20 | up2 recipe on all seasons, 6 epochs, seed 103 (`indabax-unet-r34-up2-full-s103`) | no validation possible | | full-data version of the best recipe |
| 2026-10-20 | resnet34, translation aug, 3x input upscaling, seed 32 (`indabax-unet-r34-up3`) | 0.7015 single, 0.7081 last-5 average | | 3x beats 2x by +0.005: more upscaling still helps, 4x launched (`indabax-unet-r34-up4`) |

| 2026-10-20 | resnet50, 2x upscale, validation run (`indabax-unet-r50-up2`) | 0.6993 single, 0.7040 last-4 average | | same level as resnet34 at 2x: encoder size is not the bottleneck, but it adds blend diversity |
| 2026-10-20 | translation test-time augmentation on the upscale checkpoint (`src/eval_tta.py`, shifts 4 and 8 px, every 6th validation sample) | 0.6994 without, 0.7006 with | | small but consistent on all leads; built into `train_unet.py` as `--tta 1` for new runs |
| 2026-10-20 | expected-FSS decoder instead of per-pixel thresholds (`src/fss_decode.py`, checked by `src/eval_decode.py`) on the 2x upscale run | 0.6998 thresholds, 0.7546 decoder (calibration and mu fitted on alternate weeks, scored on the other weeks) | | the largest gain of the project, +0.055 out of sample; confirmed with an independent scipy implementation of FSS, pooled and per sample |
| 2026-10-20 | shift last3 + loss last5 + up2 last5 (1:1:8), decoded (`outputs/blend10dec.json`, mu 0.425/0.375/0.35) | 0.7579 (in-sample calibration) | ready as `sub06` | predicted public about 0.768 |
| 2026-10-20 | up3 last5 + up2 last5 + r50 last4, thresholds, weights 1:1:1 / 2:1:1 | 0.7090 / 0.7095 | | best thresholded blend; decoded below |
| 2026-10-20 | same blend 2:1:1, decoded (`outputs/blend12dec.json`, mu 0.425/0.375/0.35) | 0.7624 (in-sample calibration) | ready as `sub07` | predicted public about 0.772 |
| 2026-10-20 | up3, up2, r50 + older loss and shift models (2:1:1:0.5:0.5), decoded | 0.7619 | | older non-upscaled models no longer help: 0.7624 without them |
| 2026-10-20 | decoder convergence check: 3, 6, 12 sweeps, start from threshold mask or empty mask | identical to 4 decimals | | the optimiser is not a lever; better probabilities are |
| 2026-10-20 | resnet34, 2x upscale, plain cross-entropy (pos_w 1, no soft FSS), TTA (`indabax-unet-r34-up2-bce`) | 0.6908 single, 0.6927 last-2 average (thresholds 0.35/0.30/0.275) | | worse with thresholds |
| 2026-10-20 | decoded single models, same calibration and mu search: cross-entropy run last3 vs soft-FSS up2 run last5 | 0.7458 vs 0.7568 | | the soft-FSS loss also gives better probabilities for the decoder; plain cross-entropy dropped |
| 2026-10-21 | resnet34, 4x upscale, TTA (`indabax-unet-r34-up4`) | 0.7030 single, 0.7073 last-4 average | | same as 3x at twice the cost: 3x is the sweet spot |
| 2026-10-21 | resnet34, 3x upscale, TTA, second seed (`indabax-unet-r34-up3-s35`) | 0.7024 single, 0.7066 last-4 average | | seed spread about 0.0015 |
| 2026-10-21 | blend of up3 s32, up3 s35, up4, up2, r50-up2, equal weights, thresholds | 0.7108 | | best thresholded blend |
| 2026-10-21 | same five-model blend, decoded (`outputs/blend15dec.json`, mu 0.425/0.375/0.35) | 0.7641 (in-sample calibration) | ready as `sub08` | predicted public about 0.774; candidate for the first submission of 21 October |
| 2026-10-21 | resnet50 3x and multi-scale-loss 3x kernels (`indabax-unet-r50-up3`, `indabax-unet-r34-up3-loss`) | running | | diversity for the ensemble |

From now on only validation runs: full-data training adds about +0.002, while validation runs give the probabilities needed to tune thresholds, blends and the decoder.

Kaggle kernel naming: the first three kernels are `indabax-rain-unet`, `indabax-rain-unet-b`, `indabax-rain-unet-c`. From the fourth on, the scheme is `indabax-unet-<encoder>-<variant>`.

Submission policy from 2026-10-20 evening (Colombe's decision): submit when the submission either predicts a public score above the current leader or gives information that helps get there (checking that a validation gain carries to the leaderboard). Offset validation to public so far: +0.010.

Reading of submissions 4 and 5 (corrected): the full-data shift model (0.70337) should be compared with the same recipe trained without the validation season (validation 0.691, so about 0.701 expected on the public board), not with run 1. Full-data training is therefore worth only about +0.002. Submission 5 confirms it: the matching validation pair scored 0.6957, which predicts about 0.7057, and the full-data pair scored 0.70517. Conclusion: retraining on all seasons is safe but is not where the missing points are. The validation offset of +0.010 still holds.

## Submissions used per day (limit 5)

| Date | Count |
| --- | --- |
| 2026-10-20 | 5 |
