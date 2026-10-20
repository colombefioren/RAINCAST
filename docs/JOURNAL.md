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
| 2026-10-20 | up2 recipe on all seasons, 6 epochs, seed 103 (`indabax-unet-r34-up2-full-s103`) | no validation possible | | full-data version of the best recipe |
| 2026-10-20 | resnet34, translation aug, 3x input upscaling, seed 32 (`indabax-unet-r34-up3`) | running | | does more upscaling keep helping? |

Kaggle kernel naming: the first three kernels are `indabax-rain-unet`, `indabax-rain-unet-b`, `indabax-rain-unet-c`. From the fourth on, the scheme is `indabax-unet-<encoder>-<variant>`.

Submission policy from 2026-10-20 evening (Colombe's decision): submit when the submission either predicts a public score above the current leader or gives information that helps get there (checking that a validation gain carries to the leaderboard). Offset validation to public so far: +0.010.

Reading of submission 4 (full-data single model, 0.70337): a single model of the same recipe trained on validation seasons only scored 0.69473, so training on every season is worth about +0.009 on the public leaderboard. The 4-run validated blend (0.70364) still edges it. The plan is therefore: full-data versions of the best recipes, blended, with thresholds taken from the matching validation runs.

## Submissions used per day (limit 5)

| Date | Count |
| --- | --- |
| 2026-10-20 | 4 |
