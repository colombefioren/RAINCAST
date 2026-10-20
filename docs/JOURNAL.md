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
| 2026-10-20 | resnet34, 3 epochs, no aug, dropout 0.2, advection inputs, laptop CPU | running | | isolates the augmentation effect |
| 2026-10-20 | resnet34, 5 epochs, lr 5e-4, no aug, dropout 0.2, advection (`indabax-unet-r34-advect`) | 0.680 single, 0.688 last-4-epoch average | | advection inputs: no measurable gain over the next row |
| 2026-10-20 | resnet34, 5 epochs, lr 5e-4, no aug, dropout 0.2 (`indabax-unet-r34-dropout`) | 0.678 single, 0.686 last-4-epoch average | | dropout alone removes the overfitting; peak now at epoch 4 |
| 2026-10-20 | blend of run 1 + r34-advect last4 + r34-dropout last4 + kernel v2 epoch 6 | 0.694 (thr 0.575/0.55/0.55) | 0.70364 | +0.009 over best single |
| 2026-10-20 | 4-run blend + 3x3 or 5x5 probability smoothing before threshold | 0.692 / 0.688 | | smoothing hurts, dropped |
| 2026-10-20 | resnet34, translation aug, dropout, advection, 6 epochs (`indabax-unet-r34-shift`) | running | | tests the motion-preserving augmentation |
| 2026-10-20 | efficientnet-b3, translation aug, dropout, advection, 6 epochs (`indabax-unet-effb3-shift`) | 0.684 single, 0.687 last-4 average | | same plateau as resnet34; useful as blend diversity |
| 2026-10-20 | 4-run blend + per-lead 3x3 dilation sweep | 0.694 | | dilation never selected, no gain |
| 2026-10-20 | resnet34, soft CSI + multi-scale FSS loss, pos_w 2, mixup 0.5, wet sampling, translation aug (`indabax-unet-r34-loss`) | running | | literature-backed loss and sampling changes, see docs/RESEARCH.md |

Kaggle kernel naming: the first three kernels are `indabax-rain-unet`, `indabax-rain-unet-b`, `indabax-rain-unet-c`. From the fourth on, the scheme is `indabax-unet-<encoder>-<variant>`.

Submission policy from 2026-10-20 evening (Colombe's decision): submit only when validation FSS predicts a public score above the current leader. Offset validation to public so far: +0.010.

## Submissions used per day (limit 5)

| Date | Count |
| --- | --- |
| 2026-10-20 | 3 |
