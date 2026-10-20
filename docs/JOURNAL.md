# Experiment journal

One line per run. Validation = pooled FSS on the December 2022 to March 2023 season.
Public leaderboard = score shown on Kaggle. Private leaderboard decides the final rank.

| Date | Run | Validation FSS | Public LB | Notes |
| --- | --- | --- | --- | --- |
| 2026-10-20 | persistence baseline | 0.604 | not submitted | +1h 0.76, +2h 0.59, +3h 0.46 |
| 2026-10-20 | persistence, 3x3 smoothing, threshold 1.0 | 0.602 | not submitted | smoothing does not help |
| 2026-10-20 | resnet34 U-Net, 12 epochs, lr 1e-3, no aug (kernel v1) | 0.685 (best epoch 1, thr 0.525 all leads) | 0.69473 | overfits: val 0.685 at epoch 1, falls to 0.640 by epoch 11 while train loss keeps dropping. +1h 0.79, +2h 0.67, +3h 0.59 |
| 2026-10-20 | resnet34, 6 epochs, lr 5e-4, rot90+flip aug, dropout 0.2 (kernel v2) | running | | per-epoch probs saved for epoch ensembling |
| 2026-10-20 | resnet18, same + stride 2 (kernel -b v1) | running | | |

## Submissions used per day (limit 5)

| Date | Count |
| --- | --- |
| 2026-10-20 | 1 |
