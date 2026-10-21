# Research notes: what improves short-lead rain nowcasting on neighbourhood metrics

Compiled 2026-10-20 from the papers and repositories linked below. Ranked by expected gain for this
competition against implementation cost. Items marked "done" are implemented in `src/train_unet.py`
or `src/blend.py`; the journal records whether they helped on validation.

| # | Technique | Evidence | Status |
| --- | --- | --- | --- |
| 1 | Batch-pooled soft CSI (IoU) loss added to BCE | Ko et al. 2022: rain CSI 0.426 to 0.451 over 1 to 6 h, biggest gain at long leads. Weather4cast 2022 2nd place used IoU-Dice losses. | done, `--csi_w` |
| 2 | Soft FSS loss pooled over the batch, plus 5x5 and 9x9 terms weighted toward +3h | Lagerquist and Ebert-Uphoff 2022: neighbourhood losses, best window grows with lead time. NetNCC (Meteosat IR over Africa) trains on FSS. | pooled already; multi-scale done, `--ms_fss` |
| 3 | Ensemble of light heads on a frozen backbone with a learned controller | Weather4cast 2023 winner: CSI at 1 mm/h 0.142 to 0.205. | not yet |
| 4 | Temporal mixup between consecutive hourly samples | UNetTFI, Weather4cast 2023: +3.4 % mCSI, targets small-data overfitting. | done, `--mixup` |
| 5 | Importance sampling of wet samples instead of a large positive class weight | RainAI 2023: largest single gain in their ablation. | done, `--wet_sampling`, `--pos_w` |
| 6 | Over-forecast slightly at +3h: tune threshold and a 3x3 dilation per lead | Antonio 2026: FSS at fixed threshold is maximised by forecast frequency above observed when correlation is low. | dilation sweep done in `blend.py`, no gain on current blend |
| 7 | Regression pre-training on log rain, then classification fine-tuning | Ko et al. 2022: rain CSI +0.025. | tried as a multi-task auxiliary head instead (`--aux_reg 0.5`): 0.7075 against 0.7066 and 0.7081 for two plain seeds, no gain |
| 8 | Keep direct 3-channel output rather than autoregressive rollout or lead conditioning | WeatherFusionNet, SmaAt-UNet, RainAI ablations. | already the design |
| 9 | Architecture swaps (Earthformer, 3D U-Net) | Gains reported only with 13+ input frames and MSE regression; with 3 frames 2D U-Nets won Weather4cast 2023. | skipped |
| 10 | Focal loss, Fourier losses, flip test-time augmentation | Focal helps only heavy rain; flips contradict the anisotropy seen in training. | skipped |
| 11 | Upscale the input 2x before a pretrained encoder, pool the output back | Own experiment (`--upscale 2`), motivated by the 4x4 bottleneck a ResNet produces on 128-pixel maps. | done: best single recipe, validation 0.6996 single, 0.7032 last-5 average; 3x being tested |
| 12 | Translation test-time augmentation: predict on shifted copies, shift back, average | Matches the translation augmentation used in training, unlike flips (row 10). | measured with `src/eval_tta.py`: +0.0012 on a validation subset, small but consistent |
| 13 | Bayes-optimal decoding of the mask for expected FSS (calibrate, then maximise expected FSS with iterated conditional modes) | Own derivation: FSS = 2 sum(f o) / (sum f^2 + sum o^2) is linear in the mask in the numerator; neighbourhood scores are known to reward matching coverage over exact placement. | done, `src/fss_decode.py`: about +0.05 out of sample, the largest gain of the project |
| 14 | U-Net++ decoder (nested, dense skip connections) | Zhou et al. 2018, UNet++; available in segmentation-models-pytorch. | done, `--arch UnetPlusPlus`: 0.7128 last-8 average, best single model (+0.003 over the U-Net) |
| 15 | Squeeze-and-excitation attention in the decoder (scSE) | Roy et al. 2018, concurrent spatial and channel excitation. | `--attention scse`: needs `--eval_bs 16` at 3x size and runs at float32 speed; queued on Colab |
| 16 | Mixed-precision training | NVIDIA automatic mixed precision; tensor cores on the T4. | done, `--amp 1`: 0.52 s per step against 1.05 s, same early validation behaviour |

Checks on the current setup from the same review: the soft FSS loss must pool numerator and
denominator over the whole batch (it does); the BCE positive weight of 8 combined with FSS loss and
threshold tuning triple-counts the class imbalance, so it is lowered to 2 when sampling is on;
thresholds are re-tuned after every blend; validation differences under 0.01 are within noise for a
200-sample test set.

Sources opened: https://ar5iv.labs.arxiv.org/html/2202.10555 · https://github.com/bugsuse/weather4cast-2022-stage2 · https://arxiv.org/html/2311.18341 · https://arxiv.org/abs/2203.11141 · https://nora.nerc.ac.uk/id/eprint/542326/ · https://arxiv.org/html/2401.09424v1 · https://arxiv.org/html/2311.18398v1 · https://arxiv.org/html/2603.15247v1 · https://arxiv.org/html/2211.16824v1 · https://ar5iv.labs.arxiv.org/html/2112.06543 · https://arxiv.org/html/2406.04867v2 · https://ar5iv.labs.arxiv.org/html/2207.05833
