# Members of the final submission

The final submission blends 58 validation runs, each paired with a full-data twin that predicts the test set
(`src/twins.py`). All runs use `src/train_unet.py` from random weights on the competition data only. The columns:

- **run**: the file prefix of the run (`unetplusplus_` decoder, `clim_` climatology channels, `oof_` climatology
  counted out of fold, `scse_` attention, encoder, seed).
- **session**: the run list that trained it, `runs/molab-scratch-set<N>.txt` for `molab-set<N>` and
  `runs/gpu-clim-test.txt` for `gpu-clim-test`; the base line of each list is given at the end.
- **own settings**: the arguments on the run's own line, added to the base line.
- **averaged**: the test and validation probabilities are the mean of the run's last k epochs.
- **validation FSS**: pooled FSS on the held-out season with one threshold per lead time, for the run alone.

Blend weights: 1 for the plain and climatology members of the first sessions, 2 for the out-of-fold climatology
members, 3 for the members trained with the soft CSI loss. The weighted mean of the 58 validation probability
maps, calibrated and decoded (`final/submission.json`), scores 0.769885 on the held-out season; the same blend of
the twins' test probabilities scores 0.78384 on the public leaderboard.


## Weight 1: 27 members (mean validation FSS 0.7102)

| run | session | own settings | epochs | lr | csi_w | clim | oof | averaged | validation FSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unetplusplus_resnet34_s91 | molab-set1 | --arch UnetPlusPlus --encoder resnet34 --epochs 24 | 24 | 1e-3 | 0 | 0 | 0 | last6 | 0.6969 |
| unetplusplus_resnet34_s401 | molab-set2 | --arch UnetPlusPlus --encoder resnet34 --epochs 12 | 12 | 1e-3 | 0 | 0 | 0 | last5 | 0.7102 |
| unetplusplus_resnet34_s402 | molab-set2 | --arch UnetPlusPlus --encoder resnet34 --epochs 24 --wd 0.05 | 24 | 1e-3 | 0 | 0 | 0 | last6 | 0.6975 |
| unetplusplus_resnet34_s403 | molab-set2 | --arch UnetPlusPlus --encoder resnet34 --epochs 16 | 16 | 1e-3 | 0 | 0 | 0 | last6 | 0.7112 |
| unetplusplus_resnet34_s404 | molab-set2 | --arch UnetPlusPlus --encoder resnet34 --epochs 16 --wd 0.05 --dropout 0.4 | 16 | 1e-3 | 0 | 0 | 0 | last6 | 0.7105 |
| unetplusplus_resnet34_s407 | molab-set2 | --arch UnetPlusPlus --encoder resnet34 --epochs 12 | 12 | 1e-3 | 0 | 0 | 0 | last6 | 0.7105 |
| unetplusplus_resnet34_s408 | molab-set2 | --arch UnetPlusPlus --encoder resnet34 --epochs 12 | 12 | 1e-3 | 0 | 0 | 0 | last5 | 0.7115 |
| unetplusplus_resnet34_s409 | molab-set2 | --arch UnetPlusPlus --encoder resnet34 --epochs 16 | 16 | 1e-3 | 0 | 0 | 0 | last6 | 0.7109 |
| unetplusplus_resnet34_s410 | molab-set2 | --arch UnetPlusPlus --encoder resnet34 --epochs 12 | 12 | 1e-3 | 0 | 0 | 0 | last6 | 0.7113 |
| unetplusplus_resnet34_s411 | molab-set2 | --arch UnetPlusPlus --encoder resnet34 --epochs 12 | 12 | 1e-3 | 0 | 0 | 0 | last6 | 0.7115 |
| unetplusplus_resnet34_s412 | molab-set2 | --arch UnetPlusPlus --encoder resnet34 --epochs 16 | 16 | 1e-3 | 0 | 0 | 0 | last6 | 0.7105 |
| unetplusplus_resnet34_s602 | molab-set3 | --arch UnetPlusPlus --encoder resnet34 --epochs 12 --bs 32 --lr 2e-3 | 12 | 2e-3 | 0 | 0 | 0 | last5 | 0.7107 |
| unetplusplus_resnet34_s606 | molab-set3 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 | 14 | 1e-3 | 0 | 0 | 0 | last6 | 0.7117 |
| unetplusplus_resnet50_s601 | molab-set3 | --arch UnetPlusPlus --encoder resnet50 --epochs 12 | 12 | 1e-3 | 0 | 0 | 0 | last6 | 0.7108 |
| unetplusplus_resnet34_s608 | molab-set3b | --arch UnetPlusPlus --encoder resnet34 --epochs 12 | 12 | 1e-3 | 0 | 0 | 0 | last4 | 0.7114 |
| unetplusplus_resnet34_s610 | molab-set3b | --arch UnetPlusPlus --encoder resnet34 --epochs 16 | 16 | 1e-3 | 0 | 0 | 0 | last6 | 0.7113 |
| unetplusplus_resnet34_s612 | molab-set3b | --arch UnetPlusPlus --encoder resnet34 --epochs 12 | 12 | 1e-3 | 0 | 0 | 0 | last4 | 0.7104 |
| unetplusplus_resnet34_s614 | molab-set3b | --arch UnetPlusPlus --encoder resnet34 --epochs 14 | 14 | 1e-3 | 0 | 0 | 0 | last6 | 0.7127 |
| unetplusplus_resnet34_s616 | molab-set3b | --arch UnetPlusPlus --encoder resnet34 --epochs 12 | 12 | 1e-3 | 0 | 0 | 0 | last6 | 0.7102 |
| unetplusplus_resnet34_s618 | molab-set3b | --arch UnetPlusPlus --encoder resnet34 --epochs 16 | 16 | 1e-3 | 0 | 0 | 0 | last6 | 0.7109 |
| unetplusplus_resnet34_s620 | molab-set3b | --arch UnetPlusPlus --encoder resnet34 --epochs 12 | 12 | 1e-3 | 0 | 0 | 0 | last6 | 0.7103 |
| unetplusplus_resnet34_s1006 | molab-set4 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 | 14 | 1e-3 | 0 | 0 | 0 | last6 | 0.7116 |
| unetplusplus_resnet34_s1007 | molab-set4 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 | 14 | 1e-3 | 0 | 0 | 0 | last6 | 0.7123 |
| unetplusplus_resnet34_s1008 | molab-set4 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 | 14 | 1e-3 | 0 | 0 | 0 | last6 | 0.7111 |
| unetplusplus_resnet34_s1009 | molab-set4 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 | 14 | 1e-3 | 0 | 0 | 0 | last6 | 0.7134 |
| unetplusplus_resnet34_s1010 | molab-set4 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 | 14 | 1e-3 | 0 | 0 | 0 | last6 | 0.7116 |
| unetplusplus_scse_resnet34_s1005 | molab-set4 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --attention scse | 14 | 1e-3 | 0 | 0 | 0 | last6 | 0.7117 |


## Weight 2: 14 members (mean validation FSS 0.7121)

| run | session | own settings | epochs | lr | csi_w | clim | oof | averaged | validation FSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unetplusplus_clim_resnet34_s1201 | gpu-clim-test | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 | 14 | 1e-3 | 0 | 1 | 0 | last6 | 0.7113 |
| unetplusplus_clim_resnet34_s1202 | gpu-clim-test | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 | 14 | 1e-3 | 0 | 1 | 0 | last6 | 0.7116 |
| unetplusplus_clim_resnet34_s1203 | molab-set5 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 | 14 | 1e-3 | 0 | 1 | 0 | last6 | 0.7122 |
| unetplusplus_clim_resnet34_s1204 | molab-set5 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 | 14 | 1e-3 | 0 | 1 | 0 | last6 | 0.7118 |
| unetplusplus_clim_resnet34_s1205 | molab-set5 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 | 14 | 1e-3 | 0 | 1 | 0 | last6 | 0.7111 |
| unetplusplus_clim_oof_resnet34_s1601 | molab-set7 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1 | 14 | 1e-3 | 0 | 1 | 1 | last6 | 0.7117 |
| unetplusplus_clim_oof_resnet34_s1602 | molab-set7 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1 | 14 | 1e-3 | 0 | 1 | 1 | last6 | 0.7138 |
| unetplusplus_clim_oof_resnet34_s1604 | molab-set9 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1 | 14 | 1e-3 | 0 | 1 | 1 | last6 | 0.7132 |
| unetplusplus_clim_oof_resnet34_s1605 | molab-set9 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1 | 14 | 1e-3 | 0 | 1 | 1 | last6 | 0.7115 |
| unetplusplus_clim_oof_resnet34_s1606 | molab-set9 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1 | 14 | 1e-3 | 0 | 1 | 1 | last6 | 0.7118 |
| unetplusplus_clim_oof_resnet34_s1607 | molab-set9 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1 | 14 | 1e-3 | 0 | 1 | 1 | last6 | 0.7129 |
| unetplusplus_clim_oof_resnet34_s1608 | molab-set9 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1 | 14 | 1e-3 | 0 | 1 | 1 | last6 | 0.7127 |
| unetplusplus_clim_oof_resnet34_s1609 | molab-set9 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1 | 14 | 1e-3 | 0 | 1 | 1 | last6 | 0.7117 |
| unetplusplus_clim_oof_resnet34_s1610 | molab-set9 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1 | 14 | 1e-3 | 0 | 1 | 1 | last6 | 0.7119 |


## Weight 3: 17 members (mean validation FSS 0.7142)

| run | session | own settings | epochs | lr | csi_w | clim | oof | averaged | validation FSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unetplusplus_clim_oof_resnet34_s1637 | molab-set10 | --csi_w 0.5 | 14 | 1e-3 | 0.5 | 1 | 1 | last6 | 0.7139 |
| unetplusplus_clim_oof_resnet34_s1677 | molab-set11 | --csi_w 1.0 | 14 | 1e-3 | 1.0 | 1 | 1 | last6 | 0.7147 |
| unetplusplus_clim_oof_resnet34_s1678 | molab-set11 | --csi_w 1.0 | 14 | 1e-3 | 1.0 | 1 | 1 | last6 | 0.7141 |
| unetplusplus_clim3_oof_resnet34_s1691 | molab-set11b | --clim 3 | 14 | 1e-3 | 1.0 | 3 | 1 | last6 | 0.7139 |
| unetplusplus_clim3_oof_resnet34_s1692 | molab-set11b | --clim 3 | 14 | 1e-3 | 1.0 | 3 | 1 | last5 | 0.7137 |
| unetplusplus_clim_oof_efficientnet-b2_s1644 | molab-set11b | --encoder efficientnet-b2 | 14 | 1e-3 | 1.0 | 1 | 1 | last6 | 0.7138 |
| unetplusplus_clim_oof_resnet34_s1679 | molab-set11b | --csi_w 2.0 | 14 | 1e-3 | 2.0 | 1 | 1 | last6 | 0.7144 |
| unetplusplus_clim_oof_resnet34_s1680 | molab-set11b | --csi_w 2.0 | 14 | 1e-3 | 2.0 | 1 | 1 | last6 | 0.7135 |
| unetplusplus_clim_oof_resnet34_s1681 | molab-set11b | --csi_w 3.0 | 14 | 1e-3 | 3.0 | 1 | 1 | last6 | 0.7147 |
| unetplusplus_clim_oof_resnet34_s1682 | molab-set11b | --csi_w 3.0 | 14 | 1e-3 | 3.0 | 1 | 1 | last6 | 0.7145 |
| unetplusplus_clim_oof_resnet34_s1683 | molab-set11b | --csi_w 1.5 | 14 | 1e-3 | 1.5 | 1 | 1 | last6 | 0.7137 |
| unetplusplus_clim_oof_resnet34_s1684 | molab-set11b | --csi_w 1.5 | 14 | 1e-3 | 1.5 | 1 | 1 | last6 | 0.7143 |
| unetplusplus_clim_oof_resnet34_s1685 | molab-set11b | (base settings) | 14 | 1e-3 | 1.0 | 1 | 1 | last6 | 0.7150 |
| unetplusplus_clim_oof_resnet34_s1688 | molab-set11b | (base settings) | 14 | 1e-3 | 1.0 | 1 | 1 | last6 | 0.7136 |
| unetplusplus_clim_oof_resnet34_s1689 | molab-set11b | --csi_w 2.0 | 14 | 1e-3 | 2.0 | 1 | 1 | last6 | 0.7140 |
| unetplusplus_clim_oof_resnet34_s1690 | molab-set11b | --csi_w 2.0 | 14 | 1e-3 | 2.0 | 1 | 1 | last6 | 0.7138 |
| unetplusplus_clim_oof_resnet34_s1693 | molab-set11b | --lr 2e-3 | 14 | 2e-3 | 1.0 | 1 | 1 | last6 | 0.7152 |


## Base lines of the run lists

- `gpu-clim-test.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32`
- `molab-scratch-set1.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32`
- `molab-scratch-set10.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32 --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1`
- `molab-scratch-set11.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32 --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1`
- `molab-scratch-set11b.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32 --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1 --csi_w 1.0`
- `molab-scratch-set2.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32`
- `molab-scratch-set3.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32`
- `molab-scratch-set3b.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32`
- `molab-scratch-set4.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32`
- `molab-scratch-set5.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32`
- `molab-scratch-set7.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32`
- `molab-scratch-set9.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32`
