# Members of the final submission

The final submission blends 58 validation runs, each paired with a full-data twin that predicts the test set
(`src/twins.py`). All runs use `src/train_unet.py` from random weights on the competition data only. The columns:

- **run**: the file prefix of the run (`unetplusplus_` decoder, `clim_` climatology channels, `oof_` climatology
  counted out of fold, `scse_` attention, encoder, seed).
- **session**: the run list that trained it, `runs/cloud-scratch-set<N>.txt` for `cloud-set<N>` and
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
| unetplusplus_resnet34_s91 | cloud-set1 | --arch UnetPlusPlus --encoder resnet34 --epochs 24 | 24 | 1e-3 | 0 | 0 | 0 | last6 | 0.6969 |
| unetplusplus_resnet34_s401 | cloud-set2 | --arch UnetPlusPlus --encoder resnet34 --epochs 12 | 12 | 1e-3 | 0 | 0 | 0 | last5 | 0.7102 |
| unetplusplus_resnet34_s402 | cloud-set2 | --arch UnetPlusPlus --encoder resnet34 --epochs 24 --wd 0.05 | 24 | 1e-3 | 0 | 0 | 0 | last6 | 0.6975 |
| unetplusplus_resnet34_s403 | cloud-set2 | --arch UnetPlusPlus --encoder resnet34 --epochs 16 | 16 | 1e-3 | 0 | 0 | 0 | last6 | 0.7112 |
| unetplusplus_resnet34_s404 | cloud-set2 | --arch UnetPlusPlus --encoder resnet34 --epochs 16 --wd 0.05 --dropout 0.4 | 16 | 1e-3 | 0 | 0 | 0 | last6 | 0.7105 |
| unetplusplus_resnet34_s407 | cloud-set2 | --arch UnetPlusPlus --encoder resnet34 --epochs 12 | 12 | 1e-3 | 0 | 0 | 0 | last6 | 0.7105 |
| unetplusplus_resnet34_s408 | cloud-set2 | --arch UnetPlusPlus --encoder resnet34 --epochs 12 | 12 | 1e-3 | 0 | 0 | 0 | last5 | 0.7115 |
| unetplusplus_resnet34_s409 | cloud-set2 | --arch UnetPlusPlus --encoder resnet34 --epochs 16 | 16 | 1e-3 | 0 | 0 | 0 | last6 | 0.7109 |
| unetplusplus_resnet34_s410 | cloud-set2 | --arch UnetPlusPlus --encoder resnet34 --epochs 12 | 12 | 1e-3 | 0 | 0 | 0 | last6 | 0.7113 |
| unetplusplus_resnet34_s411 | cloud-set2 | --arch UnetPlusPlus --encoder resnet34 --epochs 12 | 12 | 1e-3 | 0 | 0 | 0 | last6 | 0.7115 |
| unetplusplus_resnet34_s412 | cloud-set2 | --arch UnetPlusPlus --encoder resnet34 --epochs 16 | 16 | 1e-3 | 0 | 0 | 0 | last6 | 0.7105 |
| unetplusplus_resnet34_s602 | cloud-set3 | --arch UnetPlusPlus --encoder resnet34 --epochs 12 --bs 32 --lr 2e-3 | 12 | 2e-3 | 0 | 0 | 0 | last5 | 0.7107 |
| unetplusplus_resnet34_s606 | cloud-set3 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 | 14 | 1e-3 | 0 | 0 | 0 | last6 | 0.7117 |
| unetplusplus_resnet50_s601 | cloud-set3 | --arch UnetPlusPlus --encoder resnet50 --epochs 12 | 12 | 1e-3 | 0 | 0 | 0 | last6 | 0.7108 |
| unetplusplus_resnet34_s608 | cloud-set3b | --arch UnetPlusPlus --encoder resnet34 --epochs 12 | 12 | 1e-3 | 0 | 0 | 0 | last4 | 0.7114 |
| unetplusplus_resnet34_s610 | cloud-set3b | --arch UnetPlusPlus --encoder resnet34 --epochs 16 | 16 | 1e-3 | 0 | 0 | 0 | last6 | 0.7113 |
| unetplusplus_resnet34_s612 | cloud-set3b | --arch UnetPlusPlus --encoder resnet34 --epochs 12 | 12 | 1e-3 | 0 | 0 | 0 | last4 | 0.7104 |
| unetplusplus_resnet34_s614 | cloud-set3b | --arch UnetPlusPlus --encoder resnet34 --epochs 14 | 14 | 1e-3 | 0 | 0 | 0 | last6 | 0.7127 |
| unetplusplus_resnet34_s616 | cloud-set3b | --arch UnetPlusPlus --encoder resnet34 --epochs 12 | 12 | 1e-3 | 0 | 0 | 0 | last6 | 0.7102 |
| unetplusplus_resnet34_s618 | cloud-set3b | --arch UnetPlusPlus --encoder resnet34 --epochs 16 | 16 | 1e-3 | 0 | 0 | 0 | last6 | 0.7109 |
| unetplusplus_resnet34_s620 | cloud-set3b | --arch UnetPlusPlus --encoder resnet34 --epochs 12 | 12 | 1e-3 | 0 | 0 | 0 | last6 | 0.7103 |
| unetplusplus_resnet34_s1006 | cloud-set4 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 | 14 | 1e-3 | 0 | 0 | 0 | last6 | 0.7116 |
| unetplusplus_resnet34_s1007 | cloud-set4 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 | 14 | 1e-3 | 0 | 0 | 0 | last6 | 0.7123 |
| unetplusplus_resnet34_s1008 | cloud-set4 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 | 14 | 1e-3 | 0 | 0 | 0 | last6 | 0.7111 |
| unetplusplus_resnet34_s1009 | cloud-set4 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 | 14 | 1e-3 | 0 | 0 | 0 | last6 | 0.7134 |
| unetplusplus_resnet34_s1010 | cloud-set4 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 | 14 | 1e-3 | 0 | 0 | 0 | last6 | 0.7116 |
| unetplusplus_scse_resnet34_s1005 | cloud-set4 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --attention scse | 14 | 1e-3 | 0 | 0 | 0 | last6 | 0.7117 |


## Weight 2: 14 members (mean validation FSS 0.7121)

| run | session | own settings | epochs | lr | csi_w | clim | oof | averaged | validation FSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unetplusplus_clim_resnet34_s1201 | gpu-clim-test | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 | 14 | 1e-3 | 0 | 1 | 0 | last6 | 0.7113 |
| unetplusplus_clim_resnet34_s1202 | gpu-clim-test | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 | 14 | 1e-3 | 0 | 1 | 0 | last6 | 0.7116 |
| unetplusplus_clim_resnet34_s1203 | cloud-set5 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 | 14 | 1e-3 | 0 | 1 | 0 | last6 | 0.7122 |
| unetplusplus_clim_resnet34_s1204 | cloud-set5 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 | 14 | 1e-3 | 0 | 1 | 0 | last6 | 0.7118 |
| unetplusplus_clim_resnet34_s1205 | cloud-set5 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 | 14 | 1e-3 | 0 | 1 | 0 | last6 | 0.7111 |
| unetplusplus_clim_oof_resnet34_s1601 | cloud-set7 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1 | 14 | 1e-3 | 0 | 1 | 1 | last6 | 0.7117 |
| unetplusplus_clim_oof_resnet34_s1602 | cloud-set7 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1 | 14 | 1e-3 | 0 | 1 | 1 | last6 | 0.7138 |
| unetplusplus_clim_oof_resnet34_s1604 | cloud-set9 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1 | 14 | 1e-3 | 0 | 1 | 1 | last6 | 0.7132 |
| unetplusplus_clim_oof_resnet34_s1605 | cloud-set9 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1 | 14 | 1e-3 | 0 | 1 | 1 | last6 | 0.7115 |
| unetplusplus_clim_oof_resnet34_s1606 | cloud-set9 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1 | 14 | 1e-3 | 0 | 1 | 1 | last6 | 0.7118 |
| unetplusplus_clim_oof_resnet34_s1607 | cloud-set9 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1 | 14 | 1e-3 | 0 | 1 | 1 | last6 | 0.7129 |
| unetplusplus_clim_oof_resnet34_s1608 | cloud-set9 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1 | 14 | 1e-3 | 0 | 1 | 1 | last6 | 0.7127 |
| unetplusplus_clim_oof_resnet34_s1609 | cloud-set9 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1 | 14 | 1e-3 | 0 | 1 | 1 | last6 | 0.7117 |
| unetplusplus_clim_oof_resnet34_s1610 | cloud-set9 | --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1 | 14 | 1e-3 | 0 | 1 | 1 | last6 | 0.7119 |


## Weight 3: 17 members (mean validation FSS 0.7142)

| run | session | own settings | epochs | lr | csi_w | clim | oof | averaged | validation FSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unetplusplus_clim_oof_resnet34_s1637 | cloud-set10 | --csi_w 0.5 | 14 | 1e-3 | 0.5 | 1 | 1 | last6 | 0.7139 |
| unetplusplus_clim_oof_resnet34_s1677 | cloud-set11 | --csi_w 1.0 | 14 | 1e-3 | 1.0 | 1 | 1 | last6 | 0.7147 |
| unetplusplus_clim_oof_resnet34_s1678 | cloud-set11 | --csi_w 1.0 | 14 | 1e-3 | 1.0 | 1 | 1 | last6 | 0.7141 |
| unetplusplus_clim3_oof_resnet34_s1691 | cloud-set11b | --clim 3 | 14 | 1e-3 | 1.0 | 3 | 1 | last6 | 0.7139 |
| unetplusplus_clim3_oof_resnet34_s1692 | cloud-set11b | --clim 3 | 14 | 1e-3 | 1.0 | 3 | 1 | last5 | 0.7137 |
| unetplusplus_clim_oof_efficientnet-b2_s1644 | cloud-set11b | --encoder efficientnet-b2 | 14 | 1e-3 | 1.0 | 1 | 1 | last6 | 0.7138 |
| unetplusplus_clim_oof_resnet34_s1679 | cloud-set11b | --csi_w 2.0 | 14 | 1e-3 | 2.0 | 1 | 1 | last6 | 0.7144 |
| unetplusplus_clim_oof_resnet34_s1680 | cloud-set11b | --csi_w 2.0 | 14 | 1e-3 | 2.0 | 1 | 1 | last6 | 0.7135 |
| unetplusplus_clim_oof_resnet34_s1681 | cloud-set11b | --csi_w 3.0 | 14 | 1e-3 | 3.0 | 1 | 1 | last6 | 0.7147 |
| unetplusplus_clim_oof_resnet34_s1682 | cloud-set11b | --csi_w 3.0 | 14 | 1e-3 | 3.0 | 1 | 1 | last6 | 0.7145 |
| unetplusplus_clim_oof_resnet34_s1683 | cloud-set11b | --csi_w 1.5 | 14 | 1e-3 | 1.5 | 1 | 1 | last6 | 0.7137 |
| unetplusplus_clim_oof_resnet34_s1684 | cloud-set11b | --csi_w 1.5 | 14 | 1e-3 | 1.5 | 1 | 1 | last6 | 0.7143 |
| unetplusplus_clim_oof_resnet34_s1685 | cloud-set11b | (base settings) | 14 | 1e-3 | 1.0 | 1 | 1 | last6 | 0.7150 |
| unetplusplus_clim_oof_resnet34_s1688 | cloud-set11b | (base settings) | 14 | 1e-3 | 1.0 | 1 | 1 | last6 | 0.7136 |
| unetplusplus_clim_oof_resnet34_s1689 | cloud-set11b | --csi_w 2.0 | 14 | 1e-3 | 2.0 | 1 | 1 | last6 | 0.7140 |
| unetplusplus_clim_oof_resnet34_s1690 | cloud-set11b | --csi_w 2.0 | 14 | 1e-3 | 2.0 | 1 | 1 | last6 | 0.7138 |
| unetplusplus_clim_oof_resnet34_s1693 | cloud-set11b | --lr 2e-3 | 14 | 2e-3 | 1.0 | 1 | 1 | last6 | 0.7152 |


## Base lines of the run lists

- `gpu-clim-test.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32`
- `cloud-scratch-set1.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32`
- `cloud-scratch-set10.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32 --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1`
- `cloud-scratch-set11.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32 --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1`
- `cloud-scratch-set11b.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32 --arch UnetPlusPlus --encoder resnet34 --epochs 14 --clim 1 --clim_oof 1 --csi_w 1.0`
- `cloud-scratch-set2.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32`
- `cloud-scratch-set3.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32`
- `cloud-scratch-set3b.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32`
- `cloud-scratch-set4.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32`
- `cloud-scratch-set5.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32`
- `cloud-scratch-set7.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32`
- `cloud-scratch-set9.txt`: `--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --keep_epochs 6 --tta 0 --lr 1e-3 --aug 2 --dropout 0.2 --bs 16 --upscale 3 --amp 1 --eval_bs 32`
