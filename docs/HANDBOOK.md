# Handbook: machine learning, Kaggle, and this project

Written for Colombe. Read top to bottom once (about 40 minutes), then use as a reference. Section 6 is
the one to reread before talking to organisers or a jury.

## 1. Machine learning in one page

**What it is.** A program whose behaviour is fitted to data instead of written by hand. You give it
examples of inputs and the answers you want, it adjusts internal numbers (parameters, weights) until
its outputs match the answers as well as possible, then you use it on new inputs.

**The three families.**

| Family | Data you have | Example here |
| --- | --- | --- |
| Supervised | inputs + correct answers (labels, targets) | rain maps at t-2h..t and the true maps at t+1h..t+3h |
| Unsupervised | inputs only, find structure | clustering rain patterns into storm types |
| Reinforcement | an agent acts, gets rewards | not used in this competition |

Nearly all Kaggle competitions, including this one, are supervised.

**Regression vs classification.** Regression predicts a number (mm/h of rain). Classification predicts
a category (rain / no rain). This competition looks like regression but is scored like classification,
because the metric only asks whether each pixel is above 1 mm/h. That is why the model predicts a
probability of rain, not an amount.

**Train, validation, test.**
- Train: the model learns from these rows.
- Validation: rows you hold back to measure the model honestly. The model never learns from them.
- Test: rows whose answers you do not have. Kaggle keeps the answers and scores your predictions.

**Overfitting.** The model memorises the training rows instead of learning the general rule. Signature:
training error keeps going down, validation error goes up. Seen in this project at epoch 2 of the first
run. Cures: more or more varied data (augmentation), smaller model, dropout, weight decay, early
stopping, fewer epochs.

**Underfitting.** Model too weak or trained too little. Both errors high.

**Leakage.** The model gets information at training time that it will not have at prediction time,
or information that reveals the answer. Scores look wonderful, then collapse on the private test.
Classic forms: a feature computed from the target; random splits of time series (neighbouring rows are
near copies); using test rows' inputs that contain other test rows' answers. This dataset has the last
two traps and the competition rules forbid exploiting them.

## 2. The words you will hear

| Term | Meaning |
| --- | --- |
| Feature | one input variable, here one pixel value or one derived channel |
| Target, label | the value to predict |
| Model, architecture | the shape of the function being fitted (U-Net, ResNet, gradient boosting) |
| Parameters, weights | the numbers inside the model that training adjusts (ResNet34 U-Net: about 24 million) |
| Hyperparameter | a setting you choose, not learned: learning rate, batch size, epochs, dropout rate |
| Loss function | the number training minimises; must be differentiable |
| Metric | the number the competition scores; need not be differentiable (FSS on binary masks is not) |
| Gradient descent | move each weight a little in the direction that lowers the loss |
| Learning rate | size of that step; too high diverges, too low crawls |
| Adam, AdamW | gradient descent variants that adapt the step per weight; AdamW adds weight decay |
| Epoch | one full pass over the training set |
| Batch | the group of samples processed together in one step (32 here) |
| Early stopping | keep the checkpoint with the best validation score, ignore later epochs |
| Cross-validation | several train/validation splits, average the scores; more reliable than one split |
| Ensemble, blend | average predictions of several models; almost always better than the best single one |
| Seed | the random number generator start; fixing it makes a run reproducible |
| Inference | using the trained model to predict |
| Checkpoint | saved weights at some point in training (`.pt` files) |
| Pretrained | weights learned on another dataset (ImageNet photos) used as a starting point |
| Fine-tuning | continuing to train pretrained weights on your data |
| Augmentation | random transforms of training inputs (flips, rotations) to fight overfitting |
| Dropout | randomly zero some activations during training so the model cannot rely on any single path |
| Weight decay | penalty on large weights, another regulariser |
| Regularisation | any technique that reduces overfitting |
| Threshold | the probability above which you call a pixel "rain"; tuned, not assumed 0.5 |
| Baseline | the simplest sensible prediction; everything must beat it (persistence here, 0.604) |
| Out-of-fold (OOF) | predictions for validation rows made by a model that did not train on them |
| Public / private leaderboard | score on a visible part of the test set / on the hidden rest; private decides |

## 3. Deep learning for images

**Neural network.** Layers of weighted sums followed by a simple non-linear function (ReLU: keep
positives, zero the negatives). Stacking layers lets it represent complicated functions.

**Convolution.** A small filter (3x3 numbers) slides across the image and computes a weighted sum at
each position. The same filter everywhere, so it learns "an edge", "a blob", "a moving front" once and
detects it anywhere. A convolutional neural network (CNN) is a stack of these, with pooling layers that
shrink the image so later layers see bigger context.

**ResNet.** A CNN design with skip connections (the input of a block is added to its output), which lets
very deep networks train. ResNet18 and ResNet34 are the small members. "Pretrained on ImageNet" means
the weights already know generic image features from 1.2 million photos. Even on rain maps this is a
better start than random numbers.

**U-Net.** An encoder (ResNet, shrinks the image and extracts features) plus a decoder (grows it back to
full size) with skip connections between matching sizes. Output is an image the same size as the input,
one value per pixel. Standard for segmentation (which pixels are what) and for nowcasting (which pixels
will rain).

**Why not gradient boosting.** LightGBM, XGBoost and CatBoost win most tabular competitions (rows and
columns). Here the input is spatial: a pixel's future depends on its neighbours and on motion. CNNs
encode that; boosting on single pixels does not.

**GPU.** Training a CNN is millions of matrix multiplications. A GPU does them in parallel: the first
run took 35 seconds per epoch on a Kaggle T4 GPU and would take about 15 minutes per epoch on the
laptop CPU. Kaggle gives 30 free GPU hours per week; jobs queue when demand is high.

## 4. How Kaggle works

**Competition page tabs.** Overview (task, metric, timeline), Data (files, description, starter code),
Code (public notebooks, where most learning happens), Discussion (hosts post clarifications and fixes),
Leaderboard, Rules, Submissions (your history, pick the final two).

**Joining.** Click Join, accept the rules. Private competitions need an invitation first.

**Submission.** A CSV in the format of `sample_submission.csv`. Kaggle scores it against the hidden
test answers. Daily limit (5 here). Before the deadline you select up to two submissions as final; if
you do not, Kaggle picks the best public score, which may not be your best private score.

**Public vs private leaderboard.** The test set is split. Public score uses one part and is shown
during the competition. Private score uses the rest and is revealed at the end. Final rank is private
only. Chasing the public score is the classic beginner mistake: it is a small sample and partly noise.
Trust your own validation.

**Notebooks and kernels.** Kaggle's hosted Jupyter environment with free CPU or GPU. "Kernel" is the
old name, still used by the command-line tool. A script can be pushed from the terminal with
`kaggle kernels push`, runs on Kaggle's machines, and its outputs can be downloaded with
`kaggle kernels output`. Private by default.

**The command-line tool.** `kaggle` reads a token from `~/.kaggle/access_token`. Commands used here:

```
kaggle competitions files -c <slug>                 list files
kaggle competitions download -c <slug> -f <file>    download one file
kaggle competitions submit -c <slug> -f sub.csv -m "note"
kaggle competitions submissions -c <slug>           your scores
kaggle competitions leaderboard -c <slug> --show
kaggle kernels push -p <folder>                     upload and run a script on Kaggle
kaggle kernels status <user>/<slug>
kaggle kernels output <user>/<slug> -p <folder>     download results and log
```

**Progression.** Kaggle has tiers (Novice to Grandmaster) based on medals from competitions, datasets,
notebooks and discussions. Community competitions like this one do not award medals or points.

**Etiquette and rules that apply everywhere.** One account. No private code sharing between
participants. Public sharing only through the competition's own Code and Discussion tabs. No hand
labelling of test data. Document external data and pretrained models. Winners must be able to
reproduce their result from code.

## 5. This project, step by step

1. **Read the rules and metric first.** The metric (FSS at 1 mm/h, 3x3 windows) decided the whole
   design: predict binary rain masks, tune thresholds.
2. **Reimplement the metric** (`src/fss.py`) so every experiment is scored locally the same way Kaggle
   scores it. First submission: validation 0.685, leaderboard 0.695. Close enough to trust.
3. **Baselines** (`src/explore.py`): persistence 0.604, optical-flow advection 0.620. Any model must
   beat these.
4. **Validation split by season**, not random: consecutive hourly samples overlap, a random split would
   leak.
5. **Model** (`src/train_unet.py`): U-Net with a pretrained ResNet encoder, 8 input channels (3 rain
   maps, latitude, longitude, hour of day as sin and cos, day in season), optional 3 advected maps,
   3 output probability maps. Loss = weighted binary cross-entropy + a soft version of FSS computed on
   probabilities, so the network trains on something close to the metric.
6. **Regularise**: the first run overfitted after one epoch. Dropout, a lower learning rate, random
   translations of the maps and averaging the last epochs fixed it. Flips and 90 degree rotations were
   tried and hurt, because rain over Madagascar mostly moves in one direction.
7. **Upscale the input 2x** before the encoder and average the output back down: the pretrained
   encoder then sees rain cells at a size closer to the objects it learned on ImageNet. Best single
   change to the network (+0.012 on validation).
8. **Blend** (`src/blend.py`): average probability maps from several runs, seeds and epochs.
9. **Decode** (`src/fss_decode.py`): turn the probabilities into the rain mask that maximises the
   expected FSS, instead of one threshold per lead time. Biggest single gain of the project (about
   +0.05 on held-out data). See the question "How do you turn probabilities into the final map?" below.
10. **Submit** (`src/make_submission.py`), log every run in `docs/JOURNAL.md`.
11. **Final**: select two final submissions by validation score.

## 6. Explaining the work in your own words

Practise these aloud. They are what a jury or an organiser asks.

**What does your model do?** It takes three hourly rain maps and predicts, for each pixel, the
probability of rain above 1 mm/h one, two and three hours later. It is a U-Net with a ResNet34
encoder pretrained on ImageNet.

**Why a U-Net?** The output is a map the same size as the input, and rain evolution depends on
neighbouring pixels and motion. U-Nets are the standard architecture for that, used in Weather4cast
and in the Rain over Africa retrieval itself.

**Why probabilities and thresholds instead of rain amounts?** The metric is FSS at 1 mm/h: it only
checks whether a pixel is above the threshold. Predicting amounts wastes capacity. A probability plus a
tuned threshold directly targets the metric.

**How did you validate?** Held out the full December 2022 to March 2023 season. Random splits would
leak because consecutive samples overlap by two hours. The validation score (0.685) tracked the public
leaderboard (0.695).

**How do you turn probabilities into the final map?** Not with a single threshold. FSS compares the
fraction of rainy pixels in each 3x3 window of the forecast with the same fraction in the observation.
Where the model is unsure, say 40 % chance of rain over a wide area, a 0.5 threshold predicts all dry
and loses every window where rain does fall. The decoder instead picks rain pixels so that the forecast
fraction in each window matches the fraction the model expects to be observed: solid rain where the
model is confident, a speckled pattern where it is unsure. Formally it maximises the expected FSS. The
expected numerator is linear in the mask, the denominator penalises clustered masks, and the best mask
is found by iterated conditional modes (switch each pixel on or off if that raises the objective, until
nothing changes). The probabilities are first calibrated with isotonic regression on the validation
season, so that "0.4" really means rain 40 % of the time. On held-out weeks the gain is about +0.05.

**Isn't that gaming the metric?** It is decision theory: choose the output that maximises the expected
score given your probabilities, which is also why everyone tunes thresholds. It uses only that sample's
own predicted probabilities, never another test sample or any test target. It is fair to add that FSS
rewards the right amount of rain in each neighbourhood more than the exact pixel, so a speckled map
scores well but looks less natural; an operational service should publish the probability map itself.

**Why retrain on all seasons at the end, and how are those models calibrated?** A model trained
without the 2022-2023 season can be scored on it, which is how every choice was made. Once the recipe
is fixed, the same recipe trained on every season, including December 2023 just before the test
months, learns from about a third more data. Those models cannot be scored locally, so each one is
paired with its "validation twin": the same recipe and settings trained without the held-out season.
The calibration table and the decoder setting fitted on the twins are applied to the full-data models.
A paired test on the public leaderboard (same model, same calibration, only the training seasons
differ) showed +0.006 for the full-data version.

**What was the hardest problem?** Overfitting. Ten thousand hourly samples are highly correlated, so a
24 million parameter network memorised them after one epoch. Augmentation, dropout, a lower learning
rate and epoch ensembling fixed it.

**What external resources did you use?** ImageNet-pretrained encoder weights from the
segmentation-models-pytorch library (MIT licence), OpenCV for optical flow, PyTorch. No external data.
An AI coding assistant helped write and debug the scripts; every script was run and checked by me.

**How is it reproducible?** Fixed seeds, pinned requirements, one command per step in the README,
every run logged with its validation and leaderboard score.

**Ethics and limitations.** Rain over Africa is a satellite estimate, not gauge truth, so the model
inherits its errors, especially over mountains. A 13 km pixel cannot resolve a single storm cell. The
forecast is three hours ahead at most. Users such as disaster services should treat it as guidance
with uncertainty, not a warning system. Code will be open source if we win.

## 7. Reading list, in order

1. Kaggle Learn: Intro to Machine Learning, Intermediate Machine Learning (free, 4 hours each).
2. Kaggle Learn: Computer Vision (convolutions with code).
3. PyTorch 60-minute blitz (tensors, autograd, a first training loop).
4. Roberts and Lean (2008), the FSS paper, section 2 only.
5. Amell et al. (2025), Rain over Africa, abstract and figures.
6. The Weather4cast competition reports (RainAI, WeatherFusionNet) for how others nowcast from satellite data.
