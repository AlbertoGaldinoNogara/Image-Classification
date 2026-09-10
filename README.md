# Butterfly image classification

A convolutional network built and trained from scratch in PyTorch — no pretrained
weights, no transfer learning — on 11 species of butterflies and moths.

Coursework project — *Advanced Programming & Deep Learning for AI*, MSc Data
Analytics for Business, Università Cattolica.

## Data

Derived from the [butterfly & moth species dataset](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species)
on Kaggle, reduced to the 11 largest classes: **1,816 RGB images at 224×224**,
split 60 / 20 / 20 into train / validation / test.

## Model

Inception-style modules — parallel branches at several kernel sizes, concatenated
— in the spirit of GoogLeNet, chosen over an MLP or LeNet so the network can pick
up wing patterning at more than one scale. Trained with Adam (lr 1e-3), StepLR
(step 10, γ 0.1), cross-entropy, batch size 32, 20 epochs.

## Results

| Run | Training data | Test accuracy | Test loss |
|---|---|---|---|
| 1 | augmented | **0.838** | 0.612 |
| 2 | original, no augmentation | 0.918 | 0.297 |

Augmentation pipeline in run 1: `RandomHorizontalFlip`, `RandomRotation(15)`,
`RandomResizedCrop(224, scale=(0.8, 1.0))`, `ColorJitter(0.2, 0.2, 0.2, 0.1)`.

### Read the second row with care

The two runs are **not a clean ablation**, and the 8-point gap should not be
attributed to augmentation. Two problems:

1. **The model was not re-initialised between runs.** `CustomCNN` is instantiated
   once. Run 2 continues training the weights that run 1 had already trained for
   20 epochs, so it is a 40-epoch model compared against a 20-epoch model.
2. **The split was redrawn.** Run 2 calls `random_split` again with no fixed seed,
   so its test set is a different sample — and overlaps run 1's training set,
   which the shared weights have already seen.

Run 1 (0.838) is the defensible number: one model, one split, held-out test set.
Run 2 tells us little as it stands.

A correct version of this experiment would fix a seed, instantiate a fresh model
per condition, hold the split constant across conditions, and repeat over several
seeds — the gap between augmented and non-augmented training is well within the
range that seed variance alone can produce on 1,816 images.

The underlying question is still worth asking. These classes are separated by fine
wing patterning, and `ColorJitter` plus an aggressive `RandomResizedCrop` may well
destroy exactly the signal the task depends on. Augmentation encodes a claim about
which transformations leave the label unchanged; on fine-grained classification
that claim is not free.

## Repository

```
└── Main.ipynb    data pipeline, model definition, training loop, evaluation, both runs
```

Written for Google Colab — `data_dir` points at `/content/IMAGES`, an `ImageFolder`
tree with one subdirectory per class. Point it at a local copy to run elsewhere.

## Reproducing

```
torch, torchvision, scikit-learn, matplotlib, numpy
```

Open `Main.ipynb`, set `data_dir`, run top to bottom. GPU strongly recommended.
