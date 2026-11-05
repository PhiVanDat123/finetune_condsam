# CondSAM transfer learning on AlignSAM

This repository extends Align-SAM with the supplied `condsam.py` and `gram.py`.
The commands below run CondSAM on the models and datasets from Table 4; they do
not represent the published Align-SAM results.

## Environment

Create and activate the original Anaconda environment from the repository root:

```
conda env create -f environment.yml
conda activate agnostic_sam
```

`environment.yml` is the original environment specification and contains Linux
packages. On Windows, create a compatible PyTorch/torchvision environment if the
original specification cannot be installed directly. Triton is optional; Gram
computations have a PyTorch fallback.

## Dataset

Create `../dataset` to store datasets, as in the original README:


Datasets are automatically downloaded by the loaders when training or evaluating
with `--dataset_path ../dataset`; the original README did not provide a separate
download command. Pretrained ImageNet weights are downloaded during fine-tuning
if they are not already cached.

```
python run_script/condsam_transfer.py --mode train_eval --dataset_path ../dataset
```

Stanford Cars may require manual preparation in torchvision's expected layout
if the original download source is unavailable. All commands below run from the
`AlignSAM` directory.

## Fine-tune and immediately evaluate each checkpoint

```
python run_script/condsam_transfer.py --mode train_eval --dataset_path ../dataset
```

Runs all **18 configurations**: EfficientNet-B2/B3/B4 on OxfordIIITPet, aircraft,
Flowers102, StanfordCars, Food101 and Country211.
For each configuration, the script fine-tunes pretrained ImageNet weights for
50 epochs, saves the final checkpoint, evaluates that exact checkpoint, and
writes the result before starting the next configuration. A failure stops the batch.

Run a single configuration:

```
python run_script/condsam_transfer.py --mode train_eval --dataset_path ../dataset --models EfficientNet-B3 --datasets OxfordIIITPet
```

Preview commands without training or writing files:

```
python run_script/condsam_transfer.py --mode train_eval --dry_run
```

## Outputs

Checkpoint for each configuration:

```text
runs/condsam_transfer/<model>/<dataset>/seed42/last.pt
```

All standalone evaluation results are appended to **one TXT file**:

```text
runs/condsam_transfer/eval_results.txt
```

Each entry includes model, dataset, seed, absolute checkpoint path, Top-1/Top-5
accuracy (%) and test sample count. Each experiment directory also contains
`config.json`, `metrics.jsonl` (per-epoch metrics) and `eval.json`.
Repeated runs overwrite the checkpoint and append logs/results.
Use `--output runs/my_experiment` to choose another output directory.

## Evaluate existing checkpoints

```
python run_script/condsam_transfer.py --mode eval --dataset_path ../dataset
```

Evaluate a single configuration:

```
python run_script/condsam_transfer.py --mode eval --dataset_path ../dataset --models EfficientNet-B3 --datasets OxfordIIITPet
```

Use the same `--output` and `--seed` values as training if customized.
Evaluation loads final weights, without downloading pretrained ImageNet weights.
It does not select the best test epoch. Checkpoints support evaluation, not training resume.

## Hyperparameters

Shared settings follow [AgnosticSAM_E3_transfer.out](log_files/TransferLearning/AgnosticSAM_E3_transfer.out):

| Parameter | Value |
|---|---|
| Epochs | 50 |
| Learning rate | 0.05 |
| Momentum | 0.9 |
| Rho | 0.1 |
| Weight decay | 0.0005 |
| Train / test batch size | 128 / 128 |
| Image size | 224 |
| DataLoader workers | 2 |
| Seed | 42 |
| Optimizer / scheduler | SGD / cosine (T_max=51, matching AlignSAM code) |

CondSAM defaults from the supplied source: beta=0.95, eps=1e-8, power_iters=1,
stats_every=1. Dataset splits, augmentation and fine-tuning classifier dropout
follow the existing repository implementation.
The logged bz_val=32, agsam_lambda=2.0 and label_noise_rate=0.2 are passed for
configuration consistency but are unused by CondSAM on these six clean datasets.
One seed produces accuracy only; std/standard error is not computed.

See [CONDSAM.md](run_script/CONDSAM.md) for integration details.

### Cite
```
@inproceedings{
nguyen2026alignsam,
title={Align-{SAM}: Seeking Flatter Minima for Better Cross-Subset Alignment},
author={Van-Anh Nguyen and Mehrtash Harandi and Thanh-Toan Do and Linh Ngo Van and Dinh Phung and Trung Le},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=LvllbDxKZt}
}
```

### Acknowledgement
This repository is based on [SAM](https://github.com/davda54/sam), [OT-MDR](https://github.com/anh-ntv/OT_MDR)
