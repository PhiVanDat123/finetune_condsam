# CondSAM transfer learning

Run from the AlignSAM directory, using the environment in environment.yml:

```powershell
python run_script/condsam_transfer.py --mode train --dataset_path ../dataset
python run_script/condsam_transfer.py --mode eval --dataset_path ../dataset
```

Both commands cover EfficientNet-B2/B3/B4 and all six Table 4 datasets.
To select a single experiment, append:
`--models EfficientNet-B3 --datasets OxfordIIITPet`.
Use `--dry_run` to print commands without downloading data or training.
Relative dataset/output paths resolve against AlignSAM. Failures stop the batch.

Each experiment writes `runs/condsam_transfer/<model>/<dataset>/seed42/`:
- `config.json`: complete configuration
- `metrics.jsonl`: test Top-1/Top-5 after every epoch
- `last.pt`: final model weights and configuration (overwritten after each epoch)
- `eval.json`: metrics from standalone evaluation of last.pt

This integrates the supplied **condsam.py**, not the earlier condsam1.py.
The supplied Gram kernels and conditioning computations are preserved. A minimal
perturbation policy replaces the unavailable SAM parent; SGD handles descent.
The first backward collects factors, then a second backward at perturbed weights
computes the update. BatchNorm running statistics update only on the first pass.
Grouped/depthwise convolutions retain the ordinary SAM direction, matching source.

Shared settings match AgnosticSAM_E3_transfer.out: 50 epochs, lr 0.05, momentum 0.9,
rho 0.1, weight decay 0.0005, batch 128, test batch 128, workers 2, size 224, seed 42.
CosineAnnealingLR uses T_max=51, matching train_AlignSAM. Dataset splits,
augmentation, normalization and classifier dropout 0.5 use existing repository code.
CondSAM defaults: beta 0.95, eps 1e-8, power_iters 1, stats_every 1.
The logged bz_val=32, agsam_lambda=2 and label_noise_rate=0.2 are passed for
configuration consistency but are unused by CondSAM on these clean datasets.
A separate validation batch is not required by this method.

Standalone eval loads final weights, not the best test epoch. It does not train,
load optimizer state, or download pretrained weights. Checkpoints are for evaluation,
not training resume. One seed gives accuracy only, without std/standard error.
Repeated runs in the same directory append metrics and replace last.pt.

Pretrained weights and datasets must be available or downloadable. In particular,
StanfordCars may require manually placing the official dataset files in the layout
expected by torchvision. Food101/Country211 now honor --dataset_path without the
original machine-specific fallback. No full benchmark or GPU timing is implied.
CUDA graphs can be disabled with --cond_no_cuda_graphs. Triton is optional;
gram.py provides a PyTorch fallback.


## Train then immediately evaluate each checkpoint

```powershell
python run_script/condsam_transfer.py --mode train_eval --dataset_path ../dataset
```

This is also the default mode. For each model/dataset pair it trains, saves last.pt,
launches standalone evaluation from that same experiment directory, and appends
one result block to `runs/condsam_transfer/eval_results.txt` before moving to the next pair.
Text blocks include model, dataset, seed, absolute checkpoint path, Top-1/Top-5 accuracy and sample count.
Each successful eval is saved immediately; failures stop the batch. Repeated
invocations append new result blocks. --dry_run performs no training or file writes.
