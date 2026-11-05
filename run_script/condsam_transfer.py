"""Run all 18 CondSAM transfer benchmarks, or evaluate their final checkpoints."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

MODELS = ['EfficientNet-B2', 'EfficientNet-B3', 'EfficientNet-B4']
DATASETS = ['OxfordIIITPet', 'aircraft', 'Flowers102', 'StanfordCars', 'Food101', 'Country211']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'eval', 'train_eval'], default='train_eval')
    parser.add_argument('--dataset_path', default='../dataset')
    parser.add_argument('--output', default='runs/condsam_transfer')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--cond_no_cuda_graphs', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for model in args.models:
        for dataset in args.datasets:
            output = Path(args.output) / model / dataset / f'seed{args.seed}'
            command = [sys.executable, '-u', 'main.py', '--dataset_path', args.dataset_path,
                       '--model_name', model, '--dataset', dataset, '--training_mode', 'condsam',
                       '--epochs', '50', '--batch_size', '128', '--bz_val', '32', '--bz_test', '128',
                       '--learning_rate', '0.05', '--momentum', '0.9', '--rho', '0.1',
                       '--weight_decay', '0.0005', '--threads', '2', '--img_size', '224',
                       '--lr_schedule', 'cosine', '--base_op', 'sgd', '--random_seed', str(args.seed),
                       '--label_noise_rate', '0.2', '--agsam_lambda', '2.0',
                       '--cond_beta', '0.95', '--cond_eps', '1e-8', '--cond_power_iters', '1',
                       '--cond_stats_every', '1', '--log_dir', str(output)]
            if args.cond_no_cuda_graphs:
                command.append('--cond_no_cuda_graphs')
            phases = ['train', 'eval'] if args.mode == 'train_eval' else [args.mode]
            for phase in phases:
                phase_command = command + (['--pretrain'] if phase == 'train' else ['--eval_only'])
                print(subprocess.list2cmdline(phase_command), flush=True)
                if args.dry_run:
                    continue
                subprocess.run(phase_command, cwd=root, check=True)
                if phase == 'eval':
                    # Read only after successful evaluation of this exact checkpoint.
                    metrics = json.loads((root / output / 'eval.json').read_text())
                    results_path = root / args.output / 'eval_results.txt'
                    results_path.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint = (root / output / 'last.pt').resolve()
                    with results_path.open('a', encoding='utf-8') as stream:
                        stream.write(
                            f'Model: {model} | Dataset: {dataset} | Seed: {args.seed}\n'
                            f'Checkpoint: {checkpoint}\n'
                            f'Top-1 accuracy: {metrics["top1"]:.4f}%\n'
                            f'Top-5 accuracy: {metrics["top5"]:.4f}%\n'
                            f'Test samples: {metrics["samples"]}\n'
                            + '-' * 80 + '\n'
                        )
                    print(f'Eval result saved to {results_path}', flush=True)


if __name__ == '__main__':
    main()
