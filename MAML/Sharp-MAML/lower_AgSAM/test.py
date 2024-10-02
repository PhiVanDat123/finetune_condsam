import torch
import torch.nn.functional as F
import os
import json

from torchmeta.utils.data import BatchMetaDataLoader

from maml.datasets import get_benchmark_by_name
from maml.metalearners import ModelAgnosticMetaLearning

def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print("Train config", config)
    if args.folder is not None:
        config['folder'] = args.folder
    if args.num_steps > 0:
        config['num_steps'] = args.num_steps
    if args.num_batches > 0:
        config['num_batches'] = args.num_batches
    if args.alpha > 0:
        config['alpha'] = args.alpha
    if args.step_size > 0:
        config['step_size'] = args.step_size
    print("=============\nTest config", config)
    device = torch.device('cuda:0' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')

    benchmark = get_benchmark_by_name(config['dataset'],
                                      config['folder'],
                                      config['num_ways'],
                                      config['num_shots'],
                                      config['num_shots_test'],
                                      hidden_size=config['hidden_size'])
    
    dirname = os.path.dirname(config['model_path'])
    if args.last_model:
        last_model_path = os.path.join(dirname, "last_model.th")
        print("Test last model")
        with open(last_model_path, 'rb') as f:
            benchmark.model.load_state_dict(torch.load(f, map_location=device))
    else:
        with open(config['model_path'], 'rb') as f:
            benchmark.model.load_state_dict(torch.load(f, map_location=device))

    meta_test_dataloader = BatchMetaDataLoader(benchmark.meta_test_dataset,
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=True)

    metalearner = ModelAgnosticMetaLearning(benchmark.model,
                                            adap=config['adap'],
                                            alpha=config['alpha'],
                                            SAM_lower=config['SAM_lower'],
                                            first_order=config['first_order'],
                                            num_adaptation_steps=config['num_steps'],
                                            step_size=config['step_size'],
                                            loss_function=benchmark.loss_function,
                                            device=device)

    results = metalearner.evaluate(meta_test_dataloader,
                                   max_batches=config['num_batches'],
                                   verbose=args.verbose,
                                   desc='Test')
    print(results)
    # Save results
    with open(os.path.join(dirname, 'results.json'), 'a') as f:
        json.dump(results, f)



def main2(args):
    with open(args.config, 'r') as f:
        config = json.load(f)
    if args.folder is not None:
        config['folder'] = args.folder
    if args.num_steps > 0:
        config['num_steps'] = args.num_steps
    if args.num_batches > 0:
        config['num_batches'] = args.num_batches
    if args.num_shots_test > 0:
        config['num_shots'] = args.num_shots
    device = torch.device('cuda:0' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')

    benchmark = get_benchmark_by_name(config['dataset'],
                                      config['folder'],
                                      config['num_ways'],
                                      config['num_shots'],
                                      config['num_shots_test'],
                                      hidden_size=config['hidden_size'])

    with open(config['model_path'], 'rb') as f:
        benchmark.model.load_state_dict(torch.load(f, map_location=device))

    meta_test_dataloader = BatchMetaDataLoader(benchmark.meta_test_dataset,
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=True)
    meta_optimizer = torch.optim.Adam(benchmark.model.parameters(), lr=0.001)
    metalearner = ModelAgnosticMetaLearning(benchmark.model,
                                            meta_optimizer,
                                            adap=config['adap'],
                                            alpha=config['alpha'],
                                            SAM_lower=config['SAM_lower'],
                                            first_order=config['first_order'],
                                            num_adaptation_steps=config['num_steps'],
                                            step_size=config['step_size'],
                                            loss_function=benchmark.loss_function,
                                            device=device)

    results = metalearner.evaluate(meta_test_dataloader,
                          max_batches=config['num_batches'],
                          verbose=True,
                          desc='Testing')
    print(results)
    # Save results
    dirname = os.path.dirname(config['model_path'])
    with open(os.path.join(dirname, 'results.json'), 'a') as f:
        json.dump(results, f)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')
    parser.add_argument('config', type=str,
        help='Path to the configuration file returned by `train.py`.')
    parser.add_argument('--folder', type=int, default=None,
        help='Path to the folder the data is downloaded to. '
        '(default: path defined in configuration file).')

    # Optimization
    parser.add_argument('--num-steps', type=int, default=-1,
        help='Number of fast adaptation steps, ie. gradient descent updates '
        '(default: number of steps in configuration file).')
    parser.add_argument('--num-batches', type=int, default=-1,
        help='Number of batch of tasks per epoch '
        '(default: number of batches in configuration file).')
    parser.add_argument('--num-shots', type=int, default=-1,
        help='Number of shots '
        '(default: number of shots in configuration file).')
    parser.add_argument('--alpha', type=float, default=-1,
        help='alpha or rho in AgSAM'
        '(default: value of alpha in configuration file).')
    parser.add_argument('--step-size', type=float, default=-1,
        help='alpha or rho in AgSAM'
        '(default: value of alpha in configuration file).')
        

    # Misc
    parser.add_argument('--num-workers', type=int, default=4,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--last-model', action='store_true', help="Load the last model to test")
    

    args = parser.parse_args()
    # args.use_cuda = True
    main(args)
