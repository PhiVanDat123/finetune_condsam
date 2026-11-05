"""CondSAM two-pass training and standalone checkpoint evaluation."""
import json
from pathlib import Path
import torch
from .condsam import CondSAM
from model.smooth_cross_entropy import smooth_crossentropy


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    count = top1 = top5 = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        indices = logits.topk(min(5, logits.shape[1]), dim=1).indices
        top1 += indices[:, 0].eq(targets).sum().item()
        top5 += indices.eq(targets[:, None]).any(dim=1).sum().item()
        count += targets.numel()
    if not count:
        raise ValueError('Empty test dataset')
    return dict(top1=100 * top1 / count, top5=100 * top5 / count, samples=count)


def train_step(model, inputs, targets, optimizer, policy):
    optimizer.zero_grad(set_to_none=True)
    bn_states = [(m, m.track_running_stats) for m in model.modules()
                 if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    originals = {}
    try:
        with policy.observe(model, targets.shape[0]):
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets).mean()
            loss.backward()
        deltas = policy.perturbations()
        with torch.no_grad():
            for p, delta in deltas.items():
                originals[p] = p.detach().clone()
                p.add_(delta)
        optimizer.zero_grad(set_to_none=True)
        for module, _ in bn_states:
            module.track_running_stats = False
        smooth_crossentropy(model(inputs), targets).mean().backward()
        with torch.no_grad():
            for p, original in originals.items():
                p.copy_(original)
        originals.clear()
        optimizer.step()
        return loss.item()
    finally:
        with torch.no_grad():
            for p, original in originals.items():
                p.copy_(original)
        for module, tracking in bn_states:
            module.track_running_stats = tracking
        optimizer.zero_grad(set_to_none=True)


def run(args, model, dataset):
    device = next(model.parameters()).device
    output = Path(args.log_dir) if args.log_dir else None
    checkpoint_path = output / 'last.pt' if output else None
    if args.eval_only:
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError('CondSAM eval requires --log_dir containing last.pt')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        for key in ('model_name', 'dataset'):
            if checkpoint['args'][key] != getattr(args, key):
                raise ValueError(f'Checkpoint {key} does not match command')
        model.load_state_dict(checkpoint['state_dict'])
        metrics = evaluate(model, dataset.test, device)
        print(json.dumps(metrics))
        (output / 'eval.json').write_text(json.dumps(metrics, indent=2))
        return
    if args.fine_tune:
        raise ValueError('CondSAM: use --pretrain for ImageNet fine-tuning; resume is not implemented')
    if args.base_op != 'sgd' or args.adaptive:
        raise ValueError('CondSAM integration uses non-adaptive perturbations and SGD descent')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    policy = CondSAM(model.parameters(), rho=args.rho, beta=args.cond_beta,
                     eps=args.cond_eps, power_iters=args.cond_power_iters,
                     stats_every=args.cond_stats_every)
    policy.CUDA_GRAPHS = not args.cond_no_cuda_graphs
    # Match AlignSAM cosine schedule (epochs + 1, stepped after each epoch).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs + 1)
    if args.lr_schedule != 'cosine':
        raise ValueError('CondSAM transfer runner currently requires cosine schedule')
    if output:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
    for epoch in range(args.epochs):
        model.train()
        total_loss = count = 0
        lr = optimizer.param_groups[0]['lr']
        for inputs, targets in dataset.train:
            inputs, targets = inputs.to(device), targets.to(device)
            total_loss += train_step(model, inputs, targets, optimizer, policy) * targets.numel()
            count += targets.numel()
        scheduler.step()
        metrics = dict(epoch=epoch + 1, lr=lr, train_loss=total_loss / count,
                       **evaluate(model, dataset.test, device))
        print(json.dumps(metrics), flush=True)
        if output:
            with (output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics) + '\n')
            # Evaluation needs model weights only; avoid serializing large factor matrices.
            temporary = output / 'last.tmp'
            torch.save(dict(state_dict=model.state_dict(), args=vars(args), metrics=metrics), temporary)
            temporary.replace(checkpoint_path)
