import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


def disable_running_stats(model, backup=True):
    def _disable(module):
        # if isinstance(module, _BatchNorm) and not hasattr(module, "backup_momentum"):
        #     module.backup_momentum = module.momentum
        # if isinstance(module, _BatchNorm) and module.momentum != 0:
        #     module.backup_momentum = module.momentum
        # module.momentum = 0
        if isinstance(module, _BatchNorm):
            if not hasattr(module, "backup_momentum"):
                module.backup_momentum = module.momentum
            # print("_disable", module.momentum)
            module.momentum = 0
        # elif hasattr(module, 'momentum'):
        #     print("NOTBN:", module, module.momentum)
    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            # print("_enable", module.momentum)
            module.momentum = module.backup_momentum

    model.apply(_enable)


def get_grad_norm(state, param_groups, device=None, adaptive=False, by=None):
    norm = []
    for group in param_groups:
        is_adaptive = adaptive
        if not is_adaptive:
            is_adaptive = group["adaptive"]

        for p in group["params"]:
            norm_object = None
            if by and by in state[p]:
                norm_object = state[p][by]
            elif by is None:
                norm_object = p.grad
            if norm_object is None: continue
            s = 1.0
            if "name" in state[p] and  "weight" in state[p]["name"] and is_adaptive:
                s = torch.abs(p) + 0.01
            norm.append( (s * norm_object).norm(p=2))
    norm = torch.norm(torch.stack(norm), p=2)
    if device:
        norm = norm.to(device)
    return norm


def get_model_norm(state, param_groups, device=None, by=None):
    if by:
        norm = torch.norm(
            torch.stack([
                state[p][by].norm(p=2)
                for group in param_groups for p in group["params"] if by in state[p]
            ]),
            p=2
        )
    else:
        norm = torch.norm(
                torch.stack([
                    p.data.clone().norm(p=2)
                    for group in param_groups for p in group["params"]
                    if p is not None
                ]),
                p=2
            )
    
    if device:
        norm = norm.to(device)
    return norm