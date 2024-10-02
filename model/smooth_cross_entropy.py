import torch
import torch.nn as nn
import torch.nn.functional as F


# def smooth_crossentropy(pred, gold, smoothing=0.1):
#     n_class = pred.size(1)
#
#     one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
#     one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
#     log_prob = F.log_softmax(pred, dim=1)
#
#     return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)

def smooth_crossentropy_old(pred, gold, smoothing=0.1, islogits=True, tem=0.7):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)

    if islogits:
        log_prob = F.log_softmax(pred, dim=1)
    else:
        log_prob = torch.log(pred)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)


def smooth_crossentropy(pred, gold, smoothing=0.1, islogits=True, tem=1.0):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    # one_hot = F.softmax(one_hot/tem, dim=1)
    pred = pred/tem

    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)


def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""

    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)


class Unhinged(nn.Module):
    """Computes the Unhinged (linear) loss, from
    `"Learning with Symmetric Label Noise: The Importance of Being Unhinged"
    <https://arxiv.org/abs/1505.07634>`_


    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]

        loss = 1 - p

        # return torch.mean(loss)
        return loss