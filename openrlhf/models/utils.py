from typing import Optional, Tuple, Union

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs - log_probs_base
    return log_ratio * action_mask


def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if kl_coef <= 0.0:
        kl_coef = 0.0

    kl = compute_approx_kl(log_probs, log_probs_base, action_mask=action_mask)
    kl_reward = -kl_coef * kl

    r = r.clamp(min=-10, max=10)

    # The following code is equivalent to:
    #
    # last_reward = torch.zeros_like(kl)
    # for i in range(last_reward.size(0)):
    #     for t in reversed(range(last_reward.size(1))):
    #         if action_mask[i][t] > 0.5:
    #             last_reward[i][t] = r[i]
    #             break
    #
    eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
    last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

    reward = last_reward + kl_reward
    return reward, kl


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = None) -> torch.Tensor:
    if dim is not None:
        return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)
    else:
        return (tensor * mask).sum() / mask.sum()


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()

def parse_freeze_strategy(strategy):
    action, layers = strategy.split(':', maxsplit=1)
    if layers.startswith('[') and layers.endswith(']'):
        layer_list = []
        for i in layers.strip('][').split(','):
            if '-' in i:
                start, end = i.split('-')
                start, end = int(start), int(end)
                layer_list.extend(list(range(start, end+1)))
            else:
                layer_list.append(int(i))
    elif len(layers.split('-')) == 3:
        layer_list = []
        start, end, step = [int(i) for i in layers.split('-')]
        layer_list = list(range(start, end+1, step))
    else:
        raise Exception('冻结策略格式有误')
    return action, layer_list

def freeze_transformer_layers_for_qwen_new(model, layers, layer_path, action):
    if action == 'active':
        for param in model.parameters():
            param.requires_grad = False

    layer_path_lst = layer_path.split('.')
    for sub_module in layer_path_lst[1:]:
        model = getattr(model, sub_module)
    for i, layer in enumerate(model, start=1):
        if i in layers:
            for param in layer.parameters():
                if action == 'active':
                    param.requires_grad = True
                else:
                    param.requires_grad = False