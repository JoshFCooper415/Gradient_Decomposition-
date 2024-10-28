import torch
import torch.nn as nn
from typing import List, Tuple

def configure_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999)
) -> torch.optim.Optimizer:
    """Configure optimizer with weight decay."""
    # Separate parameters that should have weight decay from those that shouldn't
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if any(nd in name.lower() for nd in ['bias', 'layernorm', 'ln']):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    return torch.optim.AdamW(optimizer_groups, lr=lr, betas=betas)