import torch
from torch import optim
import math


# modified from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py

# closure copied from: https://github.com/SeanNaren/lightning-barlowtwins/blob/master/optimizer.py

class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=1e-6,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=True,
        lars_adaptation_filter=True,
    ):
        """
        Implements Layer-wise Adaptive Rate Scaling for large batch training.

        Args:
        params: iterable, parameters serving as optimization targets
        lr: float, learning rate
        weight_decay: float, weight decay (derivative of L2 penalty)
        momentum: float, momentum factor
        eta: float, LARS coefficient
        weight_decay_filter: bool, whether to apply weight decay to bias and norm parameters
        lar_adaptation_filter: bool, whether to apply LARS adaptation to bias and norm parameters
        """
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if not g["weight_decay_filter"] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if not g["lars_adaptation_filter"] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])
        return loss


# Copyright (c) SeanNaren, Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# copied from: https://github.com/SeanNaren/lightning-barlowtwins/blob/master/optimizer.py

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            max_steps: int,
            warmup_steps: int,
            lr: float,
            batch_size: int,
            last_epoch: int = -1,
            verbose: bool = False,
    ):
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.lr = lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        step = self.last_epoch
        max_steps = self.max_steps
        base_lr = self.lr * self.batch_size / 256
        if step < self.warmup_steps:
            lr = base_lr * step / self.warmup_steps
        else:
            step -= self.warmup_steps
            max_steps -= self.warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
            end_lr = base_lr * 0.001
            lr = base_lr * q + end_lr * (1 - q)
        return [lr]