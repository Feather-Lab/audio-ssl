import torch
from torch import nn, Tensor
import torch.nn.functional as F
import einops
import random
from typing import Tuple
import sys


class Barlow_Loss(nn.Module):
    def __init__(
        self,
        lmbda: float,
        out_dim=8192,
        scale_loss: float = 0.048,
        distributed: bool = False,
    ):
        """
        Module for computing the Barlow Twins Loss Function

        Args:
        lmbda: float, weight for the off-diagonal term
        out_dim: int, dimension of the output embeddings
        scale_loss: float, scaling factor for the loss (lets us avoid BN specific learning rates during optimization)
        distributed: bool, whether to network outputs are distributed (i.e. training using DDP)
        """
        super(Barlow_Loss, self).__init__()
        self.lmbda = lmbda
        self.bn_inv = nn.BatchNorm1d(out_dim, affine=False)
        self.scale_loss = scale_loss
        self.distributed = distributed

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        """
        Computes the Barlow Twins Loss Function

        Args:
        z1: First View. Tensor of shape (batch_size, emb_dim)
        z2: Second View. Tensor of shape (batch_size, emb_dim)

        Returns:
        loss: Tensor of shape (1,)
        """
        # sum the cross-correlation matrix between all gpus
        bs = z1.shape[0]
        if self.distributed:
            bs *= torch.distributed.get_world_size()

        c_inv = self.bn_inv(z1).T @ self.bn_inv(z2)
        c_inv.div_(bs)
        if self.distributed:
            torch.distributed.all_reduce(c_inv)

        on_diag_inv = torch.diagonal(c_inv).add_(-1).pow_(2).sum()
        off_diag_inv = off_diagonal(c_inv).pow_(2).sum()
        loss = on_diag_inv + self.lmbda * off_diag_inv
        loss = loss * self.scale_loss

        return loss


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
