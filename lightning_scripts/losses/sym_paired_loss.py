import torch
from torch import nn, Tensor
from typing import Tuple


class Symmetric_Paired_Loss(nn.Module):
    def __init__(self, loss_fn_inv: nn.Module, loss_fn_eq: nn.Module, lmda: float, **kwargs):
        """
        Module for computing the Paired/Equivariant Loss Function that is symmetric
        w.r.t. foregrounds and backgrounds. 

        Args:
        loss_fn_inv: nn.Module, base SSL loss function to be used for invariant loss
        loss_fn_eq:  nn.Module, base SSL loss function to be used for equivariant loss
        lmbda:       float, weight for equivariant loss in total loss 
        """
        super(Symmetric_Paired_Loss, self).__init__()
        self.loss_fn_inv = loss_fn_inv
        self.loss_fn_eq = loss_fn_eq
        self.lmda = lmda

    def forward(self,
                z11: Tuple[Tensor, Tensor],
                z12: Tuple[Tensor, Tensor],
                z21: Tuple[Tensor, Tensor],
                z22: Tuple[Tensor, Tensor]
                ) -> Tensor:
        """
        Compute Invariant Loss on Paired Views + Equivariant Loss
        on Difference of views with same transformation.

        Args:
        z11: First View of First Samples. Tuple of Tensors from g_inv and g_equi,
             each of shape (batch_size, emb_dim)
        z12: Second View of First Samples. Tuple of Tensors from g_inv and g_equi,
             each of shape (batch_size, emb_dim)
        z21: First View of Second Samples. Tuple of Tensors from g_inv and g_equi,
             each of shape (batch_size, emb_dim)
        z22: Second View of Second Samples. Tuple of Tensors from g_inv and g_equi,
             each of shape (batch_size, emb_dim)

        Returns:
        loss: Tensor of shape (1,)
        """
        z11_inv, z12_inv, z21_inv, z22_inv = z11[0], z12[0], z21[0], z22[0]
        z11_equi, z12_equi, z21_equi, z22_equi = z11[1], z12[1], z21[1], z22[1]
        # invariant loss for fg and bg 
        z1_inv = torch.cat([z11_inv, z21_inv, z11_inv, z12_inv], dim=0)
        z2_inv = torch.cat([z12_inv, z22_inv, z21_inv, z22_inv], dim=0)
        inv_loss = self.loss_fn_inv(z1_inv, z2_inv)

        # equivariant loss
        z1_eq = torch.cat([z12_equi - z11_equi, z21_equi - z11_equi], dim=0)
        z2_eq = torch.cat([z22_equi - z21_equi, z22_equi - z12_equi], dim=0)
        eq_loss = self.loss_fn_eq(z1_eq, z2_eq)

        loss = (1 - self.lmda) * inv_loss + self.lmda * eq_loss
        return loss, inv_loss, eq_loss
