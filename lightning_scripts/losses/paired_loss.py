import torch
from torch import nn, Tensor


class Paired_Loss(nn.Module):
    def __init__(self, loss_fn_inv: nn.Module, loss_fn_eq: nn.Module, lmda: float, **kwargs):
        """
        Module for computing the Paired/Equivariant Loss Function

        Args:
        loss_fn_inv: nn.Module, base SSL loss function to be used for invariant loss
        loss_fn_eq: nn.Module, base SSL loss function to be used for equivariant loss
        """
        super(Paired_Loss, self).__init__()
        self.loss_fn_inv = loss_fn_inv
        self.loss_fn_eq = loss_fn_eq
        self.lmda = lmda

    def forward(self, z11: Tensor, z12: Tensor, z21: Tensor, z22: Tensor) -> Tensor:
        """
        Compute Invariant Loss on Paired Views + Equivariant Loss
        on Difference of views with same transformation.

        Args:
        z11: First View of First Samples. Tensor of shape (batch_size, emb_dim)
        z12: Second View of First Samples. Tensor of shape (batch_size, emb_dim)
        z21: First View of Second Samples. Tensor of shape (batch_size, emb_dim)
        z22: Second View of Second Samples. Tensor of shape (batch_size, emb_dim)

        Returns:
        loss: Tensor of shape (1,)
        """
        # invariant loss
        z1_inv = torch.cat([z11, z21], dim=0)
        z2_inv = torch.cat([z12, z22], dim=0)
        inv_loss = self.loss_fn_inv(z1_inv, z2_inv)

        # equivariant loss
        z1_eq = z12 - z11
        z2_eq = z22 - z21
        eq_loss = self.loss_fn_eq(z1_eq, z2_eq)

        loss = (1 - self.lmda) * inv_loss + self.lmda * eq_loss
        return loss, inv_loss, eq_loss
