import torch
from torch import nn, Tensor
from typing import Tuple


class Dual_Invariance_Paired_Loss(nn.Module):
	def __init__(self, loss_fn_inv: nn.Module, loss_fn_eq: nn.Module, lmda: float, return_both_inv_losses: bool = False, **kwargs):
		"""
		Module for computing the Paired/Equivariant Loss Function

		Args:
		loss_fn_inv: nn.Module, base SSL loss function to be used for invariant loss
		loss_fn_eq:  nn.Module, base SSL loss function to be used for equivariant loss
		lmbda:       float, weight for equivariant loss in total loss
		"""
		super(Dual_Invariance_Paired_Loss, self).__init__()
		self.loss_fn_inv = loss_fn_inv
		self.loss_fn_eq = loss_fn_eq
		self.lmda = lmda
		self.return_both_inv_losses = return_both_inv_losses

	def forward(self,
			 z11: Tuple[Tensor, Tensor, Tensor],
			 z12: Tuple[Tensor, Tensor, Tensor],
			 z21: Tuple[Tensor, Tensor, Tensor],
			 z22: Tuple[Tensor, Tensor, Tensor]
			 ) -> Tensor:
		"""
		Compute Invariant Loss for foregrounds and backgrounds on Paired Views + Equivariant Loss
		on Difference of views with same transformation.

		Args:
		z11: First View of First Samples. Tuple of Tensors from g_inv_fg, g_inv_bg, and g_equi,
			each of shape (batch_size, emb_dim)
		z12: Second View of First Samples. Tuple of Tensors from g_inv_fg, g_inv_bg, and g_equi,
			each of shape (batch_size, emb_dim)
		z21: First View of Second Samples. Tuple of Tensors from g_inv_fg, g_inv_bg, and g_equi,
			each of shape (batch_size, emb_dim)
		z22: Second View of Second Samples. Tuple of Tensors from g_inv_fg, g_inv_bg, and g_equi,
			each of shape (batch_size, emb_dim)

		Returns:
		loss: Tensor of shape (1,)
		"""
		z11_inv_fg, z12_inv_fg, z21_inv_fg, z22_inv_fg = z11[0], z12[0], z21[0], z22[0]
		z11_inv_bg, z12_inv_bg, z21_inv_bg, z22_inv_bg = z11[1], z12[1], z21[1], z22[1]
		z11_equi, z12_equi, z21_equi, z22_equi = z11[2], z12[2], z21[2], z22[2]
		# invariant loss for foregrounds
		z1_inv_fg = torch.cat([z11_inv_fg, z21_inv_fg], dim=0)
		z2_inv_fg = torch.cat([z12_inv_fg, z22_inv_fg], dim=0)
		inv_loss_fg = self.loss_fn_inv(z1_inv_fg, z2_inv_fg)

		# invariant loss for backgrounds
		z1_inv_bg = torch.cat([z11_inv_bg, z12_inv_bg], dim=0)
		z2_inv_bg = torch.cat([z21_inv_bg, z22_inv_bg], dim=0)
		inv_loss_bg = self.loss_fn_inv(z1_inv_bg, z2_inv_bg)

		# total inv loss
		inv_loss = (inv_loss_fg + inv_loss_bg) / 2.0

		# equivariant loss, treating fg and bg symmetrically
		z1_eq = torch.cat([z12_equi - z11_equi, z21_equi - z11_equi], dim=0)
		z2_eq = torch.cat([z22_equi - z21_equi, z22_equi - z12_equi], dim=0)
		eq_loss = self.loss_fn_eq(z1_eq, z2_eq)

		loss = (1 - self.lmda) * inv_loss + self.lmda * eq_loss
		if self.return_both_inv_losses:
			return loss, (inv_loss, inv_loss_fg, inv_loss_bg), eq_loss

		return loss, inv_loss, eq_loss
