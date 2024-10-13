import torch
from torch import nn, Tensor
import torch.nn.functional as F


class MMCR_Loss(nn.Module):
    def __init__(self, distributed: bool = False):
        """
        Module for computing the MMCR Loss Function

        Args:
        distributed: bool, whether to network outputs are distributed (i.e. training using DDP)
        """
        super().__init__()
        self.distributed = distributed  # currently a dummy variable for compat with original implementation

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        """
        Computes the MMCR Loss Function

        Args:
        z1: First View. Tensor of shape (batch_size, emb_dim)
        z2: Second View. Tensor of shape (batch_size, emb_dim)

        Returns:
        loss: Tensor of shape (1,)
        """
        z1 = F.normalize(z1, dim=-1, p=2)
        z2 = F.normalize(z2, dim=-1, p=2)
        # gather does nothing if not in distributed training environment
        z1, z2 = self.gather(z1), self.gather(z2)
        c = (z1 + z2) / 2.0

        return -1.0 * torch.linalg.svdvals(c).sum()

    def gather(self, tensor):
        if torch.distributed.is_initialized():
            tensor_list = [
                torch.zeros_like(tensor)
                for i in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(tensor_list, tensor, async_op=False)
            tensor_list[torch.distributed.get_rank()] = tensor
            return torch.cat(tensor_list)
        else:
            return tensor
