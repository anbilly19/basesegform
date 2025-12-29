import torch
import torch.nn as nn

class LayerScale(nn.Module):
    """ LayerScale on tensors with channels in last-dim.
    """
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
            device=None,
            dtype=None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.empty(dim, device=device, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma