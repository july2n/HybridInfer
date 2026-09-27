import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep the BF16 SiLU result before multiplication, matching Qwen3.5's
        # reference MLP. Compiler fusion can remove this intermediate rounding.
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
