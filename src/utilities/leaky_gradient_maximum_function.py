from typing import Any, Tuple

import torch
from torch import Tensor
from torch.autograd import Function


class LeakyGradientMaximumFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor, other: Tensor) -> Tensor:
        return torch.maximum(input, other)

    @staticmethod
    def backward(ctx: Any, grad_outputs: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_outputs, grad_outputs
