import torch
import torch.nn.functional as F  # noqa
from torch.nn.modules.loss import _Loss, MSELoss  # noqa


class RLPairLoss(_Loss):
    """BPR损失"""

    def __init__(self, alpha: float, reduction='mean'):
        assert reduction == "mean"
        self.alpha = alpha
        super().__init__(None, None, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """input需要是二维的，且第二维为2"""
        assert len(input.shape) == 2 and input.shape[1] == 2, input.shape
        assert len(target.shape) == 2 and target.shape[1] == 2, target.shape
        pos = input[:, 0]
        neg = input[:, 1]
        mse_loss = torch.mean((input - target) * (input - target))
        pair_loss = torch.mean((pos - neg) * (pos - neg))
        return mse_loss - pair_loss
