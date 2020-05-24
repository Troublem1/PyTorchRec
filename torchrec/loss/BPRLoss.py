"""
BPR损失
"""
import torch
from torch.nn.modules.loss import _Loss  # noqa


class BPRLoss(_Loss):
    """BPR损失"""

    def __init__(self, reduction='mean'):
        super().__init__(None, None, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """input需要是二维的，且第二维为2"""
        assert len(input.shape) == 2 and input.shape[1] == 2, input.shape
        pos = input[:, 0]
        neg = input[:, 1]
        ret = -torch.log2(torch.sigmoid(pos - neg))
        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
        return ret
