"""
Loss
"""
from torch.nn.modules.loss import _Loss  # noqa

from torchrec.metric.IMetric import IMetric


class Loss(IMetric):
    """损失函数包装"""

    def __init__(self, loss: _Loss):
        self.loss = loss
        super().__init__()
        self.name = "loss"

    def __call__(self, prediction, target, *args, **kwargs):
        return self.loss(prediction, target)
