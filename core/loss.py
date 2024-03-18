
import torch
from torch.nn import Module

class QuantileLoss(Module):
    def __init__(self, q: float=0.25):
        super(QuantileLoss, self).__init__()
        self.q = q

    def forward(self, y_true, y_pred):
        error = y_true - y_pred
        loss = torch.max((self.q - 1) * error, self.q * error)
        return torch.mean(loss)