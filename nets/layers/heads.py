import torch
import torch.nn as nn
from torch.distributions import (
    Normal,
    TransformedDistribution,
    TanhTransform,
)
from torch.distributions.transforms import TanhTransform

class SquashedGaussianHead(nn.Module):
    def __init__(self, n, upper_clamp=-2.0):
        super(SquashedGaussianHead, self).__init__()
        self._n = n
        self._upper_clamp = upper_clamp

    def forward(self, x, is_training=True):
        # bt means before tanh
        mean_bt = x[..., : self._n]
        log_var_bt = (x[..., self._n :]).clamp(-10, -self._upper_clamp)  # clamp added
        std_bt = log_var_bt.exp().sqrt()
        dist_bt = Normal(mean_bt, std_bt)
        transform = TanhTransform(cache_size=1)
        dist = TransformedDistribution(dist_bt, transform)
        if is_training:
            y = dist.rsample()
            y_logprob = dist.log_prob(y).sum(dim=-1, keepdim=True)
        else:
            y_samples = dist.rsample((100,))
            y = y_samples.mean(dim=0)
            y_logprob = None

        return y, y_logprob  # dist

