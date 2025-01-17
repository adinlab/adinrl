import torch
import torch.nn as nn
from torch.distributions import (
    Normal,
    Categorical,
    TransformedDistribution,
    TanhTransform,
)
from torch.distributions.transforms import TanhTransform
import math


class GaussianHead(nn.Module):
    def __init__(self, n):
        super().__init__()
        self._n = n

    def forward(self, x):
        mean = x[..., : self._n]
        logvar = x[..., self._n :].clamp(-13.0, None)
        std = logvar.exp().sqrt()
        dist = Normal(mean, std, validate_args=False)
        y = dist.rsample()
        return y, mean, dist


class SquashedGaussianHead(nn.Module):
    def __init__(self, n, upper_clamp=-2.0):
        super().__init__()
        self._n = n
        self._upper_clamp = upper_clamp

    def forward(self, x, is_training=True):
        # bt means before tanh
        mean_bt = x[..., : self._n]
        log_var_bt = x[..., self._n :]  # .clamp(-10, -self._upper_clamp)  # clamp added
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


class CategoricalHead(nn.Module):
    def __init__(self, n):
        super().__init__()
        self._n = n

    def forward(self, x):
        logit = x
        probs = nn.functional.softmax(logit)
        dist = Categorical(probs, validate_args=False)
        y = dist.rsample()
        return y, probs, dist


class DeterministicHead(nn.Module):
    def __init__(self, n):
        super().__init__()
        self._n = n

    def forward(self, x):
        mean = x
        y = mean
        dist = None
        return y, y, dist


class SquashedGaussianProcessHead(nn.Module):
    def __init__(self, n_input, n_output, kernel, n_inducing=100, scale=1.0):
        super().__init__()
        self.n_in = n_input
        self.n_out = n_output
        self.n_z = n_inducing
        self.kernel = kernel
        self.Z = nn.Parameter(torch.randn(self.n_z, self.n_in)).to(
            "cuda"
        )  # inducing input set
        self.u = nn.Parameter(torch.randn(self.n_z, self.n_out)).to(
            "cuda"
        )  # inducing output set
        self.v = nn.Parameter(torch.randn(1, self.n_out)).to(
            "cuda"
        )  # cross covariance (Kronecker)

    def forward(self, x, is_training=True):
        K = self.kernel.K(self.Z)
        Kzz_inv = torch.linalg.inv(K)
        kxz = self.kernel.k(x, self.Z)
        kxzZ = kxz @ Kzz_inv
        mu = kxzZ @ self.u
        Sig = (self.kernel.kstar(x) - torch.sum(kxzZ * kxz, axis=1)).view(-1, 1)
        Sig = Sig @ self.v.clamp(-20, 2).exp()
        dist = Normal(mu, Sig.sqrt(), validate_args=False)
        transform = TanhTransform(cache_size=1)
        dist = TransformedDistribution(dist, transform)
        if is_training:
            y = dist.rsample()
            y_logprob = dist.log_prob(y).sum(dim=-1, keepdim=True)
        else:
            y = torch.tanh(mu)
            y_logprob = None
        return y, y_logprob  # dist
