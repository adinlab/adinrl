import torch
import torch.nn as nn
from nets.layers.heads import SquashedGaussianHead, SquashedGaussianProcessHead
from nets.layers.kernels import RBFKernel

from nets.utils import create_net


##################################################################################
class ActorNetProbabilistic(nn.Module):
    def __init__(
        self,
        dim_obs,
        dim_act,
        depth=3,
        width=256,
        act="crelu",
        has_norm=True,
        upper_clamp=-2.0,
    ):
        super().__init__()
        self.dim_act = dim_act

        self.arch = create_net(dim_obs[0], 2 * dim_act[0], depth, width, act, has_norm)

        self.head = SquashedGaussianHead(self.dim_act[0], upper_clamp)

    def forward(self, x, is_training=True):
        f = self.arch(x)
        return self.head(f, is_training)


# class ActorNetProbabilistic(nn.Module):
#     def __init__(self, dim_obs, dim_act, n_hidden=256, upper_clamp=-2.0):
#         super().__init__()
#         self.dim_act = dim_act
#         self.arch = nn.Sequential(
#             nn.Linear(dim_obs[0], n_hidden),
#             nn.ReLU(inplace=True),
#             ##
#             nn.Linear(n_hidden, n_hidden),
#             nn.ReLU(inplace=True),
#             ##
#             nn.Linear(n_hidden, 2 * dim_act[0]),
#         )
#         self.head = SquashedGaussianHead(self.dim_act[0], upper_clamp)
#
#     def forward(self, x, is_training=True):
#         f = self.arch(x)
#         return self.head(f, is_training)


##################################################################################
class ActorNet(nn.Module):
    def __init__(
        self,
        dim_obs,
        dim_act,
        depth=3,
        width=256,
        act="crelu",
        has_norm=True,
        upper_clamp=None,
    ):
        super().__init__()

        self.arch = create_net(
            dim_obs[0], dim_act[0], depth, width, act, has_norm
        ).append(nn.Tanh())

    def forward(self, x, is_training=None):
        out = self.arch(x).clamp(-0.9999, 0.9999)
        return out, None


##################################################################################
class ActorNetEnsemble(ActorNet):
    def __init__(
        self,
        dim_obs,
        dim_act,
        depth=3,
        width=256,
        act="crelu",
        has_norm=True,
        upper_clamp=None,
        n_elements=10,
    ):
        super().__init__(dim_obs, dim_act, depth, width, act, has_norm, upper_clamp)

        self.dim_act = dim_act

        self.arch = create_net(
            dim_obs[0], dim_act[0] * n_elements, depth, width, act, has_norm
        ).append(nn.Tanh())

        self.n_elements = n_elements

    def forward(self, x, is_training=None):
        out = self.arch(x).clamp(-0.9999, 0.9999)
        out = out.view(-1, self.n_elements, self.dim_act[0])
        return out, None


# class ActorNet(nn.Module):
#     def __init__(self, dim_obs, dim_act, n_hidden=256, upper_clamp=None):
#         super().__init__()
#
#         self.arch = nn.Sequential(
#             nn.Linear(dim_obs[0], n_hidden),
#             nn.ReLU(inplace=True),
#             ##
#             nn.Linear(n_hidden, n_hidden),
#             nn.ReLU(inplace=True),
#             ##
#             nn.Linear(n_hidden, dim_act[0]),
#             nn.Tanh(),
#         )
#
#     def forward(self, x, is_training=None):
#         out = self.arch(x).clamp(-0.9999, 0.9999)
#         return out, None


##################################################################################
class ActorNetSmall(nn.Module):
    def __init__(self, dim_obs, dim_act, n_hidden=256, upper_clamp=None):
        super().__init__()

        self.arch = nn.Sequential(
            nn.Linear(dim_obs[0], n_hidden),
            nn.ReLU(inplace=True),
            ##
            nn.Linear(n_hidden, dim_act[0]),
            nn.Tanh(),
        )

    def forward(self, x, is_training=None):
        out = self.arch(x).clamp(-0.9999, 0.9999)
        return out, None


#####################################################################
class ActorNetSmooth(ActorNet):
    def __init__(
        self, n_state, n_action, depth=3, width=256, act="crelu", has_norm=True, c=0.1
    ):
        super().__init__(n_state, n_action, depth=depth, width=width)
        self.c = c

    def forward(self, x, is_training=True):
        out = self.arch(x)
        if is_training:
            eps = torch.randn_like(out) * self.c
            out = (out + eps).clamp(-0.9999, 0.9999)
        return out, None


#####################################################################
class ActorNetSmoothSmall(ActorNetSmall):
    def __init__(self, n_state, n_action, n_hidden, c=0.1):
        super().__init__(n_state, n_action, n_hidden)
        self.c = c

    def forward(self, x, is_training=True):
        out = self.arch(x)
        if is_training:
            eps = torch.randn_like(out) * self.c
            out = (out + eps).clamp(-0.9999, 0.9999)
        return out, None


#####################################################################
class ActorNetGP(nn.Module):
    def __init__(self, dim_obs, dim_act, n_hidden=256):
        super().__init__()
        self.arch = nn.Sequential(
            nn.Linear(dim_obs[0], 64),
            nn.LayerNorm(64, elementwise_affine=False),
            nn.SiLU(),
            nn.Linear(64, 16),
            nn.LayerNorm(16, elementwise_affine=False),
        )
        self.head = SquashedGaussianProcessHead(
            16, dim_act[0], RBFKernel(), n_inducing=128, scale=3
        )

    def forward(self, x):
        x = self.arch(x)
        return self.head(x)
