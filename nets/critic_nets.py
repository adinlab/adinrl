from torch import nn as nn

from nets.utils import create_net


class CriticNet(nn.Module):
    def __init__(
        self, dim_obs, dim_act, depth=3, width=256, act="crelu", has_norm=True
    ):
        super().__init__()

        self.arch = create_net(
            dim_obs[0] + dim_act[0], 1, depth, width, act=act, has_norm=has_norm
        )

    def forward(self, xu):
        return self.arch(xu)


class CriticNetProbabilistic(nn.Module):
    def __init__(
        self, dim_obs, dim_act, depth=3, width=256, act="crelu", has_norm=True
    ):
        super().__init__()

        self.arch = create_net(
            dim_obs[0] + dim_act[0], 2, depth, width, act=act, has_norm=has_norm
        )

    def forward(self, xu):
        return self.arch(xu)


class EMstyle(nn.Module):

    def __init__(
        self, dim_obs, dim_act, depth=3, width=256, act="crelu", has_norm=True
    ):
        super().__init__()
        self.arch = create_net(
            dim_obs[0] + dim_act[0], width, depth, width, act=act, has_norm=has_norm
        )

        # self.head = nn.Linear(width,2)

    def forward(self, x):
        x = self.arch(x)
        return x
