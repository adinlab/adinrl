import torch
import torch.nn as nn

from nets.layers.heads import SquashedGaussianHead


class ActorNetProbabilistic(nn.Module):
    def __init__(self, n_x, n_u, n_hidden=256, upper_clamp=-2.0):
        super(ActorNetProbabilistic, self).__init__()
        self.n_u = n_u
        self.arch = nn.Sequential(
            nn.Linear(n_x[0], 256),
            nn.ReLU(),
            ##
            nn.Linear(256, 256),
            nn.ReLU(),
            ##
            nn.Linear(256, 2 * n_u[0]),
        )
        self.head = SquashedGaussianHead(self.n_u[0], upper_clamp)

    def forward(self, x, is_training=True):
        f = self.arch(x)
        return self.head(f, is_training)


##################################################################################
class ActorNet(nn.Module):
    def __init__(self, n_x, n_u, n_hidden=256):
        super(ActorNet, self).__init__()

        self.arch = nn.Sequential(
            nn.Linear(n_x[0], 256),
            nn.ReLU(),
            ##
            nn.Linear(256, 256),
            nn.ReLU(),
            ##
            nn.Linear(256, n_u[0]),
            nn.Tanh(),
        )

    def forward(self, x, is_training=None):
        return self.arch(x), None

