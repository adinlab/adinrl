import torch
import torch.nn as nn

import torch.nn.functional as F


##################################################################################
class CriticNet(nn.Module):
    def __init__(self, n_x, n_u, n_hidden=256):
        super(CriticNet, self).__init__()
        self.arch = nn.Sequential(
            nn.Linear(n_x[0] + n_u[0], n_hidden),
            nn.ReLU(),
            ###
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            ###
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x, u):
        f = torch.concat([x, u], dim=-1)
        return self.arch(f)
