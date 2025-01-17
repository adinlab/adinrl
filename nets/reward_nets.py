import torch
import torch.nn as nn


class RewardNet(nn.Module):
    def __init__(self, n_x, n_u, n_hidden=128):
        super().__init__()
        self.arch = nn.Sequential(
            nn.Linear(n_x[0] + n_u[0], n_hidden),
            nn.LayerNorm(n_hidden, elementwise_affine=False),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LayerNorm(n_hidden, elementwise_affine=False),
            nn.SiLU(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x, u):
        f = torch.concat([x, u], dim=-1)
        return self.arch(f)

    def cost(self, x, u):
        f = -self.forward(x, u).detach().clone()
        return f
