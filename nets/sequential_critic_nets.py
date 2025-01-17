import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.layers.bayesian_layers import CLTLayer
from nets.layers.bayesian_layers import VariationalBayesianLinear
from nets.layers.bayesian_layers import BayesianLinearEM


##################################################################################


class SequentialCriticNet(nn.Module):
    def __init__(self, dim_obs, dim_act, n_hidden=256):
        super().__init__()
        self.arch = nn.Sequential(
            nn.Linear(dim_obs[0] + dim_act[0], n_hidden),
            nn.ReLU(inplace=True),
            ##
            ###
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            ###
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x, u):
        f = torch.concat([x, u], dim=-1)
        return self.arch(f)


##################################################################################
class CriticNetTotalUncertainty(nn.Module):
    def __init__(self, dim_obs, dim_act, n_hidden=256):
        super().__init__()
        self.arch = nn.Sequential(
            nn.Linear(dim_obs[0] + dim_act[0], n_hidden),
            nn.ReLU(inplace=True),
            ###
            VariationalBayesianLinear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            ###
            VariationalBayesianLinear(n_hidden, 2),
        )

    def forward(self, x, u):
        h = torch.concat([x, u], dim=-1)
        return self.arch(h)


##################################################################################
class CriticNetBayesLinear(nn.Module):
    def __init__(self, dim_obs, dim_act, n_hidden=256):
        super().__init__()
        self.arch = nn.Sequential(
            nn.Linear(dim_obs[0] + dim_act[0], n_hidden),
            nn.ReLU(inplace=True),
            ###
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            ###
            # nn.Linear(n_hidden, n_hidden),
            # nn.ReLU(inplace=True),
        )

        self.head = VariationalBayesianLinear(n_hidden, 2)

    def forward(self, x, u):
        h = torch.concat([x, u], dim=-1)
        h = self.arch(h)
        return self.head(h)


##################################################################################
class CriticNetVB(nn.Module):
    def __init__(self, dim_obs, dim_act, n_hidden=256):
        super().__init__()
        self.arch = nn.Sequential(
            nn.Linear(dim_obs[0] + dim_act[0], n_hidden),
            nn.ReLU(inplace=True),
            ###
            VariationalBayesianLinear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            ###
            VariationalBayesianLinear(n_hidden, 1),
        )

    def forward(self, x, u):
        h = torch.concat([x, u], dim=-1)
        return self.arch(h)


##################################################################################
class SequentialCriticNetProbabilistic(nn.Module):
    def __init__(self, dim_obs, dim_act, n_hidden=256):
        super().__init__()
        self.arch = nn.Sequential(
            nn.Linear(dim_obs[0] + dim_act[0], n_hidden),
            nn.ReLU(inplace=True),
            ###
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            ###
            nn.Linear(n_hidden, 2),
        )

        self.head = VariationalBayesianLinear(n_hidden, 2)

    def forward(self, x, u):
        h = torch.concat([x, u], dim=-1)
        h = self.arch(h)
        return self.head(h)


##################################################################################
class CriticNetProbabilisticSmall(nn.Module):
    def __init__(self, dim_obs, dim_act, n_hidden=256):
        super().__init__()
        self.arch = nn.Sequential(
            nn.Linear(dim_obs[0] + dim_act[0], n_hidden),
            nn.ReLU(inplace=True),
            ###
            nn.Linear(n_hidden, 2),
        )

    def forward(self, x, u):
        h = torch.concat([x, u], dim=-1)
        return self.arch(h)


##################################################################################
class CriticNetEpistemic(nn.Module):
    def __init__(self, dim_obs, dim_act, n_hidden=256):
        super().__init__()
        self.arch = nn.Sequential(
            nn.Linear(dim_obs[0] + dim_act[0], n_hidden),
            nn.ReLU(inplace=True),
            ###
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            ###
        )
        self.head = VariationalBayesianLinear(256, 1, log_sig2_init=0.0)

    def forward(self, x, u):
        f = torch.concat([x, u], dim=-1)
        f = self.arch(f)
        mu, var = self.head.get_mean_var(f)
        return torch.concat([mu, var.clamp(1e-4, None).log()], dim=-1)


##################################################################################
class CriticNetBayesEM(nn.Module):
    def __init__(self, dim_obs, dim_act, n_hidden=256):
        super().__init__()
        self.arch = nn.Sequential(
            nn.Linear(dim_obs[0] + dim_act[0], n_hidden),
            nn.ReLU(inplace=True),
            ###
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            ###
        )
        self.head = BayesianLinearEM(256, 1)

    def e_step(self, x, u, y):
        f = torch.concat([x, u], dim=-1)
        f = self.arch(f)
        self.head.e_step(f, y)

    def forward(self, x, u):
        f = torch.concat([x, u], dim=-1)
        f = self.arch(f)
        mu, var = self.head.get_mean_var(f)
        return torch.concat([mu, var.log()], dim=-1)


##################################################################################
class CriticNetCLT(nn.Module):
    def __init__(self, dim_obs, dim_act, n_hidden=256):
        super().__init__()
        self.dense1 = nn.Linear(dim_obs[0] + dim_act[0], n_hidden)
        self.dense2 = nn.Linear(n_hidden, n_hidden)
        self.dense3 = CLTLayer(n_hidden, 1, isinput=True, isoutput=True)

    def forward(self, x, u):
        input = torch.concat([x, u], dim=-1).to(torch.float32)
        x = self.dense1(input)
        x = F.relu(x)
        x = self.dense2(x)
        x = F.relu(x)
        mu_pred, var_pred = self.dense3(x, None)
        return mu_pred, var_pred


##################################################################################
class ValueNet(nn.Module):
    def __init__(self, dim_obs, n_hidden=256):
        super().__init__()
        self.arch = nn.Sequential(
            nn.Linear(dim_obs[0], n_hidden),
            nn.ReLU(inplace=True),
            ###
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            ###
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        return self.arch(x)


##################################################################################
