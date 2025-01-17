import torch
import torch.nn as nn

from models.banditQL.bandit_heads import *


def get_bandit_criric(model_name):
    if model_name == "bandit_bayesucb":
        return BayesBanditCriticNet
    elif "bandit_bofucb" in model_name:
        if "weightless" in model_name:
            if "alpha" in model_name:
                return BOFWeightlessBanditCriticNetAlpha
            else:
                return BOFWeightlessBanditCriticNet
        else:
            return BOFBanditCriticNet
    elif model_name == "bandit_linucb":
        return LinBanditCriticNet
    elif "bandit_dlinucb" in model_name:
        return DLinBanditCriticNet
    elif model_name == "bandit_oful":
        return OFULBanditCriticNet
    else:
        raise ValueError("Unknown model: {}".format(model_name))


class BanditCriticNet(nn.Module):
    def __init__(self, n_x, n_u, n_hidden=256):
        super().__init__()
        self.arch = nn.Sequential(
            nn.Linear(n_x[0] + n_u[0], n_hidden),
            nn.LayerNorm(n_hidden, elementwise_affine=False),
            nn.SiLU(),
            ###
            nn.Linear(n_hidden, n_hidden),
            nn.LayerNorm(n_hidden, elementwise_affine=False),
            nn.SiLU(),
            ###
            # nn.Linear(n_hidden, n_hidden),
            # nn.LayerNorm(n_hidden, elementwise_affine=False),
            # nn.SiLU(),
            # ###
            nn.Linear(n_hidden, 32),
            # nn.Tanh(),
            nn.SiLU(),
            # nn.LayerNorm(n_hidden, elementwise_affine=False),
            ##
        )

    def forward(self, s, a, y=None, **kwargs):
        f = torch.concat([s, a], dim=-1)
        f = self.arch(f)
        return self.head(f, y, **kwargs)

    def update_belief(self, s, a, y, **kwargs):
        f = torch.concat([s, a], dim=-1)
        f = self.arch(f)
        self.head.update_belief(f, y)

    def ucb(self, s, a, y=None, **kwargs):
        h = torch.concat([s, a], dim=-1)
        h = self.arch(h)
        return self.head.ucb(h, y, **kwargs)

    def neg_log_prob(self, s, a, y, **kwargs):
        ypred_mu, ypred_std = self.forward(s, a, **kwargs)
        return (torch.log(ypred_std) + (ypred_mu - y) ** 2 / (ypred_std**2)).mean()

    # def neg_log_prob(self,s, a, y, **kwargs):
    #     ypred_mu, ypred_std = self.forward(s, a, y, **kwargs)
    #     return (torch.log(ypred_std) + (ypred_mu - y)**2/(ypred_std**2)).mean()


class BayesBanditCriticNet(BanditCriticNet):
    def __init__(self, n_x, n_u, n_hidden=256):
        super().__init__(n_x, n_u, n_hidden)
        self.head = BayesUCBHead(32)


class BOFBanditCriticNet(BanditCriticNet):
    def __init__(self, n_x, n_u, n_hidden=256):
        super().__init__(n_x, n_u, n_hidden)
        self.head = BOFUCBHead(32)


class BOFWeightlessBanditCriticNet(BanditCriticNet):
    def __init__(self, n_x, n_u, n_hidden=256):
        super().__init__(n_x, n_u, n_hidden)
        self.head = BOFUCBWeightlessHead(32)


class BOFWeightlessBanditCriticNetAlpha(BanditCriticNet):
    def __init__(self, n_x, n_u, n_hidden=256):
        super().__init__(n_x, n_u, n_hidden)
        self.head = BOFUCBWeightlessAlphaHead(32)


class LinBanditCriticNet(BanditCriticNet):
    def __init__(self, n_x, n_u, n_hidden=256):
        super().__init__(n_x, n_u, n_hidden)
        self.head = LinUCBHead(32)


class DLinBanditCriticNet(BanditCriticNet):
    def __init__(self, n_x, n_u, n_hidden=256):
        super().__init__(n_x, n_u, n_hidden)
        self.head = DLinUCBHead(32)


class OFULBanditCriticNet(BanditCriticNet):
    def __init__(self, n_x, n_u, n_hidden=256):
        super().__init__(n_x, n_u, n_hidden)
        self.head = OFULHead(32)
