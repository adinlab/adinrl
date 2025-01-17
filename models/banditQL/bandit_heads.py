import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.stats import norm
import numpy as np


class BayesianBanditHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        self.weight_mu0 = torch.zeros(in_features, 1)
        self.weight_Sig_inv0 = torch.eye(in_features)
        self.weight_Sig_inv = nn.Parameter(self.weight_Sig_inv0, requires_grad=False)
        self.weight_Sig = nn.Parameter(
            torch.linalg.inv(self.weight_Sig_inv0), requires_grad=False
        )
        self.weight_mu = nn.Parameter(self.weight_mu0, requires_grad=False)
        self.bias_mu = nn.Parameter(torch.zeros(1))
        self.noise_variance = 0

    @torch.no_grad()
    def update_belief(self, h, y):
        raise NotImplementedError

    def forward(self, h, y=None):
        raise NotImplementedError

    def ucb(self, h, y=None, **kwargs):
        raise NotImplementedError


class BayesUCBHead(BayesianBanditHead):

    def __init__(self, in_features):
        super().__init__(in_features)
        self.c = torch.tensor(0)
        self.sigma = 1

        self.lambda_ = 1

        self.weight_Sig_inv0 = torch.eye(in_features) / self.lambda_
        self.weight_Sig_inv = nn.Parameter(self.weight_Sig_inv0, requires_grad=False)
        self.weight_Sig = torch.nn.Parameter(
            torch.linalg.inv(self.weight_Sig_inv0), requires_grad=False
        )

    @torch.no_grad()
    def update_belief(self, h, y):
        weight_Sig_inv = self.weight_Sig_inv + h.T @ h / self.sigma**2
        weight_Sig = torch.linalg.inv(
            weight_Sig_inv + 1e-3 * torch.eye(self.in_features)
        )

        mu_prev = self.weight_Sig_inv @ self.weight_mu
        weight_mu = weight_Sig @ (mu_prev + h.T @ (y - self.bias_mu) / self.sigma**2)

        self.weight_Sig_inv = nn.Parameter(weight_Sig_inv, requires_grad=False)
        self.weight_Sig = nn.Parameter(weight_Sig, requires_grad=False)
        self.weight_mu = nn.Parameter(weight_mu, requires_grad=False)

    def forward(self, h, y=None, **kwargs):
        ypred_std = torch.sqrt(((h @ self.weight_Sig) * h).sum(axis=1))
        ypred_mu = F.linear(h, self.weight_mu.T, self.bias_mu)

        return ypred_mu, ypred_std.reshape(ypred_mu.shape)

    def get_beta(self, t, t_max):
        quantile = norm.ppf(1 - (1.0 / (t + 1)))
        return quantile

    def ucb(self, h, y=None, **kwargs):
        pred_mu, pred_std = self.forward(h, y, **kwargs)
        beta = self.get_beta(kwargs["t"], kwargs["t_max"])

        self.log_mu = pred_mu
        self.log_std = pred_std
        self.log_beta = beta

        # if h.shape[0] == 1:
        #     self.log_mu = pred_mu.item()
        #     self.log_std = pred_std.item()
        #     if isinstance(beta, torch.Tensor):
        #         self.log_beta = beta.item()
        #     else:
        #         self.log_beta = beta

        return pred_mu + torch.tensor(beta) * pred_std


class BOFUCBHead(BayesianBanditHead):

    def __init__(self, in_features):
        super().__init__(in_features)

        self.lambda_ = 1
        self.weight_Sig_inv0 = torch.eye(in_features) / self.lambda_
        self.weight_Sig_inv = nn.Parameter(self.weight_Sig_inv0, requires_grad=False)
        self.weight_Sig = torch.nn.Parameter(
            torch.linalg.inv(self.weight_Sig_inv0), requires_grad=False
        )

        self.weight_Sig_tilde_inv0 = torch.eye(in_features) / self.lambda_
        self.weight_Sig_tilde_inv = nn.Parameter(
            self.weight_Sig_tilde_inv0, requires_grad=False
        )
        self.weight_Sig_tilde = torch.nn.Parameter(
            torch.linalg.inv(self.weight_Sig_tilde_inv), requires_grad=False
        )
        self.prior_terms = self.weight_Sig_inv0 @ self.weight_mu0

        self.delta = 0.05
        self.d = in_features
        self.sigma = 1
        self.L = 1
        self.v = torch.trace(self.weight_Sig)
        self.gamma = 0.99999  # TODO check this
        self.S = torch.ones((in_features, 1)) * 1  # TODO check this

    @torch.no_grad()
    def update_belief(self, h, y):
        weight_Sig_inv = (
            self.gamma * self.weight_Sig_inv
            + h.T @ h / (self.sigma**2)
            + (1 - self.gamma) * self.weight_Sig_inv0
        )
        weight_Sig = torch.linalg.inv(
            weight_Sig_inv + 1e-3 * torch.eye(self.in_features)
        )

        weight_Sig_tilde_inv = (
            self.gamma**2 * self.weight_Sig_tilde_inv
            + h.T @ h / (self.sigma**2)
            + (1 - self.gamma**2) * self.weight_Sig_tilde_inv0
        )
        weight_Sig_tilde = torch.linalg.inv(
            weight_Sig_tilde_inv + 1e-3 * torch.eye(self.in_features)
        )

        weight_mu = weight_Sig @ (
            self.gamma * self.weight_Sig_inv @ self.weight_mu
            + h.T @ (y - self.bias_mu) / (self.sigma**2)
            + (1 - self.gamma) * self.weight_Sig_inv0 @ self.weight_mu0
        )

        self.weight_Sig_inv = nn.Parameter(weight_Sig_inv, requires_grad=False)
        self.weight_Sig = nn.Parameter(weight_Sig, requires_grad=False)
        self.weight_Sig_tilde_inv = nn.Parameter(
            weight_Sig_tilde_inv, requires_grad=False
        )
        self.weight_Sig_tilde = nn.Parameter(weight_Sig_tilde, requires_grad=False)
        self.weight_mu = nn.Parameter(weight_mu, requires_grad=False)

    def forward(self, h, y=None, **kwargs):
        ypred_std = torch.sqrt(
            (
                (h @ (self.weight_Sig @ self.weight_Sig_tilde_inv @ self.weight_Sig))
                * h
            ).sum(axis=1)
        )
        ypred_mu = F.linear(h, self.weight_mu.T, self.bias_mu)

        return ypred_mu, ypred_std.reshape(ypred_mu.shape)

    def get_Pi(self):

        clipped_terms = self.weight_Sig_inv0 @ self.S
        Pi = torch.sqrt(
            self.prior_terms.T @ self.weight_Sig_tilde @ self.prior_terms
        ) + torch.sqrt(clipped_terms.T @ self.weight_Sig_tilde @ clipped_terms)
        return Pi

    def get_beta(self, **kwargs):
        t = torch.tensor(kwargs["t"])
        Pi = self.get_Pi()
        beta = Pi + 1.0 / self.sigma * torch.sqrt(
            2 * math.log(1.0 / self.delta)
            + self.d
            * torch.log(
                1
                + (self.v * (self.L**2) * (1 - (self.gamma ** (2 * (t - 1)))))
                / (self.d * (self.sigma**2) * (1 - self.gamma**2))
            )
        )

        return beta

    def ucb(self, h, y=None, **kwargs):
        pred_mu, pred_std = self.forward(h, y, **kwargs)
        beta = self.get_beta(**kwargs)

        self.log_mu = pred_mu
        self.log_std = pred_std
        self.log_beta = beta

        # if h.shape[0] == 1:
        #     self.log_mu = pred_mu.item()
        #     self.log_std = pred_std.item()
        #     if isinstance(beta, torch.Tensor):
        #         self.log_beta = beta.item()
        #     else:
        #         self.log_beta = beta

        return pred_mu + beta * pred_std


class BOFUCBWeightlessHead(BayesianBanditHead):
    def __init__(self, in_features):
        super().__init__(in_features)

        self.lambda_ = 1
        self.weight_Sig_inv0 = torch.eye(in_features) / self.lambda_
        self.weight_Sig_inv = nn.Parameter(self.weight_Sig_inv0, requires_grad=False)
        self.weight_Sig = torch.nn.Parameter(
            torch.linalg.inv(self.weight_Sig_inv0), requires_grad=False
        )

        self.prior_terms = self.weight_Sig_inv0 @ self.weight_mu0

        self.delta = 0.05
        self.d = in_features
        self.sigma = 1
        self.L = 1
        self.v = torch.trace(self.weight_Sig)
        self.S = torch.ones((in_features, 1)) * 1  # TODO check this

    @torch.no_grad()
    def update_belief(self, h, y):
        weight_Sig_inv = self.weight_Sig_inv + h.T @ h / (self.sigma**2)
        weight_Sig = torch.linalg.inv(weight_Sig_inv)

        weight_mu = weight_Sig @ (
            self.weight_Sig_inv @ self.weight_mu
            + h.T @ (y - self.bias_mu) / (self.sigma**2)
        )

        self.weight_Sig_inv = nn.Parameter(weight_Sig_inv, requires_grad=False)
        self.weight_Sig = nn.Parameter(weight_Sig, requires_grad=False)
        self.weight_mu = nn.Parameter(weight_mu, requires_grad=False)

    def forward(self, h, y=None, **kwargs):

        ypred_std = torch.sqrt(((h @ self.weight_Sig) * h).sum(axis=1))
        ypred_mu = F.linear(h, self.weight_mu.T, self.bias_mu)

        return ypred_mu, ypred_std.reshape(ypred_mu.shape)

    def get_Pi(self):

        clipped_terms = self.weight_Sig_inv0 @ self.S
        Pi = torch.sqrt(self.prior_terms.T @ self.prior_terms) + torch.sqrt(
            clipped_terms.T @ clipped_terms
        )
        return Pi

    def get_beta(self, **kwargs):
        t = torch.tensor(kwargs["t"])
        Pi = self.get_Pi()
        beta = Pi + 1.0 / self.sigma * torch.sqrt(
            2 * math.log(1.0 / self.delta)
            + self.d
            * torch.log(
                1 + (self.v * (self.L**2) * (t - 1)) / (self.d * (self.sigma**2))
            )
        )

        return beta

    def ucb(self, h, y=None, **kwargs):
        pred_mu, pred_std = self.forward(h, y, **kwargs)
        beta = self.get_beta(**kwargs)

        self.log_mu = pred_mu
        self.log_std = pred_std
        self.log_beta = beta

        # if h.shape[0] == 1:
        #     self.log_mu = pred_mu.item()
        #     self.log_std = pred_std.item()
        #     if isinstance(beta, torch.Tensor):
        #         self.log_beta = beta.item()
        #     else:
        #         self.log_beta = beta
        return pred_mu + beta * pred_std


class BOFUCBWeightlessAlphaHead(BOFUCBWeightlessHead):
    def __init__(self, in_features):
        super().__init__(in_features)
        self.alpha = 1 + np.sqrt(np.log(2 / self.delta) / 2)

    def get_beta(self, **kwargs):
        return self.alpha


class LinUCBHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.in_features = in_features
        self.delta = 0.05
        self.alpha = 1 + np.sqrt(np.log(2 / self.delta) / 2)
        self.lambda_ = 1

        self.A = nn.Parameter(
            torch.eye(in_features) / self.lambda_, requires_grad=False
        )
        self.A_inv = nn.Parameter(torch.linalg.inv(self.A), requires_grad=False)
        self.b = nn.Parameter(torch.zeros(in_features, 1), requires_grad=False)
        self.theta_T = nn.Parameter((self.A_inv @ self.b).T, requires_grad=False)

        self.bias_mu = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def update_belief(self, h, y):
        A = self.A + h.T @ h
        b = self.b + h.T @ y

        self.A = nn.Parameter(A, requires_grad=False)
        self.A_inv = nn.Parameter(
            torch.linalg.inv(A + 1e-3 * torch.eye(self.in_features)),
            requires_grad=False,
        )
        self.b = nn.Parameter(b, requires_grad=False)
        self.theta_T = nn.Parameter((self.A_inv @ self.b).T, requires_grad=False)

    def forward(self, h, y=None, **kwargs):
        ypred_mu = F.linear(h, self.theta_T, bias=self.bias_mu)
        ypred_std = torch.sqrt(((h @ self.A_inv) * h).sum(axis=1))

        return ypred_mu, ypred_std.reshape(ypred_mu.shape)

    def get_beta(self, **kwargs):
        return self.alpha

    def ucb(self, h, y=None, **kwargs):
        pred_mu, pred_std = self.forward(h, y, **kwargs)
        beta = self.get_beta(**kwargs)

        self.log_mu = pred_mu
        self.log_std = pred_std
        self.log_beta = beta

        # if h.shape[0] == 1:
        #     self.log_mu = pred_mu.item()
        #     self.log_std = pred_std.item()
        #     if isinstance(beta, torch.Tensor):
        #         self.log_beta = beta.item()
        #     else:
        #         self.log_beta = beta
        return pred_mu + beta * pred_std


class OFULHead(LinUCBHead):
    def __init__(self, in_features):
        super().__init__(in_features)
        self.delta = 0.05
        self.d = in_features
        self.sigma = 1

    def get_beta(self, **kwargs):
        t = torch.tensor(kwargs["t"])
        beta = self.sigma * torch.sqrt(
            self.d * torch.log((1 + t / self.lambda_) / (self.delta))
        ) + np.sqrt(self.lambda_)
        return beta


class DLinUCBHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.in_features = in_features
        self.delta = 0.05
        self.alpha = 1 + np.sqrt(np.log(2 / self.delta) / 2)
        self.lambda_ = 1

        self.A = nn.Parameter(
            torch.eye(in_features) / self.lambda_, requires_grad=False
        )
        self.A_inv = nn.Parameter(torch.linalg.inv(self.A), requires_grad=False)
        self.A_tilde = nn.Parameter(
            torch.eye(in_features) / self.lambda_, requires_grad=False
        )
        self.A_tilde_inv = nn.Parameter(torch.linalg.inv(self.A), requires_grad=False)
        self.b = nn.Parameter(torch.zeros(in_features, 1), requires_grad=False)
        self.theta_T = nn.Parameter((self.A_inv @ self.b).T, requires_grad=False)

        self.bias_mu = nn.Parameter(torch.zeros(1))
        self.S = 1
        self.gamma = 0.99999
        self.sigma = 1
        self.L = 1
        self.d = in_features

    @torch.no_grad()
    def update_belief(self, h, y):
        A = (
            self.gamma * self.A
            + h.T @ h
            + (1 - self.gamma) * self.lambda_ * torch.eye(self.in_features)
        )
        A_tilde = (
            (self.gamma**2) * self.A_tilde
            + h.T @ h
            + (1 - (self.gamma**2)) * self.lambda_ * torch.eye(self.in_features)
        )
        b = self.gamma * self.b + h.T @ y

        self.A = nn.Parameter(A, requires_grad=False)
        self.A_inv = nn.Parameter(
            torch.linalg.inv(A + 1e-3 * torch.eye(self.in_features)),
            requires_grad=False,
        )
        self.A_tilde = nn.Parameter(A_tilde, requires_grad=False)
        self.A_tilde_inv = nn.Parameter(
            torch.linalg.inv(A_tilde + 1e-3 * torch.eye(self.in_features)),
            requires_grad=False,
        )
        self.b = nn.Parameter(b, requires_grad=False)
        self.theta_T = nn.Parameter((self.A_inv @ self.b).T, requires_grad=False)

    def forward(self, h, y=None, **kwargs):
        ypred_mu = F.linear(h, self.theta_T, bias=self.bias_mu)
        ypred_std = torch.sqrt(
            ((h @ (self.A_inv @ self.A_tilde @ self.A_inv)) * h).sum(axis=1)
        )

        return ypred_mu, ypred_std.reshape(ypred_mu.shape)

    def get_beta(self, **kwargs):
        t = torch.tensor(kwargs["t"])
        beta = np.sqrt(self.lambda_) * self.S + self.sigma * torch.sqrt(
            2 * np.log(1.0 / self.delta)
            + self.d
            * torch.log(
                1
                + ((self.L**2) * (1 - (self.gamma ** (2 * (t - 1)))))
                / (self.lambda_ * self.d * (1 - (self.gamma**2)))
            )
        )
        return beta

    def ucb(self, h, y=None, **kwargs):
        pred_mu, pred_std = self.forward(h, y, **kwargs)
        beta = self.get_beta(**kwargs)

        self.log_mu = pred_mu
        self.log_std = pred_std
        self.log_beta = beta

        # if h.shape[0] == 1:
        #     self.log_mu = pred_mu.item()
        #     self.log_std = pred_std.item()
        #     if isinstance(beta, torch.Tensor):
        #         self.log_beta = beta.item()
        #     else:
        #         self.log_beta = beta
        return pred_mu + beta * pred_std
