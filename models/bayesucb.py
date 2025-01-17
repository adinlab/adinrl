import torch
import math
from torch import nn
import numpy as np
import torch.nn.init as init
from models.cql import ContinuousQLearning
import torch.nn.functional as F
from models.ddpg import DeepDeterministicPolicyGradient
from utils.implicit import AbstractProblem
from torch.distributions import Normal
from scipy.stats import norm


#####################################################
class BayesianLinear(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        self.weight_mu0 = torch.zeros(in_features, 1)
        self.weight_Sig_inv0 = torch.eye(in_features)
        self.weight_Sig_inv = self.weight_Sig_inv0
        self.weight_Sig = torch.eye(in_features)
        self.weight_mu = nn.Parameter(self.weight_mu0, requires_grad=False)
        self.bias_mu = nn.Parameter(torch.zeros(1))
        self.noise_variance = 1
        # init.kaiming_uniform_(self.weight_mu, a=math.sqrt(self.weight_mu.shape[1]))

    def forward(self, h, y=None):
        if y is not None:  # then update belief
            with torch.no_grad():
                weight_Sig_inv = self.weight_Sig_inv + h.T @ h / self.noise_variance
                weight_Sig = torch.linalg.inv(weight_Sig_inv)

                mu_prev = self.weight_Sig_inv @ self.weight_mu
                weight_mu = weight_Sig @ (
                    mu_prev + h.T @ (y - self.bias_mu) / self.noise_variance
                )

                self.weight_Sig_inv = weight_Sig_inv
                self.weight_Sig = weight_Sig
                self.weight_mu = nn.Parameter(weight_mu)

        # ypred_std = torch.clamp_min(self.noise_variance + ((h @ self.weight_Sig) * h).sum(axis=1),1e-3).sqrt()
        # ypred_mu = F.linear(h, self.weight_mu.T, self.bias_mu)

        ypred_std = torch.clamp_min(
            self.noise_variance + ((h @ self.weight_Sig) * h).sum(axis=1), 1e-3
        ).sqrt()
        ypred_mu = F.linear(h, self.weight_mu.T, self.bias_mu)

        return ypred_mu, ypred_std

    def ucb(self, h, y=None):
        pred_mu, pred_std = self.forward(h, y)
        return pred_mu, pred_std


##########################################################################
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
            nn.Linear(n_hidden, 32),
            nn.LayerNorm(32, elementwise_affine=False),
            nn.Tanh(),
            ##
        )
        self.head = BayesianLinear(32)
        # self.head = nn.Linear(n_hidden,1)

    def forward(self, s, a, y=None):
        f = torch.concat([s, a], dim=-1)
        f = self.arch(f)
        return self.head(f, y)

    def ucb(self, s, a, y=None):
        h = torch.concat([s, a], dim=-1)
        h = self.arch(h)
        return self.head.ucb(h, y)

    def neg_log_prob(self, s, a, y):
        ypred_mu, ypred_std = self.forward(s, a, y)
        return (torch.log(ypred_std) + (ypred_mu - y) ** 2 / (ypred_std**2)).mean()


##########################################################################
class Qminimizer(AbstractProblem):
    def __init__(self, env, model, solvertype="gd", ucbtype="bayesucb"):
        super().__init__()
        self.model = model
        self.env = env
        self.ucbtype = ucbtype
        self.a_min = self.env.action_space.low[0]
        self.a_max = self.env.action_space.high[0]
        self.solvertype = solvertype
        self.c = 0

        self.ucb = self.get_ucb()

    def get_ucb(self):
        if self.ucbtype == "bayesucb":
            return self.bayes_ucb
        elif self.ucbtype == "bofucb":
            return self.bofucb
        else:
            raise NotImplementedError(f"Unknown UCB type, {self.ucbtype}")

    @property
    def _t(self):
        return min(self._get_t() + 1, self._t_max)

    def bayes_ucb(self, s, astar):
        mu, std = self.model.ucb(s, astar.clamp(min=self.a_min, max=self.a_max))
        quantile = norm.ppf(
            1.0 - (1.0 / ((self._t + 1) * (np.log(self._t_max) ** self.c)))
        )
        loss = -(mu + quantile * std).mean()
        return loss

    def bofucb(self, s, astar):
        h = torch.concat([s, astar.clamp(min=self.a_min, max=self.a_max)], dim=-1)
        x = self.model.arch(h)

        Pi_t_minus_1 = 0.0
        weight_sig_inv0 = self.model.head.weight_Sig_inv0.data.clone()
        wight_sig = self.model.head.weight_Sig.data.clone()
        weight_mu0 = self.model.head.weight_mu0.data.clone()
        theta = nn.Parameter(self.model.head.weight_mu.data.clone(), requires_grad=True)
        optim_theta = torch.optim.SGD([theta], lr=1e-1)
        for i in range(10):
            optim_theta.zero_grad()
            a = weight_sig_inv0 @ (weight_mu0 - theta)
            loss = -a.T @ wight_sig @ a
            loss.backward()
            optim_theta.step()

        beta_t_minus_1 = Pi_t_minus_1

        print(theta)

    def solve(self, s):
        astar = nn.Parameter(
            torch.tanh(torch.randn(s.shape[0], self.env.action_space.shape[0]))
        )

        if self.solvertype == "gd":
            self.optim = torch.optim.Adam([astar], lr=1e-1)
            for i in range(10):
                self.optim.zero_grad()
                loss = self.ucb(s, astar)
                loss.backward()
                self.optim.step()
        elif self.solvertype == "lbfgs":
            self.optim = torch.optim.LBFGS(
                [astar], history_size=30, max_iter=4, line_search_fn="strong_wolfe"
            )
            loss = self.ucb(s, astar)

            def closure():
                self.optim.zero_grad()
                loss.backward()
                return loss

            for i in range(10):
                self.optim.step(closure)
        else:
            print("No solver found!")

        return astar.clamp(min=self.a_min, max=self.a_max)


##########################################################################
class BAYESUCB(DeepDeterministicPolicyGradient):

    def __init__(self, env, args):
        super().__init__(env, args, None, BanditCriticNet)
        self.qminsolver = Qminimizer(
            env, self.critic_target, solvertype="gd", ucbtype=args.model
        )
        self.qminsolver._get_t = lambda: len(self.experience_memory)
        self.qminsolver._t_max = self.experience_memory.buffer_size

    def learn(self, max_iter=1):
        if self.args.batch_size > len(self.experience_memory):
            return None

        for ii in range(max_iter):
            s, a, r, sp, done, step = self.experience_memory.sample_random(
                self.args.batch_size
            )
            ap = self.qminsolver.solve(sp)
            qp, _ = self.critic_target(sp, ap)
            q_target = r.view(-1, 1) + self._gamma * qp.detach()

            self.critic_optim.zero_grad()
            td_error = self.critic.neg_log_prob(s, a, q_target.detach())
            td_error.backward()
            self.critic_optim.step()
            # self.critic.update_belief(s,a,q_target)

            self.hard_update(self.critic, self.critic_target)

    def select_action(self, s):
        s = torch.from_numpy(s).unsqueeze(0).float().to(self.device)
        a = self.qminsolver.solve(s)
        a = a.cpu().detach().clone().numpy().squeeze(0)
        return a
