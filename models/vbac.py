import torch
from models.basic.ac import ActorCritic
from models.basic.sequential.sequential_prob_critic import (
    ProbCritic,
    ProbCriticEnsemble,
)
from models.basic.actor import Actor
from nets.sequential_critic_nets import CriticNetBayesLinear
from nets.actor_nets import ActorNetProbabilistic
from models.sac import SequentialSoftActor
from models.basic.loss import VFELossTotal
import numpy as np
from torch.special import erfinv
import torch.nn as nn
from torch.autograd import grad


#####################################################################
class VBCritic(ProbCritic):  # ProbCritic
    def __init__(self, arch, args, n_state, n_action):
        super().__init__(arch, args, n_state, n_action)
        self.loss = VFELossTotal()
        self.quantile_delay = 1000
        self.p = torch.nn.Parameter(
            torch.tensor(0, requires_grad=True, device=args.device, dtype=torch.float32)
        )
        self.optim_p = torch.optim.Adam([self.p], lr=args.learning_rate)
        self.n_data = (
            args.buffer_size
        )  # for now a suboptimal helper variable to properly regularize our model

    def update(self, s, a, y):
        self.optim.zero_grad()
        q_mu, q_logvar = self.Q_dist(s, a)
        mu_t_ = y[0]
        var_t_ = y[1]
        done = y[2]

        erf = torch.erfinv(2 * torch.sigmoid(self.p) - 1)
        y_t = mu_t_ + np.sqrt(2) * torch.sqrt(var_t_) * erf * (1 - done.unsqueeze(-1))
        self.loss(q_mu, q_logvar, y_t.detach(), self.model, self.n_data).backward()
        self.optim.step()
        if self.iter % self.quantile_delay == 0:
            self.update_p(y_t, q_mu.detach())
        self.iter = self.iter + 1

    def update_p(self, y, q_mu):
        self.optim_p.zero_grad()
        # print("before",self.p)
        loss_p = ((q_mu - y) ** 2).mean()
        loss_p.backward()
        self.optim_p.step()
        # print("after",self.p)


#####################################################################
class VBCriticEnsemble(ProbCriticEnsemble):  # ProbCriticEnsemble
    def __init__(self, arch, args, n_state, n_action, CriticType=VBCritic):
        super().__init__(arch, args, n_state, n_action, CriticType)

    def get_reduced_distribution(self, s, a, is_target=False):
        if is_target:
            val = [critic.target(s, a) for critic in self.critics]
        else:
            val = [critic.model(s, a) for critic in self.critics]
        val = torch.cat(val, dim=-1)
        mu_list, var_list = val[:, 0::2], val[:, 1::2].exp()  # .clamp(-4.6, 4.6).exp()
        # idx = mu_list.argmin(dim=-1)

        mu_e = mu_list.mean(dim=1).unsqueeze(-1)  # .gather(1, idx.unsqueeze(-1))
        var_e = var_list.mean(dim=1).unsqueeze(
            -1
        )  # .gather(1, idx.unsqueeze(-1)) + 1e-4
        var_a = torch.zeros_like(var_e)
        return mu_e, var_e, var_a

    @torch.no_grad
    def get_bellman_target(self, r, sp, done, actor):
        alpha = actor.log_alpha.exp().detach() if hasattr(actor, "log_alpha") else 0
        ap, ep = actor.act(sp)
        if ep is None:
            ep = 0
        mu, var_e, var_a = self.Q_t_dist(sp, ap)
        qp_t = mu - alpha * ep
        mu_t = r.unsqueeze(-1) + (self.args.gamma * qp_t * (1 - done.unsqueeze(-1)))
        var_t = self.args.gamma**2 * (var_e + var_a)  # + 1e-4
        return mu_t, var_t, done


#####################################################################
class ProbSoftActor(SequentialSoftActor):
    def __init__(self, arch, args, n_state, n_action):
        super().__init__(arch, args, n_state, n_action)

    # def loss(self, s, critics):
    #    a, e = self.act(s)
    #    q_mu, q_var_e, q_var_a = critics.Q_dist(s, a)
    #    eps = torch.randn_like(q_mu)
    #    q_var = q_var_e + q_var_a
    #    q_var = q_var.clamp(1e-4, None)
    #    q = q_mu + eps * torch.sqrt(q_var)
    #    return (-q).mean(), e

    def loss(self, s, critics):
        a, e = self.act(s)
        q_mu, q_var_e, q_var_a = critics.Q_dist(s, a)
        return (-q_mu + self.log_alpha.exp() * e).mean(), e


#####################################################################
class VariationalBayesianAC(ActorCritic):
    _agent_name = "VBAC"

    def __init__(
        self, env, args, actor_nn=ActorNetProbabilistic, critic_nn=CriticNetBayesLinear
    ):
        super().__init__(
            env,
            args,
            actor_nn,
            critic_nn,
            CriticEnsembleType=VBCriticEnsemble,
            ActorType=ProbSoftActor,
        )

        self.actor.c = 0.05
