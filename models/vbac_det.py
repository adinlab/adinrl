import torch
from models.basic.ac import ActorCritic
from models.basic.actor import Actor
from nets.actor_nets import ActorNetProbabilistic
from models.sac import SequentialSoftActor
from models.basic.loss import VFELossTotal
import numpy as np
from torch.special import erfinv
import torch.nn as nn
from torch.autograd import grad
from models.basic.ac import ActorCritic

# from models.basic.sequential.sequential_critic import CriticEnsemble, Critic
# from nets.sequential_critic_nets import SequentialCriticNet
from models.basic.critic import Critics, Critic
from nets.critic_nets import CriticNet


#####################################################################
class VBCritic(Critic):  # ProbCritic
    def __init__(self, arch, args, n_state, n_action):
        super().__init__(arch, args, n_state, n_action)


#####################################################################
class VBCriticEnsemble(Critics):  # CriticEnsemble  Critics
    def __init__(self, arch, args, n_state, n_action, CriticType=VBCritic):
        super().__init__(arch, args, n_state, n_action, CriticType)
        self.quantile_delay = 1000
        self.p = torch.nn.Parameter(
            torch.tensor(0, requires_grad=True, device=args.device, dtype=torch.float32)
        )
        self.optim_p = torch.optim.Adam([self.p], lr=args.learning_rate)
        self.n_data = args.buffer_size

    @torch.no_grad
    def get_bellman_target(self, r, sp, done, actor):
        alpha = actor.log_alpha.exp().detach() if hasattr(actor, "log_alpha") else 0
        ap, ep = actor.act(sp)
        if ep is None:
            ep = 0
        mu = self.Q_t(sp, ap)
        # mu_new = torch.cat(mu, dim=1).mean(dim=1, keepdim=True)
        # var_new = torch.cat(mu, dim=1).var(dim=1, keepdim=True).clamp(1e-6, None)
        mu_new = mu.mean(dim=0)
        var_new = mu.var(dim=0).clamp(1e-6, None)
        qp_t = mu_new - alpha * ep
        mu_t = r.unsqueeze(-1) + (self.args.gamma * qp_t * (1 - done.unsqueeze(-1)))
        var_t = self.args.gamma**2 * (var_new)  # + 1e-4   .var(dim=0).clamp(1e-6, None)
        return mu_t, var_t, done

    def update(self, s, a, y):
        self.optim.zero_grad()
        q_mu = self.Q(s, a).mean(dim=0)
        mu_t_ = y[0]
        var_t_ = y[1]
        done = y[2]

        erf = torch.erfinv(2 * torch.sigmoid(self.p) - 1)
        y_t = mu_t_ + np.sqrt(2) * torch.sqrt(var_t_) * erf * (1 - done.unsqueeze(-1))
        # self.loss(q_mu, q_logvar, y_t.detach(), self.model, self.n_data).backward()
        nn.MSELoss()(q_mu, mu_t_).backward()
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
        q_list = critics.Q(s, a)
        # q_mu = critics.reduce(q_list)
        q_mu = q_list.mean(dim=0)
        return (-q_mu + self.log_alpha.exp() * e).mean(), e


#####################################################################
class VariationalBayesianACDet(ActorCritic):
    _agent_name = "VBAC_det"

    def __init__(
        self,
        env,
        args,
        actor_nn=ActorNetProbabilistic,
        critic_nn=CriticNet,  # SequentialCriticNet   CriticNet
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
