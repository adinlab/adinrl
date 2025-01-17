import numpy as np
import torch

from models.basic.ac import ActorCritic
from models.basic.actor import Actor
from models.basic.loss import NormalWassersteinLossLVar
from models.basic.sequential.sequential_prob_critic import (
    ProbCritic,
    ProbCriticEnsemble,
)
from nets.actor_nets import ActorNetSmooth
from nets.sequential_critic_nets import SequentialCriticNetProbabilistic


#####################################################################
class DistCritic(ProbCritic):
    def __init__(self, arch, args, n_state, n_action):
        super().__init__(arch, args, n_state, n_action)
        self.loss = NormalWassersteinLossLVar()


#####################################################################


class DistCriticEnsemble(ProbCriticEnsemble):
    def __init__(self, arch, args, n_state, n_action, CriticType=DistCritic):
        super().__init__(arch, args, n_state, n_action, CriticType)
        self.perturb_actions = False

    def Q_dist(self, s, a):
        return self.get_reduced_distribution(s, a, is_target=False)

    @torch.no_grad()
    def get_bellman_target(self, r, sp, done, actor):
        act_func = actor.act_target if actor.has_target else actor.act
        ap, ep = act_func(sp)
        if self.perturb_actions:
            ap += (0.2 * torch.randn_like(ap)).clamp(-0.5, 0.5)
        mu_p_t, var_pe_t, var_pa_t = self.Q_t_dist(sp, ap)
        var_p_t = var_pe_t + var_pa_t
        y_mu = r.unsqueeze(-1) + (self.args.gamma * mu_p_t * (1 - done.unsqueeze(-1)))
        y_var = self.args.gamma**2 * var_p_t * (1 - done.unsqueeze(-1))
        return torch.cat([y_mu, y_var], dim=-1)

    def get_reduced_distribution(self, s, a, is_target=False):
        if self.bootstrap:
            n_batch = s.shape[0]
            n_boot = int(n_batch * self.perc_boot)
            get_ids = lambda: np.random.choice(n_batch, n_boot)

            if is_target:
                val = []
                for critic in self.critics:
                    ids = get_ids()
                    val.append(critic.target(s[ids], a[ids]))
            else:
                val = []
                for critic in self.critics:
                    ids = get_ids()
                    val.append(critic.model(s[ids], a[ids]))
        else:
            val = (
                [critic.target(s, a) for critic in self.critics]
                if is_target
                else [critic.model(s, a) for critic in self.critics]
            )
        val = torch.cat(val, dim=-1)
        mu_list, var_list = val[:, 0::2], val[:, 1::2].exp()
        idx = mu_list.argmin(dim=-1)
        # if is_target:
        if True:
            mu_e = mu_list.gather(1, idx.unsqueeze(-1))
            var_e = var_list.gather(1, idx.unsqueeze(-1))
        else:
            mu_e = mu_list.mean(dim=-1, keepdim=True)
            var_e = var_list.mean(dim=-1, keepdim=True) / len(self.critics)
        # else:
        var_a = torch.zeros_like(var_e)
        return mu_e, var_e, var_a


#####################################################################
class DistActor(Actor):
    def __init__(self, arch, args, n_state, n_action):
        super().__init__(arch, args, n_state, n_action, has_target=True)

    def loss(self, s, critics):
        a, e = self.act(s)
        q, q_var_e, q_var_a = critics.Q_dist(s, a)
        q_var = q_var_e + q_var_a + 1e-6
        q = q + q_var.sqrt()
        return (-q).mean(), e


#####################################################################
class SequentialDistributionalActorCritic(ActorCritic):
    _agent_name = "SequentialDAC"

    def __init__(
        self,
        env,
        args,
        actor_nn=ActorNetSmooth,
        critic_nn=SequentialCriticNetProbabilistic,
    ):
        super().__init__(
            env,
            args,
            actor_nn,
            critic_nn,
            CriticEnsembleType=DistCriticEnsemble,
            ActorType=DistActor,
        )
