import torch
import torch as th

from models.basic.ac import ActorCritic
from models.basic.loss import WassersteinLoss
from models.basic.prob_critic import ProbCritic, ProbCritics
from models.sequential.sequential_dac import DistActor
from nets.actor_nets import ActorNetSmooth
from nets.critic_nets import CriticNetProbabilistic


class DistCritic(ProbCritic):
    def __init__(self, arch, args, n_state, n_action):
        super().__init__(arch, args, n_state, n_action)
        self.loss = WassersteinLoss()


class DistributionalActorCritic(ActorCritic):
    _agent_name = "DAC"

    def __init__(
        self,
        env,
        args,
        actor_nn=ActorNetSmooth,
        critic_nn=CriticNetProbabilistic,
    ):
        super().__init__(
            env,
            args,
            actor_nn,
            critic_nn,
            CriticEnsembleType=DistCritics,
            ActorType=DistActor,
        )


class DistCritics(ProbCritics):
    def __init__(self, arch, args, n_state, n_action, CriticType=DistCritic):
        super().__init__(arch, args, n_state, n_action, CriticType)
        self.perturb_actions = False

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
        return torch.cat((y_mu, y_var), dim=-1)

    def get_distribution(self, s, a, is_target=False, reduced=True):
        SA = self.expand(th.cat((s, a), -1))
        if is_target:
            val = self.forward_target(self.params_target, self.buffers_target, SA)
        else:
            val = self.forward_model(self.params_model, self.buffers_model, SA)

        # n_member x n_batch x (mean, var)
        mu_list = val[:, :, 0]
        var_list = val[:, :, 1].clamp(-4.6, 4.6).exp()

        if reduced:
            if True:
                idx = mu_list.argmin(0)
                mu_e = mu_list.gather(0, idx[None])
                var_e = var_list.gather(0, idx[None])
            else:
                mu_e = mu_list.mean(dim=0, keepdim=True)
                var_e = var_list.mean(dim=0, keepdim=True) / len(self.critics)

            # n_nbatch x 1 -> the reduced distribution has no members anymore
            mu_e = mu_e.squeeze(0).unsqueeze(-1)
            var_e = var_e.squeeze(0).unsqueeze(-1)
            var_a = torch.zeros_like(var_e)
            return mu_e, var_e, var_a
        else:
            return mu_list, var_list, th.zeros_like(var_list)

    def update(self, s, a, y):  # y denotes bellman target
        self.optim.zero_grad()
        mu, var_e, var_a = self.Q_dist(s, a, reduced=False)
        var = var_e + var_a
        loss = self.loss(mu, var, self.expand(y))
        loss.backward()
        self.optim.step()
        self.iter += 1
