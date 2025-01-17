import torch
import torch as th

from models.bootdqn import BootDQN, BootstrapEnsembleLoss, BootDQNCritic, BootDQNCritics

from models.basic.prob_critic import ProbCritics, ProbCritic
from models.basic.actor import Actor
from nets.actor_nets import ActorNetSmooth
from nets.critic_nets import CriticNetProbabilistic


#####################################################################
class BENLoss(BootstrapEnsembleLoss):

    def __init__(self, args):
        super().__init__(args)

    def get_kl(self, mu1, var1, mu2, var2):
        return var2.log() - var1.log() + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5

    def forward(self, q_mu, q_lvar, y):
        mu = y[:, 0][None]
        lvar = y[:, 1][None]

        bootstrap_mask = (torch.rand_like(q_mu[:-1]) >= self.bootstrap_rate) * 1.0

        KL = (
            self.get_kl(
                mu, lvar.exp().clip(1e-4, 10), q_mu, q_lvar.exp().clip(1e-4, 10)
            )[:-1]
            * bootstrap_mask
        )

        fit = (q_mu.mean(0).detach() - q_mu[-1]).pow(2).mean()

        q_loss = fit + KL.mean()

        return q_loss


class BENCritic(ProbCritic):
    def __init__(self, arch, args, n_state, n_action):
        super().__init__(arch, args, n_state, n_action)
        self.loss = BENLoss(args)


##################################################################################################
class BENCritics(ProbCritics):
    def __init__(self, arch, args, n_state, n_action, critictype=BENCritic):
        super().__init__(arch, args, n_state, n_action, critictype)

        self.n_in_target = 2

    def get_distribution(self, s, a, is_target=False):
        SA = self.expand(th.cat((s, a), -1))
        if is_target:
            val = self.forward_target(self.params_target, self.buffers_target, SA)
        else:
            val = self.forward_model(self.params_model, self.buffers_model, SA)

        # print("val")
        # print(val.shape)
        # n_member x n_batch x (mean, var)
        mu = val[:, :, 0]
        lvar = val[:, :, 1]
        lvar[-1] = -th.inf  # the last layer is deterministic, i.e., has zero variance

        return mu, lvar

    def get_reduced_distribution(self, s, a, is_target=False):
        SA = self.expand(torch.cat((s, a), -1))
        if is_target:
            val = self.forward_target(self.params_target, self.buffers_target, SA)
        else:
            val = self.forward_model(self.params_model, self.buffers_model, SA)

        # n_member x n_batch x (mean, var)
        mu_list = val[:, :, 0][:-1]
        var_list = val[:, :, 1].exp()[:-1].clip(1e-4, 10)

        # Last one is deterministic
        mu_e = mu_list.mean(0).unsqueeze(-1)
        var_e = (mu_list.pow(2).mean(0) - mu_list.mean(0).pow(2)).unsqueeze(-1)
        var_a = var_list.mean(0).unsqueeze(-1)

        return mu_e, var_e, var_a

    @torch.no_grad()
    def get_bellman_target(self, r, sp, done, actor):
        ap, _ = actor.act(sp)

        mu_p_t, var_t_e, var_t_a = self.Q_t_dist(sp, ap)
        var_t = var_t_e + var_t_a

        d = done.unsqueeze(-1)
        r = r.unsqueeze(-1)

        y_mu = r + self.args.gamma * mu_p_t * (1 - d)
        y_var = (var_t * (1 - d) * self.args.gamma**2).clamp(1e-2, 1e2)

        return torch.cat((y_mu, y_var), dim=-1)

    def update(self, s, a, y):
        self.optim.zero_grad()
        mu, lvar = self.get_distribution(s, a)
        self.loss(mu, lvar, y).backward()
        self.optim.step()
        self.iter += 1


#####################################################################
class BENActor(Actor):
    def __init__(self, arch, args, n_state, n_action):
        super().__init__(arch, args, n_state, n_action, has_target=True)
        self.iter = 1
        self.sigma_scale = 1.0

    def loss(self, s, critics):
        a, e = self.act(s)
        # q_mu, q_var_e, q_var_a = critics.Q(s, a)
        # q_var = q_var_e + q_var_a
        #
        # q = q_mu + self.sigma_scale * torch.sqrt(q_var)
        # print(q.shape)
        # print(a.shape)

        SA = critics.expand(th.cat((s, a), -1))
        # Pick the mean of the deterministic critic
        # TODO: Optimize that it doesn't need to forward propagate through everything
        q = critics.forward_model(critics.params_model, critics.buffers_model, SA)[
            -1, :, 0
        ]
        # print(q.shape)

        # self.iter += 1
        return (-q).mean(), e


class BEN(BootDQN):
    _agent_name = "BEN"

    def __init__(self, env, args, CriticEnsembleType=BENCritics):
        super().__init__(
            env,
            args,
            CriticEnsembleType=CriticEnsembleType,
            ActorType=BENActor,
            actor_nn=ActorNetSmooth,
            critic_nn=CriticNetProbabilistic,
        )
