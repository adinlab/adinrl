import torch

from models.basic.loss import NormalLLLoss, McAllester
from models.basic.sequential.sequential_prob_critic import (
    ProbCritic,
    ProbCriticEnsemble,
)
from models.sequential.sequential_sac import (
    SequentialSoftActor,
    SequentialSoftActorCritic,
)
from nets.actor_nets import ActorNetProbabilistic
from nets.sequential_critic_nets import CriticNetEpistemic


class ShootingActor(SequentialSoftActor):
    def __init__(self, arch, args, n_state, n_action):
        super().__init__(arch, args, n_state, n_action)

    def set_critics(self, critics):
        self.critics = critics

    def act(self, s, is_training=True):
        if not is_training:
            s = s.repeat(self.args.num_shooting, 1)

        a, e = self.model(s, is_training=is_training)

        if not is_training:  # then shoot
            q_list = self.calculateQ(s, a)
            i_best = torch.argsort(q_list[:, 0].view(-1), descending=True)[0]
            a = a[i_best].view(1, -1)

        return a, e

    def loss(self, s, critics):
        a, e = self.act(s)
        q = critics.Q(s, a)
        return (-q + self.log_alpha.exp() * e).mean(), e

    @torch.no_grad()
    def calculateQ(self, s, a_list):
        mu, lvar_e, _ = self.critics.Q_dist(s, a_list)
        var = torch.exp(0.5 * lvar_e).clamp_(1e-2, 1e2)
        q_list = mu.unsqueeze(-1) + var.unsqueeze(-1) * torch.randn(
            (*mu.size(), 50)
        ).to(self.args.device)
        q_list = q_list.mean(dim=-1)
        return q_list


class PAC4SACCritic(ProbCritic):
    def __init__(self, arch, args, n_state, n_action):
        super().__init__(arch, args, n_state, n_action)

        self.batch_size = args.batch_size
        self.xi = 0.01

    def set_buffer(self, buffer):
        self.buffer = buffer

    def loss(self, mu, lvar, y_all):
        y = y_all[0]
        sp = y_all[1]
        ap = y_all[2]
        q_p = self.model(sp, ap)
        qp_var = q_p[:, 1].clamp(-4.6, 4.6).exp()
        N = self.buffer.size
        return (
            NormalLLLoss()(mu, lvar, y)
            + McAllester()(self.model, N)
            - self.xi * qp_var.mean()
        )


class PAC4SACCriticEnsemble(ProbCriticEnsemble):
    def __init__(self, arch, args, n_state, n_action, critictype=PAC4SACCritic):
        super().__init__(arch, args, n_state, n_action, critictype)

    def set_buffer(self, buffer):
        [critic.set_buffer(buffer) for critic in self.critics]

    @torch.no_grad
    def get_bellman_target(self, r, sp, done, actor):
        alpha = actor.log_alpha.exp().detach() if hasattr(actor, "log_alpha") else 0
        ap, ep = actor.act(sp)
        if ep is None:
            ep = 0
        mu, var_e, var_a = self.Q_t_dist(sp, ap)
        qp_t = mu - alpha * ep
        y = r.unsqueeze(-1) + (self.args.gamma * qp_t * (1 - done.unsqueeze(-1)))
        y_all = list()
        y_all.append(y)
        y_all.append(sp)
        y_all.append(ap)
        return y_all


class PAC4SAC(SequentialSoftActorCritic):
    _agent_name = "PAC4SAC"

    def __init__(
        self, env, args, actor_nn=ActorNetProbabilistic, critic_nn=CriticNetEpistemic
    ):
        super().__init__(
            env,
            args,
            actor_nn,
            critic_nn,
            CriticEnsembleType=PAC4SACCriticEnsemble,
            ActorType=ShootingActor,
        )

        self.num_shooting = args.num_shooting
        self.actor.set_critics(self.critics)
        self.critics.set_buffer(self.experience_memory)
