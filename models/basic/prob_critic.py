import torch
import torch as th

from models.basic.critic import Critic, Critics
from models.basic.loss import NormalMLELoss


#####################################################################
class ProbCritic(Critic):
    def __init__(self, arch, args, n_state, n_action):
        super().__init__(arch, args, n_state, n_action)
        self.loss = NormalMLELoss()

    def Q_t(self, s, a):
        return self.target(th.cat((s, a), -1))[:, 0, None]

    def Q(self, s, a):
        return self.model(th.cat((s, a), -1))[:, 0, None]

    def get_distribution(self, s, a, is_target=False):
        sa = th.cat((s, a), -1)
        if is_target:
            out = self.target(sa)
        else:
            out = self.model(sa)
        mu = out[:, 0, None]
        lvar = out[:, 1, None]
        return mu, lvar

    def Q_dist(self, s, a):
        return self.get_distribution(s, a)

    def Q_t_dist(self, s, a):
        return self.get_distribution(s, a, is_target=True)

    def update(self, s, a, y):
        self.optim.zero_grad()
        mu, lvar = self.get_distribution(s, a)
        self.loss(mu, lvar, y).backward()
        self.optim.step()
        self.iter += 1


#####################################################################
class ProbCritics(Critics):
    def __init__(self, arch, args, n_state, n_action, CriticType=ProbCritic):
        super().__init__(arch, args, n_state, n_action, CriticType)

        self.loss = self.critics[0].loss

    def get_reduced_distribution(self, s, a, is_target=False):
        mu_list, var_list = self.get_distribution(s, a, is_target=is_target)

        idx = mu_list.argmin(0)
        mu_e = mu_list.gather(0, idx[None]).unsqueeze(-1)
        var_e = var_list.gather(0, idx[None]).unsqueeze(-1)
        var_a = torch.zeros_like(var_e)
        return mu_e, var_e, var_a

    def get_distribution(self, s, a, is_target=False):
        SA = self.expand(th.cat((s, a), -1))
        if is_target:
            val = self.forward_target(self.params_target, self.buffers_target, SA)
        else:
            val = self.forward_model(self.params_model, self.buffers_model, SA)

        # n_member x n_batch x (mean, var)
        mu = val[:, :, 0]
        lvar = val[:, :, 1]

        return mu, lvar

    # First two moments of a mixture of two unimodal distributions
    def Q_dist(self, s, a):
        return self.get_reduced_distribution(s, a, is_target=False)

    def Q_t_dist(self, s, a):
        return self.get_reduced_distribution(s, a, is_target=True)

    def Q(self, s, a, sampled=True):
        mu, var_a, var_e = self.get_reduced_distribution(s, a, is_target=False)
        if sampled:
            return mu + (
                (var_a + var_e).sqrt() * torch.randn_like(mu) if sampled else 0
            )
        else:
            return mu

    def Q_t(self, s, a, sampled=True):
        mu, var_a, var_e = self.get_reduced_distribution(s, a, is_target=True)
        if sampled:
            return mu + (
                (var_a + var_e).sqrt() * torch.randn_like(mu) if sampled else 0
            )
        else:
            return mu

    @torch.no_grad
    def get_bellman_target(self, r, sp, done, actor):
        alpha = actor.log_alpha.exp().detach() if hasattr(actor, "log_alpha") else 0
        ap, ep = actor.act(sp)
        if ep is None:
            ep = 0
        mu, var_e, var_a = self.Q_t_dist(sp, ap)
        qp_t = mu - alpha * ep
        y = r.unsqueeze(-1) + (self.args.gamma * qp_t * (1 - done.unsqueeze(-1)))
        return y

    def update(self, s, a, y):
        self.optim.zero_grad()
        mu, lvar = self.get_distribution(s, a)
        self.loss(mu, lvar, self.expand(y)).backward()
        self.optim.step()
        self.iter += 1
