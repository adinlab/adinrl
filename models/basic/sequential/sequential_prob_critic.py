import torch

from models.basic.sequential.sequential_critic import Critic, CriticEnsemble
from models.basic.loss import NormalMLELoss


#####################################################################
class ProbCritic(Critic):
    def __init__(self, arch, args, n_state, n_action):
        super().__init__(arch, args, n_state, n_action)
        self.loss = NormalMLELoss()

    def Q_t(self, s, a):
        return self.target(s, a)[:, 0].view(-1, 1)

    def Q(self, s, a):
        return self.model(s, a)[:, 0].view(-1, 1)

    def get_distribution(self, s, a, is_target=False):
        if is_target:
            out = self.target(s, a)
        else:
            out = self.model(s, a)
        mu = out[:, 0].view(-1, 1)
        logvar = out[:, 1].view(-1, 1)
        return mu, logvar

    def Q_dist(self, s, a):
        return self.get_distribution(s, a)

    def Q_t_dist(self, s, a):
        return self.get_distribution(s, a, is_target=True)

    def update(self, s, a, y):
        self.optim.zero_grad()
        mu, logvar = self.get_distribution(s, a)
        self.loss(mu, logvar, y).backward()
        self.optim.step()
        self.iter += 1


#####################################################################
class ProbCriticEnsemble(CriticEnsemble):
    def __init__(self, arch, args, n_state, n_action, CriticType=ProbCritic):
        super().__init__(arch, args, n_state, n_action, CriticType)
        self.bootstrap = False
        self.perc_boot = 0.5

    def get_reduced_distribution(self, s, a, is_target=False):
        if is_target:
            val = [critic.target(s, a) for critic in self.critics]
        else:
            val = [critic.model(s, a) for critic in self.critics]
        val = torch.cat(val, dim=-1)
        mu_list, var_list = val[:, 0::2], val[:, 1::2].exp()  # .clamp(-4.6, 4.6).exp()
        idx = mu_list.argmin(dim=-1)
        mu_e = mu_list.gather(1, idx.unsqueeze(-1))
        var_e = var_list.gather(1, idx.unsqueeze(-1))
        var_a = torch.zeros_like(var_e)
        return mu_e, var_e, var_a

    # First two moments of a mixture of two unimodal distributions
    def Q_dist(self, s, a):
        return self.get_reduced_distribution(s, a, is_target=False)

    def Q_t_dist(self, s, a):
        return self.get_reduced_distribution(s, a, is_target=True)

    def Q(self, s, a):
        mu, var_a, var_e = self.get_reduced_distribution(s, a, is_target=False)
        return mu + (var_a + var_e).sqrt() * torch.randn_like(mu)

    def Q_t(self, s, a):
        mu, var_a, var_e = self.get_reduced_distribution(s, a, is_target=True)
        return mu + (var_a + var_e).sqrt() * torch.randn_like(mu)

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
