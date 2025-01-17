import torch
import torch as th
from torch import nn


#####################################################################
class Critic(nn.Module):
    def __init__(self, arch, args, n_state, n_action):
        super().__init__()
        self.args = args
        self.model = arch(n_state, n_action, args.n_hidden).to(args.device)
        self.target = arch(n_state, n_action, args.n_hidden).to(args.device)
        self.init_target()
        self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), args.learning_rate)
        self.iter = 0

    def set_writer(self, writer):
        self.writer = writer

    def init_target(self):
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(local_param.data)

    @th.no_grad()
    def update_target(self):
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters()
        ):
            target_param.data.mul_(1.0 - self.args.tau)
            target_param.data.add_(self.args.tau * local_param.data)

    def Q(self, s, a):
        return self.model(s, a)

    def Q_t(self, s, a):
        return self.target(s, a)

    def update(self, s, a, y):  # y denotes bellman target
        self.optim.zero_grad()
        loss = self.loss(self.Q(s, a), y)
        loss.backward()
        self.optim.step()
        self.iter += 1


#####################################################################
class CriticEnsemble(nn.Module):
    def __init__(self, arch, args, n_state, n_action, critictype=Critic):
        super().__init__()
        self.n_elements = args.n_critics
        self.args = args
        self.critics = [
            critictype(arch, args, n_state, n_action) for _ in range(self.n_elements)
        ]
        self.iter = 0

    def __getitem__(self, item):
        return self.critics[item]

    def set_writer(self, writer):
        self.writer = writer
        [critic.set_writer(writer) for critic in self.critics]

    def Q(self, s, a):
        return [critic.Q(s, a) for critic in self.critics]

    def Q_t(self, s, a):
        return [critic.Q_t(s, a) for critic in self.critics]

    def update(self, s, a, y):
        [critic.update(s, a, y) for critic in self.critics]
        self.iter += 1

    def update_target(self):
        [critic.update_target() for critic in self.critics]

    def reduce(self, q_val_list):
        return torch.stack(q_val_list, dim=-1).min(dim=-1)[0]

    @torch.no_grad()
    def get_bellman_target(self, r, sp, done, actor):
        alpha = actor.log_alpha.exp().detach() if hasattr(actor, "log_alpha") else 0
        ap, ep = actor.act(sp)
        qp = self.Q_t(sp, ap)
        if ep is None:
            ep = 0
        qp_t = self.reduce(qp) - alpha * ep
        y = r.unsqueeze(-1) + (self.args.gamma * qp_t * (1 - done.unsqueeze(-1)))
        return y
