import torch
import torch as th
from torch import nn
import numpy
from utils.utils import tonumpy


#####################################################################
class Actor(nn.Module):
    def __init__(self, arch, args, n_state, n_action, has_target=False):
        super().__init__()
        self.model = arch(
            n_state,
            n_action,
            depth=args.depth_actor,
            width=args.width_actor,
            act=args.act_actor,
            has_norm=not args.no_norm_actor,
        ).to(args.device)
        self.optim = torch.optim.Adam(self.model.parameters(), args.learning_rate)
        self.args = args
        self.has_target = has_target
        self.iter = 0
        self.is_episode_end = False
        self.states = []
        self.print_freq = 500

        if has_target:
            self.target = arch(
                n_state,
                n_action,
                depth=args.depth_actor,
                width=args.width_actor,
                act=args.act_actor,
                has_norm=not args.no_norm_actor,
            ).to(args.device)
            self.init_target()

    def init_target(self):
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(local_param.data)

    def set_writer(self, writer):
        self.writer = writer

    def act(self, s, is_training=True):
        a, e = self.model(
            s, is_training=is_training
        )  # a: action, e: log_prob(action|state)

        if is_training:
            if self.args.verbose and self.iter % self.print_freq == 0:
                self.states.append(tonumpy(s))

        return a, e

    def act_target(self, s):
        a, e = self.target(s)
        return a, e

    def set_episode_status(self, is_end):
        self.is_episode_end = is_end

    @th.no_grad()
    def update_target(self):
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters()
        ):
            # target_param.data.copy_(
            #     self.args.tau * local_param.data
            #     + (1.0 - self.args.tau) * target_param.data
            # )
            target_param.data.mul_(1.0 - self.args.tau)
            target_param.data.add_(self.args.tau * local_param.data)

    def loss(self, s, critics):
        a, _ = self.act(s)
        q_list = critics.Q(s, a)
        q = critics.reduce(q_list)
        return (-q).mean(), None

    def update(self, s, critics):
        self.optim.zero_grad()
        loss, _ = self.loss(s, critics)
        loss.backward()
        self.optim.step()

        if self.has_target:
            self.update_target()

        self.iter += 1

    def save_actor_params(self, path):
        params = {
            "params_model": self.model.state_dict(),
        }

        params_th = {
            k: v if isinstance(v, torch.Tensor) else v  # Ensure the values are tensors
            for k, v in params.items()
        }

        torch.save(params_th, path)
