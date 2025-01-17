import numpy as np
import torch.nn as nn

from loggers.logger import Logger
from replay_buffers.experience_memory import ExperienceMemoryTorch
from utils.utils import totorch


class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.env = env
        self.dim_obs, self.dim_act = (
            self.env.observation_space.shape,
            self.env.action_space.shape,
        )
        print(f"INFO: dim_obs = {self.dim_obs} dim_act = {self.dim_act}")
        self.dim_obs_flat, self.dim_act_flat = np.prod(self.dim_obs), np.prod(
            self.dim_act
        )
        self._u_min = totorch(self.env.action_space.low, device=self.device)
        self._u_max = totorch(self.env.action_space.high, device=self.device)
        self._x_min = totorch(self.env.observation_space.low, device=self.device)
        self._x_max = totorch(self.env.observation_space.high, device=self.device)

        self._gamma = args.gamma
        self._tau = args.tau

        args.dims = {
            "state": (args.buffer_size, self.dim_obs_flat),
            "action": (args.buffer_size, self.dim_act_flat),
            "next_state": (args.buffer_size, self.dim_obs_flat),
            "reward": (args.buffer_size),
            "terminated": (args.buffer_size),
            "step": (args.buffer_size),
        }

        self.experience_memory = ExperienceMemoryTorch(args)
        self.logger = Logger(args)

    def set_writer(self, writer):
        self.writer = writer

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.mul_(1.0 - self.args.tau)
            target_param.data.add_(self.args.tau * local_param.data)

    def _hard_update(self, local_model, target_model):

        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(local_param.data)

    def learn(self, max_iter=1):
        raise NotImplementedError(f"learn() not implemented for {self.name} agent")

    def select_action(self, warmup=False, exploit=False):
        raise NotImplementedError(
            f"select_action() not implemented for {self.name} agent"
        )

    def store_transition(self, s, a, r, sp, terminated, truncated, step):
        self.experience_memory.add(s, a, r, sp, terminated, step)
        self.actor.set_episode_status(terminated or truncated)
