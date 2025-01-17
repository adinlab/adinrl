import torch
from torch import nn
import numpy as np

from agents.base_agent import Agent


class OUNoise(object):
    def __init__(
        self,
        action_space,
        mu=0.0,
        theta=0.15,
        max_sigma=0.3,
        min_sigma=0.3,
        decay_period=100000,
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self) -> object:
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        return np.clip(action + ou_state, self.low, self.high)


class DeepDeterministicPolicyGradient(Agent):
    def __init__(self, env, args, actor_nn, critic_nn):
        super().__init__(env, args)

        self._alpha = 0.05
        self.critic_loss_fcn = nn.MSELoss()
        #
        if actor_nn is not None:
            self.actor = actor_nn(self.dim_obs, self.dim_act, args.n_hidden).to(
                self.device
            )
            self.actor_target = actor_nn(self.dim_obs, self.dim_act, args.n_hidden).to(
                self.device
            )
            self.hard_update(
                self.actor, self.actor_target
            )  # hard update at the beginning
            self.actor_optim = torch.optim.Adam(
                self.actor.parameters(), args.learning_rate
            )

        self.critic = critic_nn(self.dim_obs, self.dim_act, args.n_hidden).to(
            self.device
        )
        self.critic_target = critic_nn(self.dim_obs, self.dim_act, args.n_hidden).to(
            self.device
        )
        self.hard_update(
            self.critic, self.critic_target
        )  # hard update at the beginning
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), args.learning_rate
        )

        #
        self.episode_counter = 0
        self.step_counter = 0
        self.noise_func = OUNoise(self.env.action_space, min_sigma=0)
        self.noise_func.reset()

    def actor_learn(self, s):
        a, _ = self.actor.forward(s)
        q = self.critic.forward(s, a)
        a_loss = torch.mean(-q)

        self.actor_optim.zero_grad()
        a_loss.backward()
        self.actor_optim.step()

    def critic_learn(self, s, a, r, sp, done):
        q = self.critic.forward(s, a)
        ap, _ = self.actor_target.forward(sp)
        q_ = self.critic_target.forward(sp, ap.detach()).detach()
        q_target = r.view(-1, 1) + (self._gamma * q_) * (1 - done.view(-1, 1))
        td_error = self.critic_loss_fcn(q_target, q)

        self.critic_optim.zero_grad()
        td_error.backward()
        self.critic_optim.step()

    def learn(self, max_iter=1):
        if self.args.batch_size > len(self.experience_memory):
            return None

        for ii in range(max_iter):
            s, a, r, sp, done, step = self.experience_memory.sample_random(
                self.args.batch_size
            )
            self.actor_learn(s)
            self.critic_learn(s, a, r, sp, done)
            # soft update of target critic networks
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic, self.critic_target)

    @torch.no_grad()
    def select_action(self, s):
        s = torch.from_numpy(s).unsqueeze(0).float().to(self.device)
        a, _ = self.actor(s)
        a = a.cpu().numpy().squeeze(0)

        if self.step_counter == 0:
            self.noise_func.reset()

        a = self.noise_func.get_action(a, self.episode_counter * self.step_counter)
        self.step_counter += 1
        self.episode_counter = self.step_counter % self.env.max_episode_steps

        return a

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            # target_param.data.copy_(
            #     self._tau * local_param.data + (1.0 - self._tau) * target_param.data
            # )
            target_param.data.mul_(1.0 - self.args.tau)
            target_param.data.add_(self.args.tau * local_param.data)

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            # target_param.data.copy_(local_param.data)
            target_param.data = local_param.data.clone()
            # local_param.data = target_param.data.clone()
