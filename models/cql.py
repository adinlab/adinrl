import torch
from torch import nn
import numpy as np
from torchmin import Minimizer

from models.ddpg import DeepDeterministicPolicyGradient
from utils.implicit import AbstractProblem


class Qminimizer(AbstractProblem):
    def __init__(self, env, model, solvertype="gd"):
        super().__init__()
        self.model = model
        self.env = env
        self.a_min = self.env.action_space.low[0]
        self.a_max = self.env.action_space.high[0]
        self.solvertype = solvertype

    def solve(self, s):
        astar = nn.Parameter(
            torch.tanh(torch.randn(s.shape[0], self.env.action_space.shape[0]))
        )

        if self.solvertype == "newton":
            optimizer = Minimizer([astar], method="bfgs", max_iter=10)

            def closure():
                optimizer.zero_grad()
                return -self.model(
                    s, astar.clamp(min=self.a_min, max=self.a_max)
                ).mean()

            optimizer.step(closure)
        elif self.solvertype == "gd":
            self.optim = torch.optim.Adam([astar], lr=1e-1)
            for i in range(10):
                self.optim.zero_grad()
                loss = -self.model(
                    s, astar.clamp(min=self.a_min, max=self.a_max)
                ).mean()
                loss.backward()
                self.optim.step()
        elif self.solvertype == "discrete":
            astar = torch.zeros_like(astar)
            N = s.shape[0]
            for i in range(N):
                a_s = np.linspace(self.a_min, self.a_max, 41)
                a_s = torch.from_numpy(a_s).unsqueeze(-1).float()
                q_s = self.model(s[i].repeat(41, 1), a_s)
                astar[i] = a_s[q_s.argmax()].reshape(1, 1)

        return astar


class ContinuousQLearning(DeepDeterministicPolicyGradient):

    def __init__(self, env, args, critic_nn, solvertype="gd"):
        super().__init__(env, args, None, critic_nn)
        self.qminsolver = Qminimizer(env, self.critic_target, solvertype=solvertype)

    def critic_learn(self, s, a, r, sp, done):
        q = self.critic.forward(s, a)
        ap = self.qminsolver.solve(sp)
        qp = self.critic_target.forward(sp, ap)
        q_target = r.view(-1, 1) + self._gamma * qp * (1 - done.view(-1, 1))
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
            self.critic_learn(s, a, r, sp, done)
            self.hard_update(self.critic, self.critic_target)

    def select_action(self, s):
        s = torch.from_numpy(s).unsqueeze(0).float().to(self.device)
        a = self.qminsolver.solve(s)
        a = a.cpu().detach().clone().numpy().squeeze(0)

        # if self.step_counter == 0:
        #    self.noise_func.reset()

        # a = self.noise_func.get_action(a, self.episode_counter*self.step_counter)
        # self.step_counter += 1
        # self.episode_counter = self.step_counter % self._env._max_episode_steps

        return a
