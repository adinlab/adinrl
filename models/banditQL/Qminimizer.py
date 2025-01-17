import torch
import torch.nn as nn
import numpy as np

from utils.implicit import AbstractProblem


class Qminimizer(AbstractProblem):
    def __init__(self, env, model, solvertype="gd", std_weight=0):
        super().__init__()
        self.model = model
        self.env = env
        self.a_min = self.env.action_space.low[0]
        self.a_max = self.env.action_space.high[0]
        self.solvertype = solvertype
        self.std_weight = std_weight

    def solve(self, s, **kwargs):

        if self.solvertype == "gd":
            astar = nn.Parameter(
                (torch.zeros(s.shape[0], self.env.action_space.shape[0]))
            )
            self.optim = torch.optim.Adam([astar], lr=5e-2)
            for i in range(50):
                self.optim.zero_grad()
                loss = -self.model.ucb(
                    s=s, a=astar.clamp(min=self.a_min, max=self.a_max), **kwargs
                ).mean()
                loss.backward()
                self.optim.step()
        elif self.solvertype == "discrete":
            astar = nn.Parameter(
                (torch.zeros(s.shape[0], self.env.action_space.shape[0]))
            )
            astar = torch.zeros_like(astar)
            N = s.shape[0]
            n_points = 41
            if N > 1:
                all_kwargs = kwargs.copy()
            for i in range(N):
                a_s = np.linspace(self.a_min, self.a_max, n_points)
                a_s = torch.from_numpy(a_s).unsqueeze(-1).float()
                if N > 1:
                    kwargs["t"] = all_kwargs["t"][i]
                q_s = self.model.ucb(s[i].repeat(n_points, 1), a_s, **kwargs)
                # if q_s has anny nan value assert
                if torch.isnan(q_s).any():
                    raise ValueError("NaN value in q_s!")
                q_ind = q_s.argmax()
                astar[i] = a_s[q_ind].reshape(1, 1)
                self.seleceed_q = q_ind

        # elif self.solvertype == "discrete_actions":
        #     N = s.shape[0]
        #     astar = torch.zeros(s.shape[0], 1)
        #     if N > 1:
        #         all_kwargs = kwargs.copy()
        #     for i in range(N):
        #         a_s = np.linspace(0, 2, 3)
        #         a_s = torch.from_numpy(a_s).unsqueeze(-1).float()
        #         if N > 1:
        #             kwargs["t"] = all_kwargs["t"][i]
        #         q_s = self.model.ucb(s[i].repeat(3, 1), a_s, **kwargs)
        #         # if q_s has anny nan value assert
        #         if torch.isnan(q_s).any():
        #             raise ValueError("NaN value in q_s!")
        #         q_ind = q_s.argmax()
        #         astar[i] = a_s[q_ind].reshape(1,1)
        #         self.seleceed_q = q_ind
        #     return astar

        else:
            raise NotImplementedError(f"No solver found: {self.solvertype}!")

        return astar.clamp(min=self.a_min, max=self.a_max)
