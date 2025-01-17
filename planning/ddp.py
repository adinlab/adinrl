import numpy as np
import torch

from planning.base_planning_algo import PlanningAlgo


class DDP(PlanningAlgo):
    _agent_name = "DDP"

    def __init__(self, env, args, model_nn, cost_nn, critic_nn):
        self.env = env
        self.x_n = self.env.observation_space.shape[0]
        self.u_n = self.env.action_space.shape[0]
        self.predict_window = args.planning_horizon
        self.max_iter = 3

        self.v = torch.zeros(self.predict_window + 1)
        self.v_x = torch.zeros((self.predict_window + 1, self.x_n))
        self.v_xx = torch.zeros((self.predict_window + 1, self.x_n, self.x_n))

        self.f = model_nn
        self.running_cost = cost_nn
        self.final_cost = critic_nn

        self.final_cost_x = lambda x: torch.autograd.functional.jacobian(
            self.final_cost, x
        )  # torch.autograd.grad(self.final_cost)
        self.final_cost_xx = lambda x: torch.autograd.functional.hessian(
            self.final_cost, x
        )  # torch.autograd.functional.jacobian(self.final_cost_x)
        self.running_cost_x = lambda x, u: torch.autograd.functional.jacobian(
            self.running_cost, inputs=(x, u)
        )[
            0
        ]  # torch.autograd.grad(self.running_cost, 0)
        self.running_cost_u = lambda x, u: torch.autograd.functional.jacobian(
            self.running_cost, inputs=(x, u)
        )[
            1
        ]  # torch.autograd.grad(self.running_cost, 1)
        self.running_cost_xx = lambda x, u: torch.autograd.functional.hessian(
            self.running_cost, inputs=(x, u)
        )[0][
            0
        ]  # torch.autograd.functional.jacobian(self.running_cost_x, 0)
        self.running_cost_uu = lambda x, u: torch.autograd.functional.hessian(
            self.running_cost, inputs=(x, u)
        )[1][
            1
        ]  # torch.autograd.functional.jacobian(self.running_cost_u, 1)
        self.running_cost_ux = lambda x, u: torch.autograd.functional.hessian(
            self.running_cost, inputs=(x, u)
        )[1][
            0
        ]  # torch.autograd.functional.jacobian(self.running_cost_u, 0)

        self.f_x_and_u = lambda x, u: torch.autograd.functional.jacobian(
            self.f, inputs=(x, u)
        )
        self.f_xx = lambda x, u: torch.func.hessian(self.f, 0)(
            x, u
        )  # torch.autograd.functional.jacobian(self.f_x, 0)
        self.f_uu = lambda x, u: torch.func.hessian(self.f, 1)(
            x, u
        )  # torch.autograd.functional.jacobian(self.f_u, 1)
        self.f_ux = lambda x, u: torch.func.jacfwd(torch.func.jacfwd(self.f, 1), 0)(
            x, u
        )  # torch.autograd.functional.jacobian(self.f_u, 0)

        self.x_seq = None
        self.u_seq = None

    def plan(self, s_0):
        if self.x_seq is None:
            x_seq = [torch.from_numpy(s_0.copy()).float().requires_grad_(True)]
            u_seq = [
                torch.zeros(self.u_n).float().requires_grad_(True)
                for _ in range(self.predict_window)
            ]
            x_seq = self.initial_forward_pass(x_seq, u_seq)
        else:
            x_seq = self.x_seq
            u_seq = self.u_seq
            x_seq[0] = s_0.copy()

        n_iter = 0
        while n_iter < self.max_iter:
            k_seq, kk_seq = self.backward(x_seq, u_seq)
            x_seq, u_seq = self.forward(x_seq, u_seq, k_seq, kk_seq)
            n_iter += 1

        self.x_seq = x_seq
        self.u_seq = u_seq

        return u_seq[0].detach().numpy().astype(np.float32)

    def initial_forward_pass(self, x_seq, u_seq):
        for t in range(self.predict_window):
            x_seq.append(self.next_state(x_seq[-1], u_seq[t]))
        return x_seq

    def next_state(self, x, u):
        x_p = self.f(x, u)
        # x_p = torch.clip(x_p, self.env.observation_space.low, self.env.observation_space.high)
        return x_p

    def backward(self, x_seq, u_seq):
        self.v[-1] = self.final_cost(x_seq[-1])
        self.v_x[-1] = self.final_cost_x(x_seq[-1])
        self.v_xx[-1] = self.final_cost_xx(x_seq[-1])
        k_seq = []
        kk_seq = []
        for t in range(self.predict_window - 1, -1, -1):
            f_x_t, f_u_t = self.f_x_and_u(x_seq[t], u_seq[t])
            q_x = self.running_cost_x(x_seq[t], u_seq[t]) + torch.matmul(
                f_x_t.T, self.v_x[t + 1]
            )
            q_u = self.running_cost_u(x_seq[t], u_seq[t]) + torch.matmul(
                f_u_t.T, self.v_x[t + 1]
            )
            try:
                q_xx = (
                    self.running_cost_xx(x_seq[t], u_seq[t])
                    + torch.matmul(torch.matmul(f_x_t.T, self.v_xx[t + 1]), f_x_t)
                    + torch.dot(
                        self.v_x[t + 1], torch.squeeze(self.f_xx(x_seq[t], u_seq[t]))
                    )
                )
            except:
                q_xx = (
                    self.running_cost_xx(x_seq[t], u_seq[t])
                    + torch.matmul(torch.matmul(f_x_t.T, self.v_xx[t + 1]), f_x_t)
                    + (
                        self.v_x[t + 1] * torch.squeeze(self.f_xx(x_seq[t], u_seq[t]))
                    ).sum(-1)
                )
            tmp = torch.matmul(f_u_t.T, self.v_xx[t + 1])
            q_uu = (
                self.running_cost_uu(x_seq[t], u_seq[t])
                + torch.matmul(tmp, f_u_t)
                + torch.dot(
                    self.v_x[t + 1], torch.squeeze(self.f_uu(x_seq[t], u_seq[t]))
                )
            )
            try:
                q_ux = (
                    self.running_cost_ux(x_seq[t], u_seq[t])
                    + torch.matmul(tmp, f_x_t)
                    + torch.dot(
                        self.v_x[t + 1], torch.squeeze(self.f_ux(x_seq[t], u_seq[t]))
                    )
                )
            except:
                q_ux = (
                    self.running_cost_ux(x_seq[t], u_seq[t])
                    + torch.matmul(tmp, f_x_t)
                    + (
                        self.v_x[t + 1] * torch.squeeze(self.f_ux(x_seq[t], u_seq[t]))
                    ).sum(-1)
                )
                q_ux = q_ux.float()
            inv_q_uu = torch.linalg.inv(q_uu)
            k = -torch.matmul(inv_q_uu, q_u)
            kk = -torch.matmul(inv_q_uu, q_ux)
            dv = 0.5 * torch.matmul(q_u, k)
            self.v[t] += dv
            self.v_x[t] = q_x - torch.matmul(torch.matmul(q_u, inv_q_uu), q_ux)
            self.v_xx[t] = q_xx + torch.matmul(q_ux.T, kk)
            k_seq.append(k)
            kk_seq.append(kk)
        k_seq.reverse()
        kk_seq.reverse()
        return k_seq, kk_seq

    def forward(self, x_seq, u_seq, k_seq, kk_seq):
        x_seq_hat = x_seq.copy()
        u_seq_hat = u_seq.copy()
        for t in range(len(u_seq)):
            control = k_seq[t] + torch.matmul(kk_seq[t], (x_seq_hat[t] - x_seq[t]))
            u_p = u_seq[t] + control
            u_seq_hat[t] = torch.clamp(
                u_p,
                torch.from_numpy(self.env.action_space.low),
                torch.from_numpy(self.env.action_space.high),
            )
            x_seq_hat[t + 1] = self.next_state(x_seq_hat[t], u_seq_hat[t])
        return x_seq_hat, u_seq_hat
