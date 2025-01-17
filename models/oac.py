import torch
from torch import nn
import math
from torch.distributions import (
    Normal,
    Categorical,
    TransformedDistribution,
    TanhTransform,
)
from torch.distributions.transforms import TanhTransform

from agents.base_agent import Agent


class OptimisticActorCritic(Agent):
    def __init__(self, env, args, actor_nn, critic_nn):
        super().__init__(env, args)
        self._env = env
        self._sigma = math.sqrt(0.1)
        self._sigma_target = math.sqrt(0.2)
        self._c = 0.5
        self._batch_size = args.batch_size
        self._critic_loss_fcn = nn.MSELoss()

        self._beta_ub = 4.66
        self._beta_lb = -3.65
        #
        self._actor = actor_nn(self.dim_obs, self.dim_act, args.n_hidden).to(
            self.device
        )
        self._actor_target = actor_nn(self.dim_obs, self.dim_act, args.n_hidden).to(
            self.device
        )
        self._critic_1 = critic_nn(self.dim_obs, self.dim_act, args.n_hidden).to(
            self.device
        )
        self._critic_2 = critic_nn(self.dim_obs, self.dim_act, args.n_hidden).to(
            self.device
        )
        self._critic_1_target = critic_nn(self.dim_obs, self.dim_act, args.n_hidden).to(
            self.device
        )
        self._critic_2_target = critic_nn(self.dim_obs, self.dim_act, args.n_hidden).to(
            self.device
        )

        self._hard_update(
            self._actor, self._actor_target
        )  # hard update at the beginning
        self._hard_update(
            self._critic_1, self._critic_1_target
        )  # hard update at the beginning
        self._hard_update(
            self._critic_2, self._critic_2_target
        )  # hard update at the beginning
        #
        self._actor_optim = torch.optim.Adam(
            self._actor.parameters(), args.learning_rate
        )
        self._critic_1_optim = torch.optim.Adam(
            self._critic_1.parameters(), args.learning_rate
        )
        self._critic_2_optim = torch.optim.Adam(
            self._critic_2.parameters(), args.learning_rate
        )

    def learn(self, max_iter=1):
        # pass
        if self.args.batch_size > len(self.experience_memory):
            return None
        for ii in range(max_iter):
            s, a, r, sp, done, step = self.experience_memory.sample_random(
                self.args.batch_size
            )
            # generate q targets
            with torch.no_grad():
                up_pred, _, _ = self._actor_target(sp)
                up_pred += (
                    torch.distributions.normal.Normal(0, self._sigma_target)
                    .sample(sample_shape=up_pred.shape)
                    .clamp(-self._c, self._c)
                )
                qp_1_target, _, _ = self._critic_1_target(sp, up_pred)
                qp_2_target, _, _ = self._critic_2_target(sp, up_pred)
                qp_target = torch.min(qp_1_target, qp_2_target)
                q_target = r.unsqueeze(-1) + (
                    self._gamma * qp_target * (1 - done.unsqueeze(-1))
                )
            # update critic 1
            self._critic_1_optim.zero_grad()
            q_1, _, _ = self._critic_1(s, a)
            q_1_loss = self._critic_loss_fcn(q_1, q_target)
            q_1_loss.backward()
            self._critic_1_optim.step()
            # update critic 2
            self._critic_2_optim.zero_grad()
            q_2, _, _ = self._critic_2(s, a)
            q_2_loss = self._critic_loss_fcn(q_2, q_target)
            q_2_loss.backward()
            self._critic_2_optim.step()
            # update actor
            self._actor_optim.zero_grad()
            a_pred, _, _ = self._actor(s)

            c1, _, _ = self._critic_1(s, a_pred)
            c2, _, _ = self._critic_2(s, a_pred)
            q_pi = torch.min(c1, c2)

            pi_loss = -q_pi.mean()
            pi_loss.backward()
            self._actor_optim.step()

            # soft update of target critic networks
            self._soft_update(self._actor, self._actor_target)
            self._soft_update(self._critic_1, self._critic_1_target)
            self._soft_update(self._critic_2, self._critic_2_target)

    # @torch.no_grad()
    def select_action(self, s, is_training=True):

        delta = 0.1
        s = torch.from_numpy(s).unsqueeze(0).float().to(self.device)
        # mu_sigma, action = self._actor(s)
        # mu_T, sigma_T = mu_sigma[0],mu_sigma[1]

        action, _, transformed_dist = self._actor(s)
        pre_tanh_mu_T, std_T = (
            transformed_dist.base_dist.loc,
            transformed_dist.base_dist.scale,
        )
        pre_tanh_mu_T.requires_grad_()
        tanh_mu_t = torch.tanh(pre_tanh_mu_T)

        if is_training:
            Q1, _, _ = self._critic_1(s, tanh_mu_t)
            Q2, _, _ = self._critic_2(s, tanh_mu_t)

            mu_Q = (Q1 + Q2) / 2.0
            sigma_Q = torch.abs(Q1 - Q2) / 2.0

            Q_UB = mu_Q + self._beta_ub * sigma_Q
            grad = torch.autograd.grad(Q_UB, pre_tanh_mu_T)
            grad = grad[0]

            sigma_T = std_T.square()

            denom = (
                torch.sqrt(torch.sum(torch.mul(torch.pow(grad, 2), sigma_T))) + 10e-6
            )

            mu_C = math.sqrt(2.0 * delta) * torch.mul(sigma_T, grad) / denom
            mu_E = pre_tanh_mu_T + mu_C

            dist_bt = Normal(mu_E, std_T, validate_args=False)
            transform = TanhTransform(cache_size=1)
            dist = TransformedDistribution(dist_bt, transform)

            a = dist.sample()
        else:
            a = tanh_mu_t

        a = a.detach().numpy().squeeze(0)
        self._critic_1_optim.zero_grad()
        self._critic_2_optim.zero_grad()
        self._actor_optim.zero_grad()

        return a
