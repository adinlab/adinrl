import torch
from torch import nn
import math
from agents.base_agent import Agent
import numpy as np
import torch.nn.init as init
from scipy.stats import norm


class RiskSensitiveSAC(Agent):
    _agent_name = "RSAC"

    def __init__(self, env, args, actor_nn, critic_nn):
        super().__init__(env, args)

        self._alpha = args.alpha
        self._lamda = args.lamda
        self._loss_type = args.loss_type
        self._n_ensemble = args.n_critics
        self._delta = args.delta
        self._reduce = args.reduce
        self._eps_loss = args.eps_loss
        self._scheduled_LR = args.scheduled_LR
        self._prior_type = args.prior_type
        self._critic_var_upper_clamp = args.critic_var_upper_clamp

        self._step = 0
        self._max_steps = args.max_steps
        self._c = 0

        self.critic = []
        self.critic_t = []
        self.critic_optim = []
        self.critic_scheduler = []

        for i in range(self._n_ensemble):
            self.critic.append(
                critic_nn(self.dim_obs, self.dim_act, args.n_hidden).to(self.device)
            )
            self.critic_t.append(
                critic_nn(self.dim_obs, self.dim_act, args.n_hidden).to(self.device)
            )
            optim_i = torch.optim.Adam(self.critic[i].parameters(), args.learning_rate)
            self.critic_optim.append(optim_i)
            self.critic_scheduler.append(
                torch.optim.lr_scheduler.LinearLR(
                    optimizer=optim_i, start_factor=1, end_factor=0.1, total_iters=1000
                )
            )
            self._hard_update(
                self.critic[i], self.critic_t[i]
            )  # hard update at the beginning

        self._actor = actor_nn(
            self.dim_obs, self.dim_act, args.n_hidden, args.actor_var_upper_clamp
        ).to(self.device)
        self._actor_optim = torch.optim.Adam(
            self._actor.parameters(), args.learning_rate
        )
        self._actor_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=self._actor_optim,
            start_factor=1,
            end_factor=0.1,
            total_iters=1000,
        )

    def compute_Q_target(self, s, a, r, sp, done):
        # generate q targets
        with torch.no_grad():
            ap_pred, ep_pred = self._actor(sp, is_training=True)
            qp_t_mu_list = []
            qp_t_logvar_list = []
            for j in range(self._n_ensemble):
                qp_i_t = self.critic_t[j](sp, ap_pred)
                qp_i_t_mu = qp_i_t[:, 0].view(-1, 1)
                qp_i_t_logvar = qp_i_t[:, 1].view(-1, 1)
                qp_t_mu_list.append(qp_i_t_mu)
                qp_t_logvar_list.append(qp_i_t_logvar)

            # convert list to tensor
            qp_t_mu_list = torch.cat(qp_t_mu_list, dim=-1)
            qp_t_logvar_list = torch.cat(qp_t_logvar_list, dim=-1)

            idx = qp_t_mu_list.argmin(dim=-1, keepdim=True)
            qp_t_logvar = qp_t_logvar_list.gather(1, idx)

            q_t_logvar = 2.0 * math.log(self._gamma) + qp_t_logvar
            qp_t_mu = qp_t_mu_list.gather(1, idx)

            q_t_mu = r.unsqueeze(-1) + (
                self._gamma * qp_t_mu * (1 - done.unsqueeze(-1))
            )
            qp_t_var = qp_t_logvar.clamp(-10, self._critic_var_upper_clamp).exp()

            # return q_t_mu, q_t_logvar
            return qp_t_mu, qp_t_var, ep_pred

    def Q_eval(self, s, a, critic_list):
        q_pi_mu_list = []
        q_pi_logvar_list = []
        for i in range(self._n_ensemble):
            q_pi_params = critic_list[i](s, a)
            q_pi_mu = q_pi_params[:, 0].view(-1, 1)
            q_pi_logvar = q_pi_params[:, 1].view(-1, 1)
            q_pi_mu_list.append(q_pi_mu)
            q_pi_logvar_list.append(q_pi_logvar)

        q_pi_mu_list = torch.cat(q_pi_mu_list, dim=-1)
        q_pi_logvar_list = torch.cat(q_pi_logvar_list, dim=-1)

        if self._reduce == "min":
            idx = q_pi_mu_list.argmin(dim=-1, keepdim=True)
            q_pi_logvar = q_pi_logvar_list.gather(1, idx)
            q_pi_mu = q_pi_mu_list.gather(1, idx)

        elif self._reduce == "mean":
            q_pi_logvar = q_pi_logvar_list.mean(
                dim=-1, keepdim=True
            )  # / self._n_ensemble
            q_pi_mu = q_pi_mu_list.mean(dim=-1, keepdim=True)

        else:
            raise NotImplementedError

        return q_pi_mu, q_pi_logvar.clamp(-10, self._critic_var_upper_clamp).exp()

    def learn(self, max_iter=1):
        self._step += 1

        if self.args.batch_size > len(self.experience_memory):
            return None

        for iteration in range(max_iter):
            s, a, r, sp, done, step = self.experience_memory.sample_random(
                self.args.batch_size
            )
            q_t_mu, q_t_var, ep_pred = self.compute_Q_target(s, a, r, sp, done)

            # update critic ensemble
            for i in range(self._n_ensemble):
                self.critic_optim[i].zero_grad()
                q = self.critic[i](s, a)
                q_mu = q[:, 0].view(-1, 1)
                q_logvar = q[:, 1].view(-1, 1)
                q_var = q_logvar.clamp(-10, self._critic_var_upper_clamp).exp()

                if self._loss_type == "risk":
                    target = r.unsqueeze(-1) + (
                        self._gamma
                        * (
                            q_t_mu
                            + (self._lamda * 0.5 * q_t_var)
                            - (self._alpha * ep_pred)
                        )
                        * (1 - done.unsqueeze(-1))
                    )
                    q = q_mu + 0.5 * self._lamda * q_var
                    sq_e = (target - q) ** 2
                    q_loss = sq_e.mean()

                else:
                    raise NotImplementedError

                q_loss.backward()
                self.critic_optim[i].step()

            # update actor
            self._actor_optim.zero_grad()
            a_pred, e_pred = self._actor(s, is_training=True)

            q_pi_mu, q_pi_var = self.Q_eval(s, a_pred, self.critic)
            pi_loss = (
                self._alpha * e_pred - (q_pi_mu + 0.5 * self._lamda * q_pi_var)
            ).mean()
            pi_loss.backward()
            self._actor_optim.step()

            for i in range(self._n_ensemble):
                self._soft_update(self.critic[i], self.critic_t[i])

    @torch.no_grad()
    def select_action(self, s, is_training=True):
        s = torch.from_numpy(s).view(1, -1).float().to(self.device)
        a, _ = self._actor(s, is_training=is_training)
        a = a.cpu().numpy().squeeze(0)
        return a

    @torch.no_grad()
    def Q_value(self, s, a):
        s = torch.from_numpy(s).view(1, -1).float().to(self.device)
        a = torch.from_numpy(a).view(1, -1).float().to(self.device)
        q = self.critic[0](s, a)
        q_mu = q[:, 0].view(-1, 1)
        q_logvar = q[:, 1].view(-1, 1)
        q_var = q_logvar.clamp(-10, self._critic_var_upper_clamp).exp()
        return q_mu.item(), q_var.item()
