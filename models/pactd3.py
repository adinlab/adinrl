import torch
from torch import nn
import numpy as np
from loggers.logger import *
from agents.base_agent import Agent
from nets.layers.bayesian_layers import calculate_kl_terms
import math
import torch.nn.functional as F


class PACBayesianTD3(Agent):
    def __init__(self, env, args, actor_nn, critic_nn, num_critics=2):
        super().__init__(env, args)

        self._n_iter = 0
        self.num_critics = num_critics

        self._actor_update_frequency = 5

        self._alpha = 0.05  # entropy coefficient

        # self._actor_E = actor_nn(self._nx, self._nu, args.n_hidden).to(self.device)
        # self._actor_E_optim = torch.optim.Adam(self._actor_E.parameters(), args.learning_rate)

        self.actor = actor_nn(self.dim_obs, self.dim_act, args.n_hidden).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), args.learning_rate)

        self._critic = []
        self._critic_optim = []
        for i in range(self.num_critics):
            self._critic.append(
                critic_nn(self.dim_obs, self.dim_act, args.n_hidden).to(self.device)
            )
            self._critic_optim.append(
                torch.optim.Adam(self._critic[i].parameters(), args.learning_rate)
            )

    ##############################################################################################################
    def neg_log_normal(self, x, mu, var):
        return 0.5 * (
            (x - mu) ** 2
        )  # / var + 0.5 * torch.log(var) + 0.5 * math.log(2 * math.pi)

    ##############################################################################################################
    def learn_critic(self):

        for i in range(self.num_critics):
            s, a, r, sp, done, step_ = self.experience_memory.sample_random(
                self.args.batch_size
            )

            self._critic_optim[i].zero_grad()

            # compute targets
            with torch.no_grad():
                ap, _ = self.actor(sp)
                qnext_mu = 0
                qnext_var = 0

                for j in range(self.num_critics):
                    if j != i:
                        q_j_dist = self._critic[j](sp, ap)
                        qnext_mu += q_j_dist[:, 0] / (self.num_critics - 1)
                        qnext_var += q_j_dist[:, 1].clamp(-4, 4).exp() / (
                            self.num_critics - 1
                        )

                q_target_mu = r.view(-1, 1) + (
                    self._gamma * qnext_mu * (1 - done.view(-1, 1))
                )

            qdist = self._critic[i](s, a)

            q_mu = qdist[:, 0].view(-1)
            q_var = qdist[:, 1].view(-1).clamp(-4, 4).exp()

            loss_critic_i = self.neg_log_normal(
                q_target_mu, q_mu, (q_var + qnext_var)
            ).mean()

            loss_critic_i.backward()

            self._critic_optim[i].step()

    ##############################################################################################################
    def learn_actor(self):
        s, a, r, sp, done, step_ = self.experience_memory.sample_random(
            self.args.batch_size
        )

        # learn target policy
        self.actor_optim.zero_grad()
        a, pi_dist = self.actor(s)
        e_pred = pi_dist.log_prob(a).sum(dim=-1, keepdim=True)
        q = 0
        # for i in range(self.num_critics):
        qdist = self._critic[0](s, a)
        # q += (qdist[:,0] + qdist[:,1].clamp(-4,4).exp().sqrt() * torch.randn_like(qdist[:,1])) #/ self.num_critics
        q += qdist[
            :, 0
        ]  # / self.num_critics # + qdist[:,1].clamp(-4,4).exp().sqrt() * torch.randn_like(qdist[:,1]))/ self.num_critics

        pi_loss_t = (-q.view(-1) + e_pred).mean()

        pi_loss_t.backward()

        # print("pi_loss_t: ", pi_loss_t.item(),  q.view(-1).mean().item(), e_pred.mean().item())

        self.actor_optim.step()

    ##############################################################################################################
    def learn(self, max_iter=1):
        if self.args.batch_size > len(self.experience_memory):
            return None

        for ii in range(max_iter):
            self.learn_critic()

        self._n_iter += 1

        if self._n_iter % self._actor_update_frequency == 0:
            for jj in range(self._actor_update_frequency * max_iter):
                self.learn_actor()

    ##############################################################################################################
    @torch.no_grad()
    def select_action(self, s, is_training=True):
        s = torch.from_numpy(s).unsqueeze(0).float().to(self.device)

        # if is_training:
        #    a, _ = self._actor_E(s)
        # else:

        a, _ = self.actor(s)

        # if a.max() > 1 or a.min() < -1:
        #    print("Action out of bounds")

        a = a.cpu().numpy().squeeze(0)
        return a
