import torch
from torch import nn
import numpy as np

from agents.base_agent import Agent


class TwinDelayedDeepDeterministicPolicyGradient(Agent):
    def __init__(self, env, args, actor_nn, critic_nn):
        super(TwinDelayedDeepDeterministicPolicyGradient, self).__init__(env, args)
        self._alpha = 0.05  # 0.2
        self._sigma = np.sqrt(0.1)
        self._sigma_target = np.sqrt(0.2)
        self._c = 0.5
        self._d = 2
        self._critic_loss_fcn = nn.MSELoss()
        #
        self._actor = actor_nn(self._nx, self._nu, args.n_hidden).to(self.device)
        self._actor_target = actor_nn(self._nx, self._nu, args.n_hidden).to(self.device)
        self._critic_1 = critic_nn(self._nx, self._nu, args.n_hidden).to(self.device)
        self._critic_2 = critic_nn(self._nx, self._nu, args.n_hidden).to(self.device)
        self._critic_1_target = critic_nn(self._nx, self._nu, args.n_hidden).to(
            self.device
        )
        self._critic_2_target = critic_nn(self._nx, self._nu, args.n_hidden).to(
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

        self._n_iter = 0

    def learn(self, max_iter=1):
        # pass
        if self.args.batch_size > len(self.experience_memory):
            return None

        for ii in range(max_iter):
            self._n_iter += 1
            s, a, r, sp, done, step = self.experience_memory.sample_random(
                self.args.batch_size
            )
            # generate q targets
            with torch.no_grad():
                up_pred, _ = self._actor_target(sp)
                noise = (torch.randn_like(up_pred) * self._sigma_target).clamp(
                    -self._c, self._c
                )
                up_pred += noise
                qp_1_target = self._critic_1_target(sp, up_pred)
                qp_2_target = self._critic_2_target(sp, up_pred)
                qp_target = torch.min(qp_1_target, qp_2_target)
                q_target = r.unsqueeze(-1) + (
                    self._gamma * qp_target * (1 - done.unsqueeze(-1))
                )
            # update critic 1
            self._critic_1_optim.zero_grad()
            q_1 = self._critic_1(s, a)
            q_1_loss = self._critic_loss_fcn(q_1, q_target)
            q_1_loss.backward()
            self._critic_1_optim.step()
            # update critic 2
            self._critic_2_optim.zero_grad()
            q_2 = self._critic_2(s, a)
            q_2_loss = self._critic_loss_fcn(q_2, q_target)
            q_2_loss.backward()
            self._critic_2_optim.step()
            if self._n_iter % self._d == 0:
                # update actor
                self._actor_optim.zero_grad()
                a_pred, _ = self._actor(s)
                # ---
                q_1 = self._critic_1(s, a_pred)
                pi_loss = -q_1.mean()
                pi_loss.backward()
                self._actor_optim.step()
                # soft update of target critic networks
                self._soft_update(self._actor, self._actor_target)
                self._soft_update(self._critic_1, self._critic_1_target)
                self._soft_update(self._critic_2, self._critic_2_target)

    @torch.no_grad()
    def select_action(self, s, is_training=True):
        s = torch.from_numpy(s).unsqueeze(0).float().to(self.device)
        a, _ = self._actor(s)
        if is_training:
            noise = (torch.randn_like(a) * self._sigma).clamp(-self._c, self._c)
            a += noise
        a = a.clamp(-1.000, 1.000).cpu().numpy().squeeze(0)
        return a
    
