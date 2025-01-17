import torch

from models.ddpg import DeepDeterministicPolicyGradient
from models.banditQL.utils import get_bandit_criric
from models.banditQL.Qminimizer import Qminimizer


class BanditQLearning(DeepDeterministicPolicyGradient):
    def __init__(self, env, args):
        super().__init__(env, args, None, get_bandit_criric(args.model))
        if "gamma" in args.model:
            self.critic.head.gamma = args.bofucb_gamma
            self.critic_target.head.gamma = args.bofucb_gamma

        self.qminsolver = Qminimizer(env, self.critic_target, solvertype="gd")
        self.t = 0
        self.log_mus = []
        self.log_stds = []
        self.log_betas = []

    def learn(self, max_iter=1):
        s, a, r, sp, done, step = self.experience_memory.get_last_observation()

        ap = self.qminsolver.solve(sp, t=step, t_max=self.experience_memory.buffer_size)
        qp, _ = self.critic_target(sp, ap, t=step)
        q_target = r.view(-1, 1) + self._gamma * qp.detach()

        self.critic.update_belief(s, a, q_target.detach(), t=step)
        self.hard_update(self.critic, self.critic_target)

        if self.args.batch_size > len(self.experience_memory):
            return None

        for ii in range(max_iter):
            s, a, r, sp, done, step = self.experience_memory.sample_random(
                self.args.batch_size
            )

            ap = self.qminsolver.solve(
                sp, t=step, t_max=self.experience_memory.buffer_size
            )
            qp, _ = self.critic_target(sp, ap, t=step)
            q_target = r.view(-1, 1) + self._gamma * qp.detach()

            self.critic_optim.zero_grad()
            td_error = self.critic.neg_log_prob(s, a, q_target.detach(), t=step)
            td_error.backward()
            self.critic_optim.step()
            self.hard_update(self.critic, self.critic_target)

    def select_action(self, s):
        self.t += 1
        s = torch.from_numpy(s).unsqueeze(0).float().to(self.device)
        a = self.qminsolver.solve(s, t=self.t, t_max=self.experience_memory.buffer_size)
        a = a.cpu().detach().clone().numpy().squeeze(0)
        self.log_mus.append(self.qminsolver.model.head.log_mu)
        self.log_stds.append(self.qminsolver.model.head.log_std)
        self.log_betas.append(self.qminsolver.model.head.log_beta)
        return a

    def __del__(self):
        # save logs
        import numpy as np

        np.save(f"{self.logger.path}/log_mus.npy", np.array(self.log_mus))
        np.save(f"{self.logger.path}/log_stds.npy", np.array(self.log_stds))
        np.save(f"{self.logger.path}/log_betas.npy", np.array(self.log_betas))
