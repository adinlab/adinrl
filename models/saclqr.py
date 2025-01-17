import torch

from models.sequential.sequential_sac import SequentialSoftActorCritic
from models.lqr import LinearQuadraticRegulator


class SoftActorCriticWithLinearQuadraticRegulator(SequentialSoftActorCritic):
    _agent_name = "SACLQR"

    def __init__(self, env, args, actor_nn, critic_nn):
        super().__init__(env, args, actor_nn, critic_nn)
        lqr_args = args
        lqr_args.batch_size = 128
        self.lqr = LinearQuadraticRegulator(env, lqr_args)
        self.step_counter = 0

    def learn(self, max_iter=1):
        if self.step_counter % self.env.max_episode_steps == 0:
            self.lqr.experience_memory.clone(self.experience_memory)
            self.lqr.learn(max_iter)
        else:
            super().learn(max_iter)

    @torch.no_grad()
    def select_action(self, s):
        self.step_counter += 1
        if (
            (self.env.max_episode_steps * 4)
            < self.step_counter
            <= (self.env.max_episode_steps * 5)
        ):
            if self.step_counter == self.env.max_episode_steps * 4 + 1:
                self.logger.log(f"LQR is testing")
            elif self.step_counter == self.env.max_episode_steps * 5:
                self.step_counter = 0

            return self.lqr.select_action(s)
        else:
            s = torch.from_numpy(s).unsqueeze(0).float().to(self.device)
            a, _ = self._actor(s)
            a = a.cpu().numpy().squeeze(0)
            return a
