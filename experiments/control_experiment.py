import time
import numpy as np
import torch

from experiments.base_experiment import Experiment


class ControlExperiment(Experiment):
    def __init__(self, args):
        super(ControlExperiment, self).__init__(args)

    def train(self):
        time_start = time.time()

        information_dict = {
            "episode_rewards": np.zeros(10000),
            "episode_steps": np.zeros(10000),
            "step_rewards": np.empty((2 * self.args.max_steps), dtype=object),
        }

        s, _ = self.env.reset()
        r_cum = 0
        episode = 0
        e_step = 0

        for step in range(self.args.max_steps):

            e_step += 1

            if step % self.args.eval_frequency == 0:
                self.eval(step)

            if step < self.args.warmup_steps:
                a = self.env.action_space.sample()
            else:
                a = self.agent.select_action(s)

            a = np.clip(a, -1.0, 1.0)
            sp, r, done, truncated, info = self.env.step(a)

            self.agent.store_transition(s, a, r, sp, done, step + 1)

            information_dict["step_rewards"][self.n_total_steps + step] = (
                episode,
                step,
                r,
            )

            s = sp  # Update state
            r_cum += r  # Update cumulative reward

            if (
                step >= self.args.warmup_steps
                and (step % self.args.learn_frequency) == 0
            ):
                # print("Learning at step: ", step)
                self.agent.learn(max_iter=self.args.max_iter)

            if done or truncated:

                information_dict["episode_rewards"][episode] = r_cum
                information_dict["episode_steps"][episode] = step
                self.agent.logger.episode_summary(episode, step, information_dict)
                s, _ = self.env.reset()
                r_cum = 0
                episode += 1
                e_step = 0

                # ddpg reset noise
                if self.args.model == "ddpg":
                    self.agent.reset()
                    
                if step % self.args.reporter_save_frequency == 0:
                    self.agent.logger.save(self.agent, information_dict, episode, step)

        self.eval(step)
        time_end = time.time()
        self.agent.logger.save(self.agent, information_dict, episode, step)
        self.agent.logger.log(f"Training time: {time_end - time_start:.2f} seconds")

    @torch.no_grad()
    def eval(self, n_step):
        self.agent.eval()
        results = np.zeros(self.args.eval_episodes)
        q_values = np.zeros((self.args.eval_episodes, 2))
        avg_reward = np.zeros(self.args.eval_episodes)

        for episode in range(self.args.eval_episodes):
            s, _ = self.eval_env.reset()
            step = 0
            a = self.agent.select_action(s, is_training=False)
            done = False

            while not done:
                a = self.agent.select_action(s, is_training=False)
                sp, r, term, trunc, info = self.eval_env.step(a)
                done = term or trunc
                s = sp
                results[episode] += r
                avg_reward[episode] += self.args.gamma**step * r
                step += 1

        self.agent.logger.save_eval_results(n_step, results)
        self.agent.train()
