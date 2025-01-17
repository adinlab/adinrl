import time

import numpy as np

from experiments.control_experiment import ControlExperiment


class SingleEpisodeControlExperiment(ControlExperiment):
    def __init__(self, args):
        super().__init__(args)
        self.agent.env = self.env.env

    def train(self):
        time_start = time.time()

        information_dict = {
            "episode_rewards": np.zeros(2 * self.args.max_steps),
            "episode_steps": np.zeros(2 * self.args.max_steps),
            "step_rewards": np.empty((2 * self.args.max_steps), dtype=object),
        }
        self.env.env.seed(self.args.repetition)
        s = self.env.reset()
        self.agent.logger.log(f"Starting state {s}")
        r_cum = 0  # Cumulative episode reward
        step = 0  # Episode step counter
        episode = 0

        while True:
            if self.args.render and step % self.args.render_frequency == 0:
                time.sleep(0.01)
                self.env.render()

            if self.n_total_steps + step < self.args.warmup_steps:
                a = self.env.action_space.sample()
            else:
                a = self.agent.select_action(s)
            sp, r, terminated, info = self.env.step(a)
            self.agent.store_transition(
                s, a, r, sp, terminated, self.n_total_steps + step + 1
            )

            information_dict["step_rewards"][step] = (
                episode,
                step,
                r,
            )  # Store a tuple of (Episode, Step, Reward)
            information_dict["episode_rewards"][step] = r
            information_dict["episode_steps"][step] = step
            s = sp  # Update state
            r_cum += r  # Update cumulative reward
            step += 1  # Update step counter

            if step % self.args.report_frequency == 0:
                self.agent.logger.step_summary(step, r, r_cum)

            if (
                step % self.args.learn_frequency == 0
                and self.n_total_steps + step >= self.args.warmup_steps
            ):  # first fill in the replay buffer
                self.agent.learn(max_iter=self.args.max_iter)

            if step % self.args.reporter_save_frequency == 0:
                self.agent.logger.save(self.agent, information_dict, step - 1, step - 1)

            if self.n_total_steps + step >= self.args.max_steps:  # episode end
                self.n_total_steps += step
                self.agent.logger.save(self.agent, information_dict, step - 1, step - 1)
                break

        time_end = time.time()
        self.agent.logger.log(f"Training time: {time_end - time_start:.2f} seconds")
