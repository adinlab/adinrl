import time

import numpy as np
import torch
import torch as th
from tqdm import tqdm

from experiments.base_experiment import Experiment
from utils.utils import totorch, tonumpy


class ControlExperiment(Experiment):
    def __init__(self, args):
        super().__init__(args)

    def train(self):
        time_start = time.time()

        information_dict = {
            "episode_rewards": th.zeros(1000000),
            "episode_steps": th.zeros(1000000),
            "step_rewards": np.empty((2 * self.args.max_steps), dtype=object),
        }

        s, _ = self.env.reset()
        s = totorch(s, device=self.args.device)
        r_cum = np.zeros(1)
        episode = 0
        e_step = 0

        for step in tqdm(
            range(self.args.max_steps), leave=True, disable=not self.args.progress
        ):
            e_step += 1

            if (
                step > self.args.warmup_steps
                and self.args.reset_frequency > 0
                and step % self.args.reset_frequency == 0
            ):
                self.agent.critics.reset()

            if step % self.args.eval_frequency == 0:
                self.eval(step)

            if step < self.args.warmup_steps:
                a = self.env.action_space.sample()
                a = totorch(np.clip(a, -1.0, 1.0), device=self.args.device)

            else:
                a = self.agent.select_action(s).clip(-1.0, 1.0)

            sp, r, done, truncated, info = self.env.step(tonumpy(a))
            sp = totorch(sp, device=self.args.device)

            if self.args.verbose and "sp" in self.args.env:
                print("X pos: ", info["x_pos"], "Action norm: ", info["action_norm"])
                # TODO: Write this instead into a file!

            self.agent.store_transition(s, a, r, sp, done, truncated, step + 1)

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
                self.agent.learn(
                    max_iter=self.args.max_iter, n_epochs=self.args.n_epochs
                )
            ##################
            if step == self.args.max_steps - 1:
                if self.args.saveparams:
                    self.agent.critics.save_params(
                        f"_logs/{self.args.env}/{self.args.model}/seed_0{self.args.seed}/params.pth"
                    )
                    self.agent.actor.save_actor_params(
                        f"_logs/{self.args.env}/{self.args.model}/seed_0{self.args.seed}/Actor_params.pth"
                    )
            ##################
            if done or truncated:

                information_dict["episode_rewards"][episode] = r_cum.item()
                information_dict["episode_steps"][episode] = step
                self.agent.logger.episode_summary(episode, step, information_dict)
                s, _ = self.env.reset()
                s = totorch(s, device=self.args.device)
                r_cum = np.zeros(1)
                episode += 1
                e_step = 0

                if step % self.args.reporter_save_frequency == 0:
                    self.agent.logger.save(self.agent, information_dict, episode, step)

        self.eval(step)
        time_end = time.time()
        if self.args.tensorboard:
            self.writer.close()
        self.agent.logger.save(self.agent, information_dict, episode, step)
        self.agent.logger.log(f"Training time: {time_end - time_start:.2f} seconds")

    @torch.no_grad()
    def eval(self, n_step):
        self.agent.eval()
        results = th.zeros(self.args.eval_episodes)
        q_values = th.zeros((self.args.eval_episodes, 2))
        avg_reward = th.zeros(self.args.eval_episodes)
        collect_infos = {}
        performance_eval_dict = {
            "episode_info": np.empty((2 * self.args.max_steps), dtype=object),
            "trajectory": [],
        }
        if self.args.verbose:
            print("ASDOFASD")
            print(self.agent.actor.states)
            self.agent.logger.save_states(n_step, self.agent.actor.states)

        for episode in range(self.args.eval_episodes):
            collect_infos[episode] = []
            s, info = self.eval_env.reset()
            s = totorch(s, device=self.args.device)
            step = 0
            a = self.agent.select_action(s, is_training=False)
            q_values[episode] = self.agent.Q_value(s, a)
            done = False

            while not done:
                a = self.agent.select_action(s, is_training=False)

                sp, r, term, trunc, info = self.eval_env.step(tonumpy(a))
                collect_infos[episode].append(info)
                ###################
                if self.args.validationrounds:
                    performance_eval_dict["trajectory"].append(
                        (episode, step, s, a, sp, r, term, trunc, info)
                    )
                ###################
                done = term or trunc
                s = totorch(sp, device=self.args.device)
                results[episode] += r
                avg_reward[episode] += self.args.gamma**step * r
                step += 1

            if self.args.validationrounds:
                performance_eval_dict["episode_info"][episode] = (
                    episode,
                    avg_reward[episode],
                )  # episode, discounted_reward

        if self.args.tensorboard:
            self.writer.add_scalar("Cum reward", results.mean(), n_step)
            self.writer.add_scalar("Disc reward", avg_reward.mean(), n_step)
            self.writer.add_scalar("Q-pred", q_values.mean(), n_step)

        self.agent.actor.states = []

        self.agent.logger.save_info_dict(n_step, collect_infos)
        self.agent.logger.save_eval_results(n_step, results)
        self.agent.logger.save_overestimation_results(n_step, avg_reward, q_values)
        self.agent.logger.save_performance_info(n_step, performance_eval_dict)
        self.agent.train()
