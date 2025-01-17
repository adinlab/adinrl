import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch as th


class Logger:
    def __init__(self, args):
        self.path = (
            f"{args.result_path}/{args.env}/{args.model}/seed_{str(args.seed).zfill(2)}"
        )
        self.average_last_k = args.reporter_average_last_k
        self.logger = self.create_logger()
        self.logger.info(f"Experiment with seed no {args.seed}")
        self.logger.info(f"Args: \n{args}")
        self.eval_results = {}
        self.eval_dis_results = {}
        self.q_results = {}
        self.save_over_results = {}
        self.max_first_half = []
        self.max_steps = args.max_steps
        self.eval_frequency = args.eval_frequency
        self.maxval = 0

    def create_logger(self):
        os.makedirs(self.path, exist_ok=True)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            filename=f"{self.path}/log.log",
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%m-%d %H:%M-%S",
            level=logging.INFO,
            filemode="w",
        )

        logging.disable(logging.DEBUG)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        return logger

    def log(self, message):
        self.logger.info(message)
        # print(message)

    def critical(self, message):
        self.logger.critical(message)
        # print(message)

    def __call__(self, message: str):
        self.log(message)

    def episode_summary(self, episode, steps, info):
        reward = info["episode_rewards"][episode]
        message = (
            f"Episode: {episode + 1:4d}\tN-steps: {steps:7d}\tReward: {reward:10.3f}"
        )
        self.log(message)

    def step_summary(self, step, reward, cumulatuve_reward):
        message = f"N-steps: {step:7d}\tReward: {reward:10.3f}\tCumulative Reward: {cumulatuve_reward:10.3f}"
        self.log(message)

    def plot_rewards(self, rewards, steps):
        plt.plot(steps, rewards)
        plt.xlabel("Steps")
        plt.ylabel("Normalized Per-Episode Reward")
        plt.savefig(f"{self.path}/learning-curve.png")
        plt.close()

    def save(self, model, info, episode, n_step):
        if info:
            episode_rewards = info["episode_rewards"][: episode + 1]
            episode_steps = info["episode_steps"][: episode + 1]
            step_rewards = info["step_rewards"][: n_step + 1]
            np.save(f"{self.path}/episode_rewards.npy", episode_rewards)
            np.save(f"{self.path}/step_rewards.npy", step_rewards)

            self.plot_rewards(episode_rewards, episode_steps)

    def IQM_reward_calculator(self, rewards):
        sorted = np.sort(rewards)
        q1 = np.percentile(sorted, 25)
        q3 = np.percentile(sorted, 75)
        return np.mean(sorted[(sorted >= q1) & (sorted <= q3)])

    def save_info_dict(self, n_step, infos):
        th.save(infos, f"{self.path}/infos_{n_step}.pt")

    def save_performance_info(self, n_step, infos):
        th.save(infos, f"{self.path}/performance_infos_{n_step}.pt")

    def save_states(self, n_step, states):
        th.save(states, f"{self.path}/states_{n_step}.pt")

    def save_eval_results(self, n_step, rewards):
        iqm = self.IQM_reward_calculator(rewards)

        self.critical(
            f"EVALUATION\tN-steps: {n_step:7d}\tMen_Reward: {rewards.mean():10.3f}\tIQM_Reward: {iqm:10.3f}"
        )
        self.eval_results[n_step] = rewards

        np.save(f"{self.path}/eval_results.npy", self.eval_results)
        x = list(self.eval_results.keys())
        y_mean = np.array(
            list(map(lambda x: self.eval_results[x].mean(), self.eval_results))
        )
        y_std = np.array(
            list(map(lambda x: self.eval_results[x].std(), self.eval_results))
        )

        plt.plot(x, y_mean)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
        plt.xlabel("Steps")
        plt.ylabel("Eval Reward")
        plt.savefig(f"{self.path}/eval-curve.png")
        plt.close()

    def save_overestimation_results(self, n_step, rewards, q_value):
        self.critical(
            f"EVALUATION\tN-steps: {n_step:7d}\tReward: {rewards.mean():10.3f}\tQ-value: {q_value[:,0].mean():10.3f}"
        )
        self.eval_dis_results[n_step] = rewards
        self.q_results[n_step] = q_value[:, 0]
        self.save_over_results[n_step] = [rewards, q_value[:, 0], q_value[:, 1]]
        np.save(f"{self.path}/overestimation_results.npy", self.save_over_results)
        x = list(self.q_results.keys())
        a = list(self.eval_dis_results.keys())  # [::20]
        y_mean = np.array(list(map(lambda x: self.q_results[x].mean(), self.q_results)))
        y_std = np.array(list(map(lambda x: self.q_results[x].std(), self.q_results)))
        y_rewards = np.array(
            list(map(lambda a: self.eval_dis_results[a].mean(), self.eval_dis_results))
        )

        # Create the primary plot
        fig, ax1 = plt.subplots(figsize=(12, 10))  # creates a single figure and axis

        # Plot the primary data on ax1
        ax1.plot(x, y_mean, label="Estimated Value")
        ax1.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
        ax1.plot(a, y_rewards, label="Discounted Reward")
        ax1.set_xlabel("steps")
        ax1.set_ylabel("True Discounted Reward/ Estimates")
        ax1.legend(loc="upper left")

        # Create a secondary y-axis sharing the same x-axis
        ax2 = ax1.twinx()
        ax2.plot(
            x,
            y_mean - y_rewards,
            label="Estimation error (Q-value - reward)",
            color="green",
        )
        ax2.set_ylabel(
            "Estimation error (Pos. = Overest. / Neg. = Underest.)", color="green"
        )
        ax2.legend(loc="upper right")

        # Save the plot
        fig.savefig(f"{self.path}/overestimation-curve.png")  # save the figure

        plt.close(fig)  # Close the figure to free up memory
