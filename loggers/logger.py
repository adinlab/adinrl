import logging
import os
import numpy as np
import torch
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, args):
        self.path = f"{args.result_path}/{args.env}/{args.model}/experiment_{str(args.runid).zfill(2)}"
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


        state = {
           'model_state_dict': model.state_dict()
        }
        torch.save(state, f"{self.path}/model.pt")

    def IQM_reward_calculator(self, rewards):
        sorted = np.sort(rewards)
        q1 = np.percentile(sorted, 25)
        q3 = np.percentile(sorted, 75)
        return np.mean(sorted[(sorted >= q1) & (sorted <= q3)])

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
