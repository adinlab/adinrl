import gymnasium as gym
import numpy as np


class PositionDelayWrapper(gym.RewardWrapper):
    def __init__(self, env, position_delay=2, ctrl_w=0.001):
        super().__init__(env)
        self.position_delay = position_delay
        self.ctrl_w = ctrl_w

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        info["x_pos"] = self.data.qpos[0]
        info["action_norm"] = np.sum(np.square(action))
        return (
            observation,
            self.reward(observation, action),
            terminated,
            truncated,
            info,
        )

    def reward(self, observation, action):
        x_pos = self.data.qpos[0]
        x_vel = self.data.qvel[0]
        ctrl_cost = self.ctrl_w * np.sum(np.square(action))
        forward_reward = (x_pos >= self.position_delay) * x_vel
        rewards = forward_reward - ctrl_cost
        return rewards
