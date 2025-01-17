import gymnasium as gym
import numpy as np


class NoisyActionWrapper(gym.ActionWrapper):
    def __init__(self, env, noise_act=0.1):
        super().__init__(env)
        self.noise_act = noise_act

    def step(self, action):
        if isinstance(self.action_space, gym.spaces.Discrete):
            if np.random.random() < self.noise_act:
                action = self.action_space.sample()
        else:
            eps = self.noise_act * np.random.randn(*action.shape)
            action = np.clip(
                action + eps, self.action_space.low, self.action_space.high
            )
        return self.env.step(action)


class NoisyObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_obs=0.1):
        super().__init__(env)
        self.noise_obs = noise_obs

    def observation(self, obs):
        if isinstance(obs, np.ndarray):
            eps = self.noise_obs * np.random.randn(*obs.shape)
            return obs + eps
        elif isinstance(obs, dict):
            noisy_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    eps = self.noise_obs * np.random.randn(*value.shape)
                    noisy_obs[key] = value + eps
                else:
                    noisy_obs[key] = value
            return noisy_obs
        else:
            raise ValueError("Unsupported")
