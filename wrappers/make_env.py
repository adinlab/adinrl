"""
This is adapted from https://github.com/ikostrikov/jaxrl

"""

import gymnasium as gym

import random

from gymnasium.wrappers import RescaleAction
from typing import Optional
import numpy as np
import torch

import wrappers
from utils.environment.noisywrappers import NoisyActionWrapper, NoisyObservationWrapper
from utils.environment.reward_wrappers import PositionDelayWrapper


def make_env(
    env_name: str,
    seed: int,
    save_folder: Optional[str] = None,
    add_episode_monitor: bool = True,
    action_repeat: int = 1,
    frame_stack: int = 1,
    from_pixels: bool = False,
    pixels_only: bool = True,
    image_size: int = 84,
    sticky: bool = False,
    gray_scale: bool = False,
    flatten: bool = True,
    terminate_when_unhealthy: bool = True,
    action_concat: int = 1,
    obs_concat: int = 1,
    continuous: bool = True,
    noisy_act: float = 0.0,
    noisy_obs: float = 0.0,
    position_delay: float = None,
    ctrl_cost_weight: float = None,
) -> gym.Env:

    # Check if the env is in gym.
    env_ids = list(gym.envs.registry.keys())

    if env_name in env_ids:
        env = gym.make(env_name)
        save_folder = None

    elif "Navigation" in env_name:
        env = wrappers.NavigationND(env_name)

    elif "metaworld" in env_name:
        env_name = "_".join(env_name.split("_")[1:])
        mt1 = metaworld.MT1(env_name, seed=0)
        env = mt1.train_classes[env_name]()
        task = random.choice(mt1.train_tasks)
        env.set_task(task)
        env = wrappers.OldToNewGym(env, duration=env.max_path_length)
        save_folder = None

    else:
        domain_name, task_name = env_name.split("-")
        env = wrappers.DMCGym(
            domain=domain_name, task=task_name, task_kwargs={"random": seed}
        )
        # env=dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=seed)

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
        env = wrappers.FlattenAction(env)

    # if add_episode_monitor:
    #     env = wrappers.EpisodeMonitor(env)

    if action_repeat > 1:
        env = wrappers.RepeatAction(env, action_repeat)

    if continuous:
        env = RescaleAction(env, -1.0, 1.0)

    # if action_concat > 1:
    #     env = wrappers.ConcatAction(env, action_concat)
    # if obs_concat > 1:
    #     env = wrappers.ConcatObs(env, obs_concat)

    # if save_folder is not None:
    #     env = wrappers.VideoRecorder(env, save_folder=save_folder)

    # TODO: should probably move this inside DMC
    if from_pixels:
        if env_name in env_ids:
            camera_id = 0
        else:
            camera_id = 2 if domain_name == "quadruped" else 0
        env = PixelObservationWrapper(
            env,
            pixels_only=pixels_only,
            render_kwargs={
                "pixels": {
                    "height": image_size,
                    "width": image_size,
                    "camera_id": camera_id,
                }
            },
        )
        env = wrappers.TakeKey(env, take_key="pixels")
        if gray_scale:
            env = wrappers.RGB2Gray(env)
    else:
        env = wrappers.SinglePrecision(env)

    if frame_stack > 1:
        env = wrappers.FrameStack(env, num_stack=frame_stack)

    if sticky:
        env = wrappers.StickyActionEnv(env)

    if noisy_act > 0:
        env = NoisyActionWrapper(env, noise_act=noisy_act)
    if noisy_obs > 0:
        env = NoisyObservationWrapper(env, noise_obs=noisy_obs)

    if position_delay is not None or ctrl_cost_weight is not None:
        print("Using sparse version")
        env = PositionDelayWrapper(
            env, position_delay=position_delay, ctrl_w=ctrl_cost_weight
        )

    ############################################################
    # perturb environment observation and reward
    # env = TransformReward(env, lambda r: 0.01*r)
    # env = TransformObservation(env, lambda obs: obs + 0.01 * np.random.randn(*obs.shape))
    # env = TransformObservation(
    #    env,
    #    lambda obs: obs
    #    + ((5 * np.sqrt(np.linalg.norm(obs)) / (obs.shape[0]) ** 2))
    #    * np.random.randn(*obs.shape),
    # )
    ############################################################

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return env
