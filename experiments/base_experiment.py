from models.get_model import get_model
from wrappers import make_env


class Experiment(object):
    def __init__(self, args):
        self.args = args
        self.n_total_steps = 0
        self.max_steps = self.args.max_steps

        environments = {
            # mujoco
            "MountainCar": "MountainCarContinuous-v0",
            "noisyMountainCar": "noisyMountainCarContinuous-v0",
            "noisyacrobot": "noisyacrobot-v0",
            "noisypendulum": "noisypendulum-v0",
            "walker2d": "Walker2d-v4",
            "hopper": "Hopper-v4",
            "ant": "Ant-v4",
            "humanoid": "Humanoid-v4",
            "cheetah": "HalfCheetah-v4",
            # deepmind control
            "dmcartpole": "cartpole-swingup",
            "dmacrobot": "acrobot-swingup",
            "dmfinger": "finger-turn_hard",
            "dmant": "quadruped-run",
            "dmhopper": "hopper-hop",
            "dmreacher": "reacher-hard",
            "dmwalker": "walker-run",
            "dmfish": "fish-swim",
            "dmcheetah": "cheetah-run",
            "dmhumanoid": "humanoid-run",
            "dmswimmer": "swimmer-swimmer15",
            "dmdog": "dog-run",
        }
        self.env = make_env(environments[self.args.env], self.args.seed)
        self.eval_env = make_env(environments[self.args.env], self.args.seed + 100)
        self.agent = get_model(self.args, self.env)
        self.agent.logger.log(f"Model: \n{self.agent}")

    def train(self):
        raise NotImplementedError(
            f"train() not implemented for {self.__class__.__name__}!"
        )

    def test(self):
        raise NotImplementedError(
            f"test() not implemented for {self.__class__.__name__}!"
        )
