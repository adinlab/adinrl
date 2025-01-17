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
            "walker2d": "Walker2d-v4",
            "hopper": "Hopper-v4",
            "ant": "Ant-v4",
            "humanoid": "Humanoid-v4",
            "cheetah": "HalfCheetah-v4",
            "reacher": "Reacher-v4",
            "swimmer": "Swimmer-v4",
            "pusher": "Pusher-v4",
            "humanoidstandup": "HumanoidStandup-v4",
            "inverteddoublependulum": "InvertedDoublePendulum-v4",
            # deepmind control
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
            "acrobot": "acrobot-swingup_sparse",
            "cartpole": "cartpole-swingup_sparse",
            "ballincup": "ball_in_cup-catch",
            "pendulum": "pendulum-swingup",
            # metaworld
            "mwbutton": "metaworld_button-press-v2",
            "mwdoor": "metaworld_door-open-v2",
            "mwdrawer": "metaworld_drawer-close-v2",
            "mwdrawer2": "metaworld_drawer-open-v2",
            "mwhammer": "metaworld_hammer-v2",
            "mwpeg": "metaworld_peg-insert-side-v2",
            "mwpick": "metaworld_pick-place-v2",
            "mwpush": "metaworld_push-v2",
            "mwreach": "metaworld_reach-v2",
            "mwwindow": "metaworld_window-open-v2",
            "mwwindow2": "metaworld_window-close-v2",
            "mwbasketball": "metaworld_basketball-v2",
            "mwdialturn": "metaworld_dial-turn-v2",
            "mwsweep": "metaworld_sweep-into-v2",
            "mwassembly": "metaworld_assembly-v2",
            # maze
            "umaze": "AntMaze_UMaze-v4",
        }

        # control coefficients
        ctl_coefs = {
            "ant": 0.5,
            "cheetah": 0.1,
            "humanoid": 0.1,
            "hopper": 0.001,
            "walker2d": 0.001,
            "swimmer": 0.0001,
        }

        if "-" in self.args.env:
            prefix, delay, suffix = (self.args.env).split("-")
            if prefix == "sp":
                pos_delay = float(delay)
                print("Position delay: ", pos_delay)
                ctrl_cost_weight = ctl_coefs[suffix]
        else:
            suffix = self.args.env
            pos_delay = None
            ctrl_cost_weight = None

        env_name = environments[suffix]

        self.env = make_env(
            env_name,
            self.args.seed,
            noisy_act=self.args.noisy_act,
            noisy_obs=self.args.noisy_obs,
            position_delay=pos_delay,
            ctrl_cost_weight=ctrl_cost_weight,
        )

        self.eval_env = make_env(
            env_name,
            self.args.seed + 100,
            noisy_act=self.args.noisy_act,
            noisy_obs=self.args.noisy_obs,
            position_delay=pos_delay,
            ctrl_cost_weight=ctrl_cost_weight,
        )

        self.agent = get_model(self.args, self.env)
        self.agent.logger.log(f"Model: \n{self.agent}")
        if self.args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter()
        else:
            self.writer = None
        self.agent.set_writer(self.writer)

    def train(self):
        raise NotImplementedError(
            f"train() not implemented for {self.__class__.__name__}!"
        )

    def test(self):
        raise NotImplementedError(
            f"test() not implemented for {self.__class__.__name__}!"
        )
