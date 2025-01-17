import torch


class PlanningAlgo:
    def __init__(self, env, args):
        self.args = args
        self.device = args.device
        self.env = env
        self.horizon = args.planning_horizon  # MPC horizon

        self._nx, self._nu = (
            self.env.observation_space.shape,
            self.env.action_space.shape,
        )
        self._u_min = (
            torch.from_numpy(self.env.action_space.low).float().to(self.device)
        )
        self._u_max = (
            torch.from_numpy(self.env.action_space.high).float().to(self.device)
        )
        self._x_min = (
            torch.from_numpy(self.env.observation_space.low).float().to(self.device)
        )
        self._x_max = (
            torch.from_numpy(self.env.observation_space.high).float().to(self.device)
        )

    def plan(self):
        raise NotImplementedError("plan() not implemented")
