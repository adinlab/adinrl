import math

import torch

from models.basic.ac import ActorCritic
from models.basic.critic import Critics

from models.sequential.sequential_sac import SequentialSoftActor
from nets.actor_nets import ActorNetProbabilistic
from nets.critic_nets import CriticNet


class SoftActor(SequentialSoftActor):
    def __init__(self, arch, args, n_state, n_action, has_target=False):
        super().__init__(arch, args, n_state, n_action, has_target)
        self.H_target = -n_action[0] * 0.5
        self.log_alpha = torch.tensor(
            math.log(args.alpha), requires_grad=True, device=args.device
        )
        self.optim_alpha = torch.optim.Adam([self.log_alpha], args.learning_rate)

    def loss(self, s, critics):
        a, e = self.act(s)
        q_list = critics.Q(s, a)
        q = critics.reduce(q_list)
        return (-q + self.log_alpha.exp() * e).mean(), e


class SoftActorCritic(ActorCritic):
    _agent_name = "SAC"

    def __init__(
        self,
        env,
        args,
        actor_nn=ActorNetProbabilistic,
        critic_nn=CriticNet,
        CriticEnsembleType=Critics,
        ActorType=SoftActor,
    ):
        super().__init__(env, args, actor_nn, critic_nn, CriticEnsembleType, ActorType)
