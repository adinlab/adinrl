import math

import torch

from models.basic.ac import ActorCritic
from models.basic.actor import Actor
from models.basic.sequential.sequential_critic import CriticEnsemble
from nets.actor_nets import ActorNetProbabilistic
from nets.sequential_critic_nets import SequentialCriticNet


#####################################################################
class SequentialSoftActor(Actor):
    def __init__(self, arch, args, n_state, n_action, has_target=False):
        super().__init__(arch, args, n_state, n_action, has_target)
        self.H_target = -n_action[0]
        self.log_alpha = torch.tensor(
            math.log(args.alpha), requires_grad=True, device=args.device
        )
        self.optim_alpha = torch.optim.Adam([self.log_alpha], args.learning_rate)

    # TODO: Consider updating to the version that Abdullah uses
    def update_alpha(self, e):
        self.optim_alpha.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (e + self.H_target).detach()).mean()
        alpha_loss.backward()
        self.optim_alpha.step()

    def loss(self, s, critics):
        a, e = self.act(s)
        q_list = critics.Q(s, a)
        q = critics.reduce(q_list)
        return (-q + self.log_alpha.exp() * e).mean(), e

    def update(self, s, critics):
        self.optim.zero_grad()
        loss, e = self.loss(s, critics)
        loss.backward()
        self.optim.step()
        self.update_alpha(e)
        self.iter += 1


#####################################################################
class SequentialSoftActorCritic(ActorCritic):
    _agent_name = "SequentialSAC"

    def __init__(
        self,
        env,
        args,
        actor_nn=ActorNetProbabilistic,
        critic_nn=SequentialCriticNet,
        CriticEnsembleType=CriticEnsemble,
        ActorType=SequentialSoftActor,
    ):
        super().__init__(env, args, actor_nn, critic_nn, CriticEnsembleType, ActorType)
