import torch

from models.basic.ac import ActorCritic
from models.basic.sequential.sequential_critic import CriticEnsemble, Critic
from models.sequential.sequential_sac import SequentialSoftActor
from nets.actor_nets import ActorNetProbabilistic
from nets.sequential_critic_nets import SequentialCriticNet


##################################################################################################


class REDQCriticEnsemble(CriticEnsemble):
    def __init__(self, arch, args, n_state, n_action, critictype=Critic):
        super().__init__(arch, args, n_state, n_action, critictype)
        self.n_in_target = 2

    def reduce(self, q_val_list):
        i_targets = torch.randperm(self.n_elements)[: self.n_in_target]
        return torch.stack([q_val_list[i] for i in i_targets], dim=-1).min(-1)[0]


#####################################################################
class SequentialRandomEnsembleDoubleQLearning(ActorCritic):
    _agent_name = "SequentialREDQ"

    def __init__(
        self, env, args, actor_nn=ActorNetProbabilistic, critic_nn=SequentialCriticNet
    ):
        super().__init__(
            env,
            args,
            actor_nn,
            critic_nn,
            CriticEnsembleType=REDQCriticEnsemble,
            ActorType=SequentialSoftActor,
        )

        self.actor.c = args.explore_noise
