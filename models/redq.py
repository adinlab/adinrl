import torch

from models.basic.ac import ActorCritic
from models.basic.critic import Critics, Critic
from models.sac import SoftActor
from nets.actor_nets import ActorNetProbabilistic
from nets.critic_nets import CriticNet


##################################################################################################


class REDQCritics(Critics):
    def __init__(self, arch, args, n_state, n_action, critictype=Critic):
        super().__init__(arch, args, n_state, n_action, critictype)
        self.n_in_target = 2

    def reduce(self, q_val_list):
        i_targets = torch.randint(0, self.n_members, (self.n_in_target,))
        return torch.stack([q_val_list[i] for i in i_targets], dim=-1).min(-1)[0]


#####################################################################
class RandomEnsembleDoubleQLearning(ActorCritic):
    _agent_name = "REDQ"

    def __init__(self, env, args, actor_nn=ActorNetProbabilistic, critic_nn=CriticNet):
        super().__init__(
            env,
            args,
            actor_nn,
            critic_nn,
            CriticEnsembleType=REDQCritics,
            ActorType=SoftActor,
        )

        self.actor.c = args.explore_noise
