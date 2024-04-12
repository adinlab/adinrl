from nets.actor_nets import *
from nets.critic_nets import *

from models.sac import SoftActorCritic
from models.ddpg import DeepDeterministicPolicyGradient
from models.td3 import TwinDelayedDeepDeterministicPolicyGradient


def get_model(args, env):
    model_name = args.model.lower()
    if model_name == "sac":
        return SoftActorCritic(env, args, ActorNetProbabilistic, CriticNet)
    elif model_name == "ddpg":
        return DeepDeterministicPolicyGradient(env, args, ActorNet, CriticNet)
    elif model_name == "td3":
        return TwinDelayedDeepDeterministicPolicyGradient(
            env, args, ActorNet, CriticNet
        )
    else:
        raise ValueError("Unknown model: {}".format(model_name))
