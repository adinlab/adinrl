import models.bootdqnprior as bdqnv2
from models.ben import BEN
from models.drnd import DRND
from models.dac import DistributionalActorCritic
from models.redq import RandomEnsembleDoubleQLearning
from models.sac import SoftActorCritic
from models.vbac import VariationalBayesianAC
from models.vbac_det import VariationalBayesianACDet
from nets.actor_nets import *
from nets.sequential_critic_nets import *


def get_model(args, env):
    model_name = args.model.lower()
    match model_name:
        case "sac":
            return SoftActorCritic(env, args)

        case str(name) if "drnd" in name:
            return DRND(env, args)

        case str(name) if "ben" in name:
            return BEN(env, args)

        case str(name) if "bootdqnv2" in name:
            return bdqnv2.BootDQNPrior(env, args)

        case "vbac":
            return VariationalBayesianAC(env, args)

        case "vbac_det":
            return VariationalBayesianACDet(env, args)

        case "dac":
            return DistributionalActorCritic(env, args)

        case str(name) if "redq" in name:
            return RandomEnsembleDoubleQLearning(env, args)

        case _:
            raise ValueError(f"Unknown model: {model_name}")
