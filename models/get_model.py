import models.bootdqnprior as bdqnv2
from models.ben import BEN
from models.ddpg import DeepDeterministicPolicyGradient
from models.drnd import DRND
from models.oac import OptimisticActorCritic
from models.pac4sac import PAC4SAC
from models.pactd3 import PACBayesianTD3
from models.dac import DistributionalActorCritic
from models.redq import RandomEnsembleDoubleQLearning
from models.validationround import ValidationRound
from models.rsac import RiskSensitiveSAC
from models.sac import SoftActorCritic
from models.td3 import TwinDelayedDeepDeterministicPolicyGradient
from models.vbac import VariationalBayesianAC
from models.vbac_det import VariationalBayesianACDet
from nets.actor_nets import *
from nets.sequential_critic_nets import *


def get_model(args, env):
    model_name = args.model.lower()
    match model_name:
        case "sac":
            return SoftActorCritic(env, args)

        case "ddpg":
            return DeepDeterministicPolicyGradient(
                env, args, ActorNet, SequentialCriticNet
            )

        case "td3":
            return TwinDelayedDeepDeterministicPolicyGradient(
                env, args, ActorNet, SequentialCriticNet
            )

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

        case "oac":
            return OptimisticActorCritic(
                env, args, ActorNetProbabilistic, SequentialCriticNet
            )

        case "dac":
            return DistributionalActorCritic(env, args)

        case "pac4sac":
            return PAC4SAC(env, args)

        case str(name) if "redq" in name:
            return RandomEnsembleDoubleQLearning(env, args)
        
        case str(name) if "validation" in name:
            return ValidationRound(env, args)

        case "rsac":
            return RiskSensitiveSAC(
                env, args, ActorNetProbabilistic, CriticNetEpistemic
            )

        case "pactd3-recent":
            return PACBayesianTD3(
                env, args, ActorNetProbabilistic, SequentialCriticNetProbabilistic
            )

        case _:
            raise ValueError(f"Unknown model: {model_name}")
