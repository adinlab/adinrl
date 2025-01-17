import torch
import math
from models.basic.ac import ActorCritic
from models.basic.critic import Critics, Critic
from models.sac import SoftActor
from models.sequential.sequential_sac import SequentialSoftActor
from models.basic.actor import Actor
from nets.actor_nets import ActorNetProbabilistic
from nets.critic_nets import CriticNet


##################################################################################################


class REDQCritics(Critics):
    def __init__(self, arch, args, n_state, n_action, critictype=Critic):
        super().__init__(arch, args, n_state, n_action, critictype)
        self.n_in_target = 2
        params = torch.load(f"_logs/{self.args.env}/redq/seed_0{self.args.seed}/params.pth")
        for i in range(args.n_critics):
            self.critics_model[i].load_state_dict(params[i]["params_model"])
            self.critics_target[i].load_state_dict(params[i]["params_target"])
        self.optim.load_state_dict(params[0]["optim"])

    def reduce(self, q_val_list):
        i_targets = torch.randint(0, self.n_members, (self.n_in_target,))
        return torch.stack([q_val_list[i] for i in i_targets], dim=-1).min(-1)[0]
    



######################################################################
class SoftActor(Actor):
    def __init__(self, arch, args, n_state, n_action, has_target=False):
        super().__init__(arch, args, n_state, n_action, has_target)
        params_ = torch.load(f"_logs/{self.args.env}/redq/seed_0{self.args.seed}/Actor_params.pth")
        self.model.load_state_dict(params_["params_model"])
        # self.H_target = -n_action[0] * 0.5
        # self.log_alpha = torch.tensor(
        #     math.log(args.alpha), requires_grad=True, device=args.device
        # )
        # self.optim_alpha = torch.optim.Adam([self.log_alpha], args.learning_rate)

    def loss(self, s, critics):
        pass
        # a, e = self.act(s)
        # q_list = critics.Q(s, a)
        # q = critics.reduce(q_list)
        # return (-q + self.log_alpha.exp() * e).mean(), e

        # TODO: Consider updating to the version that Abdullah uses
    def update_alpha(self, e):
        # self.optim_alpha.zero_grad()
        # alpha_loss = -(self.log_alpha.exp() * (e + self.H_target).detach()).mean()
        # alpha_loss.backward()
        # self.optim_alpha.step()
        pass

    def update(self, s, critics):
        # self.optim.zero_grad()
        # loss, e = self.loss(s, critics)
        # loss.backward()
        # self.optim.step()
        # self.update_alpha(e)
        self.iter += 1
#####################################################################
class ValidationRound(ActorCritic):
    _agent_name = "validation_round"

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