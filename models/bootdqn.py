import torch

from models.basic.ac import ActorCritic
from models.basic.actor import Actor
from models.basic.loss import ProbabilisticLoss
from models.basic.critic import Critics, Critic
from nets.actor_nets import ActorNetEnsemble
from nets.critic_nets import CriticNet
from utils.utils import tonumpy


#####################################################################
class BootstrapEnsembleLoss(ProbabilisticLoss):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bootstrap_rate = self.args.bootstrap_rate

    def forward(self, q, y, weights):
        bootstrap_mask = (torch.rand_like(q) >= self.bootstrap_rate) * 1.0
        emp_loss = ((q - y) * bootstrap_mask) ** 2
        prior_loss = weights.pow(2).sum() * self.args.dqn_l2_reg
        return emp_loss.mean() + prior_loss


##################################################################################################
class BootDQNCritic(Critic):
    def __init__(self, arch, args, n_state, n_action):
        super().__init__(arch, args, n_state, n_action)
        self.loss = BootstrapEnsembleLoss(args)


##################################################################################################
class BootDQNCritics(Critics):
    def __init__(self, arch, args, n_state, n_action, critictype=BootDQNCritic):
        super().__init__(arch, args, n_state, n_action, critictype)

    @torch.no_grad()
    def get_bellman_target(self, r, sp, done, actor):
        ap = actor.get_action(sp)
        sp = self.expand(sp)
        ap = ap.swapaxes(0, 1)
        SA = torch.cat((sp, ap), 2)
        qp_t = self.forward_target(self.params_target, self.buffers_target, SA)
        q_t = r.unsqueeze(-1) + (self.args.gamma * qp_t * (1 - done.unsqueeze(-1)))
        return q_t

    @torch.no_grad()
    def Q_t(self, s, a):
        if len(a.shape) == 1:
            a = a.view(-1, 1)
        SA = self.expand(torch.cat((s, a), -1))
        return self.forward_target(self.params_target, self.buffers_target, SA)

    def update(self, s, a, y):
        self.optim.zero_grad()
        l_layer = len(self.base_model.arch) - 1
        weights = self.params_model[f"arch.{l_layer}.weight"]
        self.loss(self.Q(s, a), y, weights).backward()
        self.optim.step()
        self.iter += 1


#####################################################################
class ThompsonActor(Actor):
    def __init__(self, arch, args, n_state, n_action):
        super().__init__(arch, args, n_state, n_action)
        self.idx_active_critic = 0
        self.interaction_iter = 0
        self.sampling_rate = self.args.posterior_sampling_rate

        self.iter_num = []
        # self.actions = []
        # self.states = []
        self.q_vars_mean = []
        self.q_vars_std = []
        self.print_freq = 500

    def act(self, s, is_training=True):
        a, e = self.model(s)

        if is_training:
            if self.is_episode_end or (self.interaction_iter % self.sampling_rate == 0):
                self.idx_active_critic = torch.randint(0, a.size(1), (1,)).item()
            self.interaction_iter += 1
            a = a[:, self.idx_active_critic, :].squeeze()

            if self.args.verbose and self.iter % self.print_freq == 0:
                self.states.append(tonumpy(s))

            # TODO: Move to separate function
            # if self.args.verbose and self.iter % self.print_freq == 0:
            #     self.iter_num.append(self.iter / self.args.max_iter)
            #     self.actions.append(a.detach().item())
            #     self.states.append(s[[0, 2]].cpu().detach().view(1, -1))
            #     # plt.plot(self.iter_num,self.actions, label="Actions")
            #     action_ds = pd.DataFrame(self.actions, columns=["actions"])
            #     sns.displot(
            #         action_ds, x="actions", kind="kde", fill=True, color="orange"
            #     )
            #     plt.savefig(
            #         f"../_logs/{self.args.env}/{self.args.model}/seed_0{self.args.seed}/actions.png"
            #     )
            #     plt.close()
            #
            #     state_np = torch.cat(self.states, dim=0).numpy()
            #     # convert to sns dataframe
            #     state_np = pd.DataFrame(
            #         state_np, columns=["cos(theta1)", "cos(theta2)"]
            #     )
            #     sns.kdeplot(
            #         data=state_np,
            #         x="cos(theta1)",
            #         y="cos(theta2)",
            #         fill=True,
            #         color="orange",
            #     )
            #     plt.savefig(
            #         f"../_logs/{self.args.env}/{self.args.model}/seed_0{self.args.seed}/states.png"
            #     )
            #     plt.close()

        else:
            a = a.mean(dim=1).squeeze()

        return a, e

    def get_action(self, s):
        a, _ = self.model(s)
        return a

    def loss(self, s, critics):
        a = self.get_action(s)
        s = critics.expand(s)
        a = a.swapaxes(0, 1)
        SA = torch.cat((s, a), 2)
        q = critics.forward_model(critics.params_model, critics.buffers_model, SA)

        # if self.iter % self.print_freq == 0 and self.args.verbose:
        #     # TODO: Fix this
        #     q_var_mean = q.var(dim=0).mean().cpu().detach()
        #     q_var_std = q.var(dim=0).mean().cpu().detach()
        #     self.q_vars_mean.append(q_var_mean)
        #     self.q_vars_std.append(q_var_std)
        #     # q_vars = torch.cat([q_var_mean.unsqueeze(0), q_var_std.unsqueeze(0)], dim=0)
        #     plt.plot(self.iter_num, self.q_vars_mean, label="Q variances")
        #     # plt.fill_between(self.iter_num, q_vars[0], q_vars[1], alpha=0.5, label="Q variances")
        #     plt.xlabel("Iter")
        #     plt.ylabel("Q variances")
        #     plt.legend()
        #     plt.savefig(
        #         f"../_logs/{self.args.env}/{self.args.model}/seed_0{self.args.seed}/q_var.png"
        #     )
        #     plt.close()

        return (-q).mean(), None


#####################################################################
class BootDQN(ActorCritic):
    _agent_name = "BootDQN"

    def __init__(
        self,
        env,
        args,
        actor_nn=ActorNetEnsemble,
        critic_nn=CriticNet,
        CriticEnsembleType=BootDQNCritics,
        ActorType=ThompsonActor,
    ):
        super().__init__(
            env,
            args,
            actor_nn,
            critic_nn,
            CriticEnsembleType=CriticEnsembleType,
            ActorType=ActorType,
        )
