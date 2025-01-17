import torch
import torch as th
from torch import func as thf

from models.basic.ac import ActorCritic
from models.basic.actor import Actor
from models.basic.loss import ProbabilisticLoss
from models.basic.critic import Critics, Critic
from nets.actor_nets import ActorNetEnsemble
from nets.critic_nets import CriticNet
from utils.utils import dim_check
from utils.utils import tonumpy


##################################################################################################
class BootDQNCritic(Critic):
    def __init__(self, arch, args, n_state, n_action):
        super().__init__(arch, args, n_state, n_action)
        # self.loss = torch.nn.HuberLoss() if self.args.use_huber else torch.nn.MSELoss()
        # self.loss = torch.nn.MSELoss()
        self.loss = ModBootstrapEnsembleLoss(args)


class ModBootstrapEnsembleLoss(ProbabilisticLoss):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bootstrap_rate = self.args.bootstrap_rate

    def forward(self, q, y):
        bootstrap_mask = (torch.rand_like(q) >= self.bootstrap_rate) * 1.0
        emp_loss = ((q - y) * bootstrap_mask).pow(2)
        return emp_loss.mean()


##################################################################################################
class BootDQNPriorCritics(Critics):
    def __init__(self, arch, args, n_state, n_action, critictype=BootDQNCritic):
        super().__init__(arch, args, n_state, n_action, critictype)

        gen_prior = lambda: arch(
            self.n_state,
            self.n_action,
            depth=args.depth_critic,
            width=args.width_critic,
            act=args.act_critic,
            has_norm=not args.no_norm_critic,
        ).to(args.device)

        self.priors = [gen_prior() for _ in range(self.n_members)]
        self.prior_model = gen_prior().to("meta")
        self.prior_scale = args.prior_scale  # Taken from Osband's bsuite code

        self.params_prior, self.buffers = thf.stack_module_state(self.priors)

        for p in self.params_prior.items():
            p[1].requires_grad = False

        def _fmodel(base_model, params, buffers, x):
            return thf.functional_call(base_model, (params, buffers), (x,))

        self.forward_priors = thf.vmap(
            lambda p, b, x: _fmodel(self.prior_model, p, b, x)
        )

    @th.no_grad()
    def get_prior(self, input):
        return self.forward_priors(self.params_prior, self.buffers, input)

    @torch.no_grad()
    def get_bellman_target(self, r, sp, done, actor):
        ap = actor.get_action(sp)
        sp = self.expand(sp)
        ap = ap.swapaxes(0, 1)
        qp_t = self.Q_t(sp, ap)
        assert len(r.shape) == 1
        # To allow broadcasting to (n_members, reward, 1)
        r = r[None, :, None]
        done = done[None, :, None]
        q_t = r + (self.args.gamma * qp_t * (1 - done))
        return q_t

    @torch.no_grad()
    def Q_t(self, s, a):
        SA = self.expand(torch.cat((s, a), -1))
        return (
            self.forward_target(self.params_target, self.buffers_target, SA)
            + self.prior_scale * self.get_prior(SA).detach()
        )

    def Q(self, s, a):
        SA = self.expand(th.cat((s, a), -1))
        return (
            self.forward_model(self.params_model, self.buffers_model, SA)
            + self.prior_scale * self.get_prior(SA).detach()
        )

    def update(self, s, a, y):
        self.optim.zero_grad()
        pred = self.Q(s, a)
        dim_check(pred, y)
        self.loss(pred, y).backward()
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
        self.actions = []
        self.states = []
        self.q_vars_mean = []
        self.q_vars_std = []
        self.print_freq = 500

    def act(self, s, is_training=True):
        a, e = self.model(s)

        if is_training:
            if self.is_episode_end or (self.interaction_iter % self.sampling_rate == 0):
                self.idx_active_critic = torch.randint(a.size(1), ())
            self.interaction_iter += 1
            a = a[:, self.idx_active_critic, :].squeeze()

            if self.args.verbose and self.iter % self.print_freq == 0:
                self.states.append(tonumpy(s))

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
        q = critics.Q(s, a)

        return (-q).mean(), None


#####################################################################
class BootDQNPrior(ActorCritic):
    _agent_name = "BootDQNPrior"

    def __init__(
        self,
        env,
        args,
        actor_nn=ActorNetEnsemble,
        critic_nn=CriticNet,
        CriticEnsembleType=BootDQNPriorCritics,
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
