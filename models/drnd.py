import math

import torch
import torch as th
from torch import func as thf
from torch import nn

from models.basic.ac import ActorCritic
from models.basic.critic import Critics, Critic
from models.sequential.sequential_sac import SequentialSoftActor
from nets.actor_nets import ActorNetProbabilistic
from nets.critic_nets import CriticNet
from nets.utils import create_net
from utils.utils import dim_check


class DRNDBonus(nn.Module):
    def __init__(self, args, n_state, n_action):
        super().__init__()
        self.n_members = args.n_targets_drnd
        self.args = args
        self.n_state = n_state
        self.n_action = n_action
        self.alpha = 0.9  # via Yang et al. (2024)

        self.gen_net = lambda: create_net(
            n_state[0] + n_action[0],
            32,
            4,
            256,
            act="relu",
            has_norm=False,
        ).to(args.device)
        # Assume that dim(x) = (n_batch, vector)
        #   If dim(x) = (n_member, n_batch, vector) return as is
        self.expand = lambda x: (
            x.expand(self.n_members, *x.shape) if len(x.shape) < 3 else x
        )

        self.reset()

    def reset(self):
        self.priors = [self.gen_net() for _ in range(self.n_members)]

        self.prior_model = self.gen_net().to("meta")

        self.params_prior, self.buffers = thf.stack_module_state(self.priors)

        # Remove gradient requirements for prior mode
        for p in self.params_prior.items():
            p[1].requires_grad = False

        def _fmodel(base_model, params, buffers, x):
            return thf.functional_call(base_model, (params, buffers), (x,))

        self.forward_priors = thf.vmap(
            lambda p, b, x: _fmodel(self.prior_model, p, b, x)
        )

        self.predictor = self.gen_net()
        self.optim_pred = th.optim.Adam(self.predictor.parameters(), lr=1e-4)

    @th.no_grad()
    def get_priors(self, input):
        return self.forward_priors(self.params_prior, self.buffers, input)

    @th.no_grad()
    def bonus(self, s, a):
        sa = th.cat((s, a), -1)
        SA = self.expand(sa)

        prior_pred = self.get_priors(SA)

        mu = prior_pred.mean(0)
        mu2 = mu.pow(2)
        B2 = prior_pred.pow(2).mean(0)

        pred = self.predictor(sa)

        dim_check(pred, mu)

        fst = (pred - mu).pow(2).sum(1, keepdim=True)
        # TODO: Check for further numerical stability requirements in github
        # This clipping is an undocumented feature in the code
        # WARNING: The original code has both mean -> sqrt and sqrt -> mean
        #   The second one is what seems to be used more often...
        # snd = th.sqrt(((pred.pow(2) - mu2).abs() / (B2 - mu2)).clip(1e-3, 1).mean(1))
        snd = th.sqrt(((pred.pow(2) - mu2).abs() / (B2 - mu2)).clip(1e-3, 1)).mean(
            1, keepdim=True
        )

        return self.alpha * fst + (1 - self.alpha) * snd

    @th.no_grad
    def mu(self, sa):
        return self.get_priors(sa).mean(0)

    @th.no_grad
    def B2(self, sa):
        return self.get_priors(sa).pow(2).mean(0)

    def update_predictor(self, s, a):
        sa = th.cat((s, a), -1)
        self.optim_pred.zero_grad()
        c = th.randint(self.n_members, ())
        # TODO: Split it similarly to how the parallelensemble does it to save runtime
        ctar = self.get_priors(self.expand(sa))[c]
        pred = self.predictor(sa)

        dim_check(ctar, pred)
        loss = (pred - ctar).pow(2)
        loss.mean().backward()
        self.optim_pred.step()


class DRNDActor(SequentialSoftActor):
    def __init__(self, arch, args, n_state, n_action, has_target=False):
        super().__init__(arch, args, n_state, n_action, has_target)
        self.H_target = -n_action[0] * 0.5
        self.lambda_actor = 1.0
        self.log_alpha = torch.tensor(
            math.log(args.alpha), requires_grad=True, device=args.device
        )
        self.optim_alpha = torch.optim.Adam([self.log_alpha], args.learning_rate)

    def loss(self, s, critics, targets):
        a, e = self.act(s)
        q_list = critics.Q(s, a)
        q = critics.reduce(q_list)

        bonus = targets.bonus(s, a)

        dim_check(q, bonus)
        dim_check(e, bonus)

        return (-q + self.log_alpha.exp() * e + self.lambda_actor * bonus).mean(), e

    def update(self, s, critics, targets):
        self.optim.zero_grad()
        loss, e = self.loss(s, critics, targets)
        loss.backward()
        self.optim.step()
        self.update_alpha(e)
        self.iter += 1


class DRNDCritics(Critics):
    def __init__(self, arch, args, n_state, n_action, critictype=Critic):
        super().__init__(arch, args, n_state, n_action, critictype)
        self.lambda_critic = 1.0

    @torch.no_grad()
    def get_bellman_target(self, r, sp, done, actor, drnd_targets):
        alpha = actor.log_alpha.exp().detach() if hasattr(actor, "log_alpha") else 0
        ap, ep = actor.act(sp)

        qp = self.Q_t(sp, ap)
        if ep is None:
            ep = 0
        bonus = drnd_targets.bonus(sp, ap)
        red_qp = self.reduce(qp)
        dim_check(bonus, red_qp)
        qp_t = red_qp - alpha * ep - self.lambda_critic * bonus
        y = r.unsqueeze(-1) + (self.args.gamma * qp_t * (1 - done.unsqueeze(-1)))
        return y


class DRND(ActorCritic):
    _agent_name = "DRND"

    def __init__(
        self,
        env,
        args,
        actor_nn=ActorNetProbabilistic,
        critic_nn=CriticNet,
        CriticEnsembleType=DRNDCritics,
        ActorType=DRNDActor,
        BonusEnsemble=DRNDBonus,
    ):
        super().__init__(env, args, actor_nn, critic_nn, CriticEnsembleType, ActorType)

        self.bonus_ensemble = BonusEnsemble(args, self.dim_obs, self.dim_act)

    def learn(self, max_iter=1):
        if self.args.batch_size > len(self.experience_memory):
            return None

        for ii in range(max_iter):
            s, a, r, sp, done, step = self.experience_memory.sample_random(
                self.args.batch_size
            )

            y = self.critics.get_bellman_target(
                r, sp, done, self.actor, self.bonus_ensemble
            )
            self.critics.update(s, a, y)

            if self.n_iter % self.policy_delay == 0:
                self.actor.update(s, self.critics, self.bonus_ensemble)
            self.critics.update_target()
            self.bonus_ensemble.update_predictor(s, a)
            self.n_iter += 1
