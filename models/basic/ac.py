import torch

from agents.base_agent import Agent
from models.basic.actor import Actor
from models.basic.sequential.sequential_critic import CriticEnsemble
from models.basic.critic import Critics


#####################################################################
class ActorCritic(Agent):
    _agent_name = "AC"

    def __init__(
        self,
        env,
        args,
        actor_nn,
        critic_nn,
        CriticEnsembleType=CriticEnsemble,
        ActorType=Actor,
    ):
        super().__init__(env, args)
        self.critics = CriticEnsembleType(critic_nn, args, self.dim_obs, self.dim_act)
        self.actor = ActorType(actor_nn, args, self.dim_obs, self.dim_act)
        self.n_iter = 0
        self.policy_delay = args.policy_delay

    def set_writer(self, writer):
        self.writer = writer
        self.actor.set_writer(writer)
        self.critics.set_writer(writer)

    def learn(self, max_iter=1, n_epochs=0):
        if self.args.batch_size > len(self.experience_memory):
            return None

        if n_epochs > 0:
            # get an initial set of batches if it hasn't been shuffled.
            # after the initial shuffle reshuffling will happen automatically
            if not self.experience_memory.batches:
                self.experience_memory.shuffle(self.args.batch_size)
            n_steps = self.args.n_epochs * len(self.experience_memory.batches)
            sampler = self.experience_memory.sample_batch

        else:
            n_steps = max_iter
            sampler = self.experience_memory.sample_random

        for ii in range(n_steps):
            s, a, r, sp, done, step = sampler(self.args.batch_size)
            y = self.critics.get_bellman_target(r, sp, done, self.actor)
            self.critics.update(s, a, y)

            if self.n_iter % self.policy_delay == 0:
                self.actor.update(s, self.critics)
            self.critics.update_target()
            self.n_iter += 1

    @torch.no_grad()
    def select_action(self, s, is_training=True):
        a, _ = self.actor.act(s, is_training=is_training)
        return a

    def Q_value(self, s, a):

        if len(s.shape) == 1:
            s = s[None]
        if len(a.shape) == 1:
            a = a[None]
        if isinstance(self.critics, Critics):
            self.critics.unstack(target=False, single=True)

        q = self.critics[0].Q(s, a)
        return q.item()
