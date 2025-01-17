import torch

from agents.model_predictive_control_agent import ModelPredictiveControlAgent
from planning.ddp import DDP


class DifferentialDynamicProgrammingWithKnwonEnv(ModelPredictiveControlAgent):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.model_nn = env.env.env.env._state_eq_torch
        self.cost_nn = lambda x, u: 0.5 * torch.sum(torch.square(u))
        self.critic_nn = lambda x: 0.5 * (
            torch.square(1.0 - torch.cos(x[2]))
            + torch.square(x[1])
            + torch.square(x[3])
        )
        self.ddp = DDP(self.env, self.args, self.model_nn, self.cost_nn, self.critic_nn)

    # @torch.no_grad()
    def select_action(self, x):
        del self.ddp
        self.ddp = DDP(self.env, self.args, self.model_nn, self.cost_nn, self.critic_nn)
        self.set_policy_search_algo(self.ddp)
        return self.policy_search_algo.plan(x)

    def learn(self, max_iter=1):
        pass


class DifferentialDynamicProgrammingWithModel(ModelPredictiveControlAgent):
    def __init__(self, env, args, model_nn):
        super().__init__(env, args)
        self.model_net = model_nn(self.dim_obs, self.dim_act, args.n_hidden).to(
            self.device
        )
        self.cost_nn = lambda x, u: 0.5 * torch.sum(torch.square(u))
        self.critic_nn = lambda x: 0.5 * (
            torch.square(1.0 - torch.cos(x[2]))
            + torch.square(x[1])
            + torch.square(x[3])
        )

        self.optim_model = torch.optim.Adam(
            self.model_net.parameters(), args.learning_rate
        )
        self.ddp = DDP(self.env, self.args, self.model_nn, self.cost_nn, self.critic_nn)

    def select_action(self, x):
        del self.ddp
        self.ddp = DDP(self.env, self.args, self.model_nn, self.cost_nn, self.critic_nn)
        self.set_policy_search_algo(self.ddp)
        return self.policy_search_algo.plan(x)

    def model_nn(self, x, u):
        self.optim_model.zero_grad()
        sp, _ = self.model_net(x, u)
        return sp

    def learn_model(self, samples):
        self.optim_model.zero_grad()
        loss_model = 0
        s_pred = samples[0][0]
        for h in range(self.ssm_horizon):
            s_pred, _ = self.model_net(s_pred, samples[h][1])
            loss_model += ((s_pred - samples[h][3]) ** 2).mean()

        loss_model.backward()
        self.optim_model.step()

    def learn(self, max_iter=1):
        for iter in range(max_iter):
            try:
                samples = self.experience_memory.sample_random_sequence_snippet(
                    self.args.batch_size, self.ssm_horizon
                )
            except:
                continue

            self.learn_model(samples)


class DifferentialDynamicProgrammingWithCritic(ModelPredictiveControlAgent):
    def __init__(self, env, args, critic_nn):
        super().__init__(env, args)
        self.model_nn = env.env.env.env._state_eq_torch
        self.cost_nn = lambda x, u: 0.5 * torch.sum(torch.square(u))
        self.critic_net = critic_nn(self.dim_obs, args.n_hidden).to(self.device)

        self.optim_critic = torch.optim.Adam(
            self.critic_net.parameters(), args.learning_rate
        )
        self.ddp = DDP(self.env, self.args, self.model_nn, self.cost_nn, self.critic_nn)

        self.gamma = 0.99

    def select_action(self, x):
        del self.ddp
        self.ddp = DDP(self.env, self.args, self.model_nn, self.cost_nn, self.critic_nn)
        self.set_policy_search_algo(self.ddp)
        return self.policy_search_algo.plan(x)

    def critic_nn(self, x):
        self.optim_critic.zero_grad()
        return self.critic_net(x)

    def learn_critic(self, samples):
        self.optim_critic.zero_grad()
        loss_critic = 0
        for h in range(self.ssm_horizon):
            s = samples[h][0]
            s_p = samples[h][3]
            reward = samples[h][2]
            loss_critic += (
                (self.critic_net(s) - (reward + self.gamma * self.critic_net(s_p))) ** 2
            ).mean()

        loss_critic.backward()
        self.optim_critic.step()

    def learn(self, max_iter=1):
        for iter in range(max_iter):
            try:
                samples = self.experience_memory.sample_random_sequence_snippet(
                    self.args.batch_size, self.ssm_horizon
                )
            except:
                continue

            self.learn_critic(samples)


class DifferentialDynamicProgrammingWithModelCritic(ModelPredictiveControlAgent):
    def __init__(self, env, args, model_nn, critic_nn):
        super().__init__(env, args)
        self.model_net = model_nn(self.dim_obs, self.dim_act, args.n_hidden).to(
            self.device
        )
        self.cost_nn = lambda x, u: 0.5 * torch.sum(torch.square(u))
        self.critic_net = critic_nn(self.dim_obs, args.n_hidden).to(self.device)

        self.optim_model = torch.optim.Adam(
            self.model_net.parameters(), args.learning_rate
        )
        self.optim_critic = torch.optim.Adam(
            self.critic_net.parameters(), args.learning_rate
        )
        self.ddp = DDP(self.env, self.args, self.model_nn, self.cost_nn, self.critic_nn)

    def select_action(self, x):
        del self.ddp
        self.ddp = DDP(self.env, self.args, self.model_nn, self.cost_nn, self.critic_nn)
        self.set_policy_search_algo(self.ddp)
        return self.policy_search_algo.plan(x)

    def model_nn(self, x, u):
        self.optim_model.zero_grad()
        sp, _ = self.model_net(x, u)
        return sp

    def critic_nn(self, x):
        self.optim_critic.zero_grad()
        return self.critic_net(x)

    def learn_critic(self, samples):
        self.optim_critic.zero_grad()
        loss_critic = 0
        for h in range(self.ssm_horizon):
            s = samples[h][0]
            s_p = samples[h][3]
            reward = samples[h][2]
            loss_critic += (
                (self.critic_net(s) - (reward + self.gamma * self.critic_net(s_p))) ** 2
            ).mean()

        loss_critic.backward()
        self.optim_critic.step()

    def learn(self, max_iter=1):
        for iter in range(max_iter):
            try:
                samples = self.experience_memory.sample_random_sequence_snippet(
                    self.args.batch_size, self.ssm_horizon
                )
            except:
                continue

            self.learn_model(samples)
            self.learn_critic(samples)
