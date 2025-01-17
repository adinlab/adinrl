import torch

from agents.base_agent import Agent


############################################################
class ModelBasedAgent(Agent):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.policy_search_algo = None

    def set_policy_search_algo(self, policy_search_algo):
        del self.policy_search_algo
        self.policy_search_algo = policy_search_algo

    @torch.no_grad()
    def select_action(self, x):
        return self.policy_search_algo.select_action(x)

    def learn(self, max_iter=1):
        raise NotImplementedError
