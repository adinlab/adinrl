import torch

from agents.model_based_agent import ModelBasedAgent


class ModelPredictiveControlAgent(ModelBasedAgent):
    def __init__(self, env, args):
        super().__init__(env, args)

    @torch.no_grad()
    def select_action(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        return self.policy_search_algo.plan(x).cpu().numpy()
