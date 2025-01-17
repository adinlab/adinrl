import pickle as pkl
import time

from experiments.control_experiment import ControlExperiment
from models.get_model import get_model
from replay_buffers.experience_memory import ExperienceMemoryNumpy


class OfflineControlExperiment(ControlExperiment):
    def __init__(self, args):
        super().__init__(args)

    def generate_data(self, n_experts=1):
        expert_data = ExperienceMemoryNumpy(self.args)
        expert_data.reset(buffer_size=expert_data.buffer_size * n_experts)

        for i in range(n_experts):
            self.agent = get_model(self.args, self.env)
            super().train()
            expert_data.extend(self.agent.experience_memory)

        file = open(self.data_path, "wb")
        pkl.dump(expert_data, file)
        file.close()

    def train(self):
        time_start = time.time()

        file = open(self.data_path, "rb")

        self.agent = get_model(self.args, self.env)
        expert_memo = pkl.load(file)
        self.agent.experience_memory.clone(expert_memo)
        self.agent.learn(max_iter=self.args.max_iter)

        self.agent.logger.save(self.agent, None, None, self.n_total_steps)

        time_end = time.time()
        self.agent.logger.log(f"Training time: {time_end - time_start:.2f} seconds")

    def test(self, n_trials=5):
        self.args.max_iter = 0
        self.args.max_episodes = n_trials
        super().train()
