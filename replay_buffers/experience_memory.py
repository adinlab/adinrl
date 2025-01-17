import copy
import os

import numpy as np
import torch
import torch as th

from utils.deprecation import deprecated


class BasicMemoryBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, args):
        self.device = args.device
        self.buffer_size = args.buffer_size
        self.dims = args.dims
        self.field_names = (
            tuple(args.field_names) if hasattr(args, "field_names") else ["default"]
        )
        self.reset()

    def reset(self, buffer_size=None):
        if buffer_size is not None:
            self.buffer_size = buffer_size
        self.data_size = 0
        self.pointer = 0
        self.batches = []
        self.memory = {
            field: th.empty(self.dims[field], device=self.device)
            for field in self.field_names
            if field in self.dims.keys()
        }

    def add(self, state, action, reward, next_state, terminated, step):
        for field, value in zip(
            self.field_names, [state, action, reward, next_state, terminated, step]
        ):
            self.memory[field][self.pointer] = value
        self.pointer = (self.pointer + 1) % self.buffer_size
        self.data_size = min(self.data_size + 1, self.buffer_size)

    def sample_by_index(self, index):
        return tuple(self.memory[field][index] for field in self.field_names)

    def sample_by_index_fields(self, index, fields):
        if len(fields) == 1:
            return self.memory[fields[0]][index]  # return a tensor
        return tuple(self.memory[field][index] for field in fields)

    def sample_random(self, batch_size):
        index = th.randint(self.data_size, (batch_size,))
        return self.sample_by_index(index)

    def sample_batch(self, batch_size):
        if not self.batches:
            self.shuffle(batch_size)
        index = self.batches.pop()
        return self.sample_by_index(index)

    @staticmethod
    def set_diff_1d(t1, t2, assume_unique=False):
        """
        Set difference of two 1D tensors.
        Returns the unique values in t1 that are not in t2.
        Source: https://stackoverflow.com/questions/55110047/finding-non-intersection-of-two-pytorch-tensors/72898627#72898627
        """
        if not assume_unique:
            t1 = torch.unique(t1)
            t2 = torch.unique(t2)
        return t1[(t1[:, None] != t2).all(dim=1)]

    def filter_by_nonterminal_steps_with_horizon(self, horizon):
        all_indices = th.arange(self.data_size - horizon + 1)
        terminal_indices = th.argwhere(self.memory["terminated"] == True)
        if terminal_indices.size == 0:
            return all_indices

        terminal_with_horizon_indices = th.tensor(
            [
                th.arange(terminal - horizon + 2, terminal + 1)
                for terminal in terminal_indices
            ]
        ).flatten()
        nonterminal_indices = th.setdiff1d(all_indices, terminal_with_horizon_indices)
        return nonterminal_indices

    def sample_random_sequence_snippet(self, batch_size, sequence_length):
        non_terminal_indices = self.filter_by_nonterminal_steps_with_horizon(
            sequence_length
        )
        indices = th.randint(non_terminal_indices, (batch_size,))
        output = []
        # TODO: Why this loop?
        for i in range(sequence_length):
            output.append(self.sample_by_index(indices + i))
        return output

    def sample_all(self):
        return self.sample_by_index(range(self.data_size))

    def clone(self, other_memory):
        self.data_size = other_memory.data_size
        self.memory = copy.deepcopy(other_memory.memory)

    def extend(self, other_memory):
        for field in self.field_names:
            self.memory[field].extend(other_memory.memory[field])
        self.data_size = len(self.memory[field])

    def __len__(self):
        return self.data_size

    @property
    def size(self):
        return self.data_size

    def save(self, path):
        th.save(self.memory, os.path.join(path, "experience_memory.pt"))
        # read with np.load("path", allow_pickle=True).item() -> dict = memory

    def get_last_observation(self):
        return self.sample_by_index([-1])

    def get_last_observations(self, batch_size):
        return self.sample_by_index(range(-batch_size, 0))

    def shuffle(self, batch_size, drop_last=True):
        indices = th.randperm(self.data_size)
        self.batches = list(th.split(indices, batch_size))
        if drop_last:
            if len(self.batches[-1]) != batch_size:
                self.batches.pop()


# Just for backward compatibility for now
class ExperienceMemoryTorch(BasicMemoryBuffer):
    def __init__(self, args):
        super().__init__(args)
        self.field_names = [
            "state",
            "action",
            "reward",
            "next_state",
            "terminated",
            "step",
        ]
        self.reset()


@deprecated  # Use `ExperienceMemoryNumpy instead
class ExperienceMemoryNumpy:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, args):
        self.device = args.device
        self.buffer_size = args.buffer_size
        self.field_names = [
            "state",
            "action",
            "reward",
            "next_state",
            "terminated",
            "step",
        ]
        self.dims = args.dims
        self.reset()

    def reset(self, buffer_size=None):
        if buffer_size is not None:
            self.buffer_size = buffer_size
        self.data_size = 0
        self.pointer = 0
        self.memory = {field: np.empty(self.dims[field]) for field in self.field_names}

    def add(self, state, action, reward, next_state, terminated, step):
        for field, value in zip(
            self.field_names, [state, action, reward, next_state, terminated, step]
        ):
            self.memory[field][self.pointer] = value
        self.pointer = (self.pointer + 1) % self.buffer_size
        self.data_size = min(self.data_size + 1, self.buffer_size)

    def sample_by_index(self, index):
        return tuple(
            torch.from_numpy(self.memory[field][index]).to(self.device).float()
            for field in self.field_names
        )

    def sample_by_index_fields(self, index, fields):
        if len(fields) == 1:
            return (
                torch.from_numpy(self.memory[fields[0]][index]).to(self.device).float()
            )  # return a tensor
        return tuple(
            torch.from_numpy(self.memory[field][index]).to(self.device).float()
            for field in fields
        )

    def sample_random(self, batch_size):
        index = np.random.choice(self.data_size, batch_size)
        return self.sample_by_index(index)

    def filter_by_nonterminal_steps_with_horizon(self, horizon):
        all_indices = np.arange(self.data_size - horizon + 1)
        terminal_indices = np.argwhere(self.memory["terminated"] == True)
        if terminal_indices.size == 0:
            return all_indices

        terminal_with_horizon_indices = np.array(
            [
                np.arange(terminal - horizon + 2, terminal + 1)
                for terminal in terminal_indices
            ]
        ).flatten()
        nonterminal_indices = np.setdiff1d(all_indices, terminal_with_horizon_indices)
        return nonterminal_indices

    def sample_random_sequence_snippet(self, batch_size, sequence_length):
        non_terminal_indices = self.filter_by_nonterminal_steps_with_horizon(
            sequence_length
        )
        indices = np.random.choice(non_terminal_indices, size=batch_size)
        output = []
        for i in range(sequence_length):
            output.append(self.sample_by_index(indices + i))
        return output

    def sample_all(self):
        return self.sample_by_index(range(self.data_size))

    def clone(self, other_memory):
        self.data_size = other_memory.data_size
        self.memory = copy.deepcopy(other_memory.memory)

    def extend(self, other_memory):
        for field in self.field_names:
            self.memory[field].extend(other_memory.memory[field])
        self.data_size = len(self.memory[field])

    def __len__(self):
        return self.data_size

    @property
    def size(self):
        return self.data_size

    def save(self, path):
        np.save(os.path.join(path, "experience_memory.npy"), self.memory)
        # read with np.load("path", allow_pickle=True).item() -> dict = memory

    def get_last_observation(self):
        return self.sample_by_index([-1])
        # return self.memory["state"][-1], self.memory["action"][-1], self.memory["reward"][-1], self.memory["next_state"][-1]

    def get_last_observations(self, batch_size):
        return self.sample_by_index(range(-batch_size, 0))
