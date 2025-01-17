import torch
import torch.nn as nn


################################################
class Kernel:
    def __init__(self):
        pass

    def k(self, X, Z):
        raise NotImplementedError

    def K(self, X):
        return self.k(X, X) + (1e-2) * torch.eye(X.shape[0]).cuda()

    def kstar(self, X):
        raise NotImplementedError


################################################
class RBFKernel(Kernel):
    def __init__(self):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(1)).cuda()

    def k(self, X, Z):
        sqd = torch.sum((X[:, None, :] - Z[None, :, :]) ** 2, 2)

        return torch.exp(-sqd / torch.exp(self.log_sigma.clamp(-20, 5)))

    def kstar(self, X):
        return torch.ones_like(X[:, 0])  # TODO: add multiplicative/additive noise later


################################################
class CosineSimilarityKernel(Kernel):
    def __init__(self):
        super().__init__()

    def norm(self, X):
        return torch.sum(X**2, axis=1).sqrt()

    def k(self, X, Z):
        return X @ Z.t() / (self.norm(X).view(-1, 1) @ self.norm(Z).view(1, -1))

    def kstar(self, X):
        return torch.ones_like(X[:, 0])  # TODO: add multiplicative/additive noise later
