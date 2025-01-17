import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math


################################################
class SparseKernelMachine(nn.Module):
    def __init__(self, kernel, n_inducing, n_input, n_output):
        super().__init__()
        self.n_in = n_input
        self.n_out = n_output
        self.n_z = n_inducing
        self.Z = nn.Parameter(th.randn(self.n_z, self.n_in))  # inducing input set
        self.kernel = kernel  # the kernel function


################################################
class GaussianProcessLayer(SparseKernelMachine):
    def __init__(self, kernel, n_inducing, n_input, n_output):
        super().__init__(kernel, n_inducing, n_input, n_output)
        self.u = nn.Parameter(th.randn(self.n_z, self.n_out))  # inducing output set
        self.v = nn.Parameter(th.randn(1, self.n_out))  # cross covariance (Kronecker)

    def sample(self, x):
        mu, sigma = self.get_mean_var(x)
        return mu + sigma.sqrt() * th.randn_like(mu)

    def forward(self, x):
        mu, sigma = self.get_mean_var(x)
        return mu

    def get_mean_var(self, x):
        self.K = self.kernel.K(self.Z)
        Kzz_inv = th.linalg.inv(self.K)
        kxz = self.kernel.k(x, self.Z)
        kxzZ = kxz @ Kzz_inv
        mu = kxzZ @ self.u
        Sig = (self.kernel.kstar(x) - th.sum(kxzZ * kxz, axis=1)).sqrt().view(-1, 1)
        return mu, Sig

    def get_kl(self):
        return (th.trace(self.K) - th.logdet(self.K)) * 0.5


################################################
class GaussianLayer(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.mu = nn.Parameter(th.empty(n_output, n_input))
        self.bias = nn.Parameter(th.empty(n_output))
        self.log_sig = nn.Parameter(
            th.empty(n_output, n_input)
        )  # LL^T gives covariance
        self.n_in = n_input
        self.n_out = n_output
        init.kaiming_uniform_(self.mu, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.mu)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)
        init.constant_(self.log_sig, -10)

    def get_kl(self):
        kl = 0.5 * (
            -self.log_sig + (self.log_sig.clamp(-8, 8).exp() + (self.mu) ** 2) - 1.0
        )
        return kl.mean()

    def sample(self, x):
        mu, sigma = self.get_mean_var(x)
        return mu + sigma.sqrt() * th.randn_like(mu)

    def forward(self, x):
        mu, sigma = self.get_mean_var(x)
        return mu

    def get_mean_var(self, x):
        x_mu = F.linear(x, self.mu, self.bias)
        x_sigma = F.linear(x**2, self.log_sig.clamp(-8, 8).exp(), None)
        return x_mu, x_sigma


################################################
class RelevanceVectorMachineLayer(SparseKernelMachine):
    def __init__(self, kernel, n_inducing, n_input, n_output):
        super().__init__(kernel, n_inducing, n_input, n_output)
        self.log_a = nn.Parameter(
            th.randn(self.n_z, self.n_out)
        )  # relevance vector coefficients

    def forward(self, x):
        kxz = self.kernel.k(x, self.Z)
        W = self.log_a  # th.randn(self.n_z, self.n_out) * th.exp(self.log_a).sqrt()
        out = kxz @ W
        return out


################################################
class AttentiveKernelMachineLayer(SparseKernelMachine):
    def __init__(self, kernel, n_inducing, n_input, n_output):
        super().__init__(kernel, n_inducing, n_input, n_output)
        self.n_key = 20
        self.W_key = nn.Parameter(
            th.randn(self.n_out, self.n_key, self.n_in)
        )  # attention head

    def attend(self, x):
        # Replace below with einsum to gain speed
        Q = th.bmm(self.W_key, x.t().repeat(self.n_out, 1, 1))
        K = th.bmm(self.W_key, self.Z.t().repeat(self.n_out, 1, 1))
        V = th.bmm(Q.swapaxes(1, 2), K)
        out = F.gumbel_softmax(V, dim=2)
        return out

    def forward(self, x):
        kxz = self.kernel.k(x, self.Z)
        A = self.attend(x)
        out = th.sum(kxz * A, dim=2).t()
        return out
