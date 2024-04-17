import math
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
import math


class VariationalBayesianLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 prior_log_sig2 =0) -> None:
        super(VariationalBayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_log_sig2 = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_mu_prior = nn.Parameter(torch.zeros((out_features, in_features)), requires_grad=False)
        self.weight_log_sig2_prior = nn.Parameter(prior_log_sig2*torch.zeros((out_features, in_features)), requires_grad=False)
        if self.has_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_log_sig2 = nn.Parameter(torch.Tensor(out_features))
            self.bias_mu_prior = nn.Parameter(torch.zeros(out_features), requires_grad=False)
            self.bias_log_sig2_prior = nn.Parameter(prior_log_sig2*torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sig2', None)
        self.reset_parameters(prior_log_sig2=prior_log_sig2)

    def reset_parameters(self, prior_log_sig2: float) -> None:
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(self.weight_mu.shape[1]))
        init.constant_(self.weight_log_sig2, -10)      
        if self.has_bias:
            init.zeros_(self.bias_mu)
            init.constant_(self.bias_log_sig2, -10)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output_mu = F.linear(input, self.weight_mu, self.bias_mu)
        output_sig2 = F.linear(input.pow(2), self.weight_log_sig2.exp(), self.bias_log_sig2.exp())
        return output_mu + output_sig2.sqrt() * torch.randn_like(output_sig2) #,output_mu,output_sig2

    def get_mean_var(self, input: torch.Tensor) -> torch.Tensor:
        mu = F.linear(input, self.weight_mu, self.bias_mu)
        sig2 = F.linear(input.pow(2), self.weight_log_sig2.exp(), self.bias_log_sig2.exp())
        return mu, sig2
        
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.has_bias
        )

    def update_prior(self, newprior):
        self.weight_mu_prior.data = newprior.weight_mu.data.clone()
        self.weight_mu_prior.data.requires_grad = False
        self.weight_log_sig2_prior.data = newprior.weight_log_sig2.data.clone()
        self.weight_log_sig2_prior.data.requires_grad = False
        if self.has_bias:
            self.bias_mu_prior.data = newprior.bias_mu.data.clone()
            self.bias_mu_prior.data.requires_grad = False
            self.bias_log_sig2_prior.data = newprior.bias_log_sig2.data.clone()
            self.bias_log_sig2_prior.data.requires_grad = False           

    def kl_loss(self):
        kl_weight = 0.5 * (self.weight_log_sig2_prior - self.weight_log_sig2 + (self.weight_log_sig2.exp() + (self.weight_mu_prior-self.weight_mu)**2) / self.weight_log_sig2_prior.exp() - 1.0)
        kl = kl_weight.sum()
        n = len(self.weight_mu.view(-1))
        if self.has_bias:
            kl_bias = 0.5 * (self.bias_log_sig2_prior - self.bias_log_sig2 + (self.bias_log_sig2.exp() + (self.bias_mu_prior-self.bias_mu)**2) / (self.bias_log_sig2_prior.exp()) - 1.0 )
            kl += kl_bias.sum()
            n += len(self.bias_mu.view(-1))
        return kl, n

def update_prior_bnn(model: nn.Module, newprior: nn.Module):
    
    """ Function to update priors of bayesian neural network """
    curmodel = list(model.children())
    newmodel = list(newprior.children())

    # iterate over the nn.Module layers
    for i in range(len(curmodel)):
        if curmodel[i].__class__.__name__.startswith('Variational'):
            curmodel[i].update_prior(newmodel[i])

def calculate_kl_terms(model: nn.Module):
    """ Function to calculate KL loss of bayesian neural network """
    kl, n = 0, int(0)
    for m in model.modules():
        if m.__class__.__name__.startswith('Variational'):
            kl_, n_ = m.kl_loss()
            kl += kl_
            n += n_
        if m.__class__.__name__.startswith('CLTLayer'):
            kl_, n_ = m.KL()
            kl += kl_
            n += n_
    return kl, n

