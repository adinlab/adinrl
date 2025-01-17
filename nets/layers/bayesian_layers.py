import math

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class BayesianLinearEM(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_log_sig2 = nn.Parameter(torch.empty((out_features, in_features)))
        self.log_noise_var = nn.Parameter(torch.empty(1))
        init.ones_(self.log_noise_var)
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(self.weight_mu.shape[1]))
        init.constant_(self.weight_log_sig2, -10)

        if self.has_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            init.zeros_(self.bias_mu)

    def get_mean_var(self, input: torch.Tensor) -> torch.Tensor:
        # M-step
        mu = F.linear(input, self.weight_mu, self.bias_mu)
        sig2 = F.linear(input.pow(2), self.weight_log_sig2.exp())

        return mu, sig2 + self.log_noise_var.exp()  # .clamp(-10, None)

    # def e_step(self,input: torch.Tensor, target: torch.Tensor):
    #    # E-step
    #    input_sum = (input ** 2).sum(dim=0)
    #    self.weight_log_sig2 = (-(input_sum/self.log_noise_var.clamp(-10,None).exp() + 1/self.weight_log_sig2.exp()).log()).detach()
    #    self.weight_log_sig2 = self.weight_log_sig2.clamp(-10,None)
    #    input_mat = self.weight_log_sig2 * input / self.log_noise_var.clamp(-10,None).exp()
    #    self.weight_mu = ( torch.transpose( input_mat @ target, 0, 1) ).detach()


class VariationalBayesianLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        prior_log_sig2=0,
        log_sig2_init=-4.6,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_log_sig2 = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_mu_prior = nn.Parameter(
            torch.zeros((out_features, in_features)), requires_grad=False
        )
        self.weight_log_sig2_prior = nn.Parameter(
            prior_log_sig2 * torch.ones((out_features, in_features)),
            requires_grad=False,
        )
        if self.has_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias_mu", None)
        self.reset_parameters(log_sig2_init)

    def reset_parameters(self, log_sig2_init) -> None:
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(self.weight_mu.shape[1]))
        init.constant_(self.weight_log_sig2, log_sig2_init)
        if self.has_bias:
            init.zeros_(self.bias_mu)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output_mu = F.linear(input, self.weight_mu, self.bias_mu)
        output_sig2 = F.linear(
            input.pow(2), self.weight_log_sig2.exp(), bias=None  # .clamp(-10, 10)
        )

        return output_mu + output_sig2.sqrt() * torch.randn_like(output_sig2)

    def get_mean_var(self, input: torch.Tensor) -> torch.Tensor:
        mu = F.linear(input, self.weight_mu, self.bias_mu)
        sig2 = F.linear(input**2, self.weight_log_sig2.exp(), bias=None)
        return mu, sig2

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.has_bias
        )

    def update_prior(self, newprior):
        self.weight_mu_prior.data = newprior.weight_mu.data.clone()
        self.weight_mu_prior.data.requires_grad = False
        self.weight_log_sig2_prior.data = newprior.weight_log_sig2.data.clone()
        self.weight_log_sig2_prior.data.requires_grad = False
        # if self.has_bias:
        #    self.bias_mu_prior.data = newprior.bias_mu.data.clone()
        #    self.bias_mu_prior.data.requires_grad = False
        #    self.bias_log_sig2_prior.data = newprior.bias_log_sig2.data.clone()
        #    self.bias_log_sig2_prior.data.requires_grad = False

    def kl_loss(self):
        kl_weight = 0.5 * (
            self.weight_log_sig2_prior
            - self.weight_log_sig2
            + (
                self.weight_log_sig2.exp()
                + (self.weight_mu_prior - self.weight_mu) ** 2
            )
            / self.weight_log_sig2_prior.exp()
            - 1.0
        )
        kl = kl_weight.sum()
        n = len(self.weight_mu.view(-1))
        # if self.has_bias:
        #    kl_bias = 0.5 * (
        #        self.bias_log_sig2_prior
        #        - self.bias_log_sig2
        #        + (self.bias_log_sig2.exp() + (self.bias_mu_prior - self.bias_mu) ** 2)
        #        / (self.bias_log_sig2_prior.exp())
        #        - 1.0
        #    )
        #    kl += kl_bias.sum()
        #    n += len(self.bias_mu.view(-1))
        return kl, n


def update_prior_bnn(model: nn.Module, newprior: nn.Module):
    """Function to update priors of bayesian neural network"""
    curmodel = list(model.children())
    newmodel = list(newprior.children())

    # iterate over the nn.Module layers
    for i in range(len(curmodel)):
        if curmodel[i].__class__.__name__.startswith("Variational"):
            curmodel[i].update_prior(newmodel[i])


def calculate_kl_terms(model: nn.Module):
    """Function to calculate KL loss of bayesian neural network"""
    kl, n = 0, int(0)
    for m in model.modules():
        if m.__class__.__name__.startswith("Variational"):
            kl_, n_ = m.kl_loss()
            kl += kl_
            n += n_
        if m.__class__.__name__.startswith("CLTLayer"):
            kl_, n_ = m.KL()
            kl += kl_
            n += n_
    return kl, n


def normal_cdf(x, mu=0.0, sig=1.0):
    return 0.5 * (1 + torch.erf((x - mu) / (sig * math.sqrt(2))))


def normal_pdf(x, mu=0.0, sig=1.0):
    return (1 / (math.sqrt(2 * math.pi) * sig)) * torch.exp(
        -0.5 * ((x - mu) / sig).pow(2)
    )


## CLT Layers
class CLTLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        alpha=10,
        act="relu",
        isinput=False,
        isoutput=False,
    ):
        super().__init__()
        self.n_in = in_features
        self.n_out = out_features
        self.isoutput = isoutput
        self.isinput = isinput
        self.alpha = alpha
        assert act in ["relu", "crelu"]
        self.act = act

        self.Mbias = nn.Parameter(torch.Tensor(out_features))

        self.M = Parameter(torch.Tensor(out_features, in_features))
        self.logS = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.M.size(1))
        self.M.data.normal_(0, stdv)
        self.logS.data.zero_().normal_(-9, 0.001)
        self.Mbias.data.zero_()

    def KL(self):
        logS = self.logS.clamp(-11, 11)
        kl = 0.5 * (self.alpha * (self.M.pow(2) + logS.exp()) - logS).sum()
        return kl, 0

    def relu_moments(self, mu, sig):
        # compute mean and variance of relu(x)
        alpha = mu / sig
        cdf = normal_cdf(alpha)
        pdf = normal_pdf(alpha)
        relu_mean = mu * cdf + sig * pdf
        relu_var = (sig.pow(2) + mu.pow(2)) * cdf + mu * sig * pdf - relu_mean.pow(2)
        relu_mean[sig.eq(0)] = mu[sig.eq(0)] * (mu[sig.eq(0)] > 0)
        relu_var[sig.eq(0)] = 0.0
        return relu_mean, relu_var

    def neg_relu_moments(self, mu, sig):
        # compute mean and variance of relu(-x)
        alpha = mu / sig
        cdf = normal_cdf(alpha)
        pdf = normal_pdf(alpha)
        neg_relu_mean = mu * cdf + sig * pdf - mu
        neg_relu_var = (
            (mu.pow(2) + sig.pow(2)) * (1 - cdf) - mu * sig * pdf - neg_relu_mean.pow(2)
        )
        neg_relu_mean[sig.eq(0)] = -mu[sig.eq(0)] * (mu[sig.eq(0)] < 0)
        neg_relu_var[sig.eq(0)] = 0.0
        return neg_relu_mean, neg_relu_var

    def crelu_moments(self, mu, sig):
        # [relu(x), relu(-x)]
        relu_mean, relu_var = self.relu_moments(mu, sig)
        neg_relu_mean, neg_relu_var = self.neg_relu_moments(mu, sig)
        return th.cat((relu_mean, neg_relu_mean), -1), th.cat(
            (relu_var, neg_relu_var), -1
        )

    def forward(self, mu_h, var_h=None):
        if var_h is None:
            assert (
                self.isinput
            ), "We do not allow determinstic input to non-input layers."

        M = self.M
        var_s = self.logS.clamp(-11, 11).exp()

        mu_f = F.linear(mu_h, M, self.Mbias)
        # No input variance
        if self.isinput:
            var_f = F.linear(mu_h**2, var_s)
        else:
            var_f = F.linear(var_h + mu_h.pow(2), var_s) + F.linear(var_h, M.pow(2))

        # compute relu moments if it is not an output layer
        if not self.isoutput:
            if self.act == "relu":
                return self.relu_moments(mu_f, var_f.sqrt())
            elif self.act == "crelu":
                return self.crelu_moments(mu_f, var_f.sqrt())

        else:
            return mu_f, var_f

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_in)
            + " -> "
            + str(self.n_out)
            + f", isinput={self.isinput}, isoutput={self.isoutput})"
        )


class CLTLayerDet(nn.Module):
    def __init__(self, in_features, out_features, act="relu", isoutput=False):
        super().__init__()
        self.n_in = in_features
        self.n_out = out_features
        self.isoutput = isoutput
        assert act in ["relu", "crelu"]
        self.act = act
        self.EPS = 1e-8

        self.mean_layer = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        self.mean_layer.reset_parameters()

    def relu_moments(self, mu, var):
        zeros = var.eq(0)
        sig = var.clamp(self.EPS).sqrt()
        alpha = mu / sig.clamp(self.EPS)

        cdf = normal_cdf(alpha)
        pdf = normal_pdf(alpha)
        relu_mean = mu * cdf + sig * pdf
        relu_var = (sig.pow(2) + mu.pow(2)) * cdf + mu * sig * pdf - relu_mean.pow(2)
        relu_var = relu_var.clamp(self.EPS)

        # If the variance is zero, we get a simple relu mean
        relu_mean = mu * (mu > 0) * zeros + relu_mean * ~zeros
        relu_var = th.zeros_like(relu_var) * zeros + relu_var * ~zeros
        return relu_mean, relu_var

    def neg_relu_moments(self, mu, sig):
        # compute mean and variance of relu(-x)
        alpha = mu / sig
        cdf = normal_cdf(alpha)
        pdf = normal_pdf(alpha)
        neg_relu_mean = mu * cdf + sig * pdf - mu
        neg_relu_var = (
            (mu.pow(2) + sig.pow(2)) * (1 - cdf) - mu * sig * pdf - neg_relu_mean.pow(2)
        )
        neg_relu_mean[sig.eq(0)] = -mu[sig.eq(0)] * (mu[sig.eq(0)] < 0)
        neg_relu_var[sig.eq(0)] = 0.0
        return neg_relu_mean, neg_relu_var

    def crelu_moments(self, mu, var):
        # [relu(x), relu(-x)]
        relu_mean, relu_var = self.relu_moments(mu, var)
        neg_relu_mean, neg_relu_var = self.neg_relu_moments(mu, var.sqrt())
        return th.cat((relu_mean, neg_relu_mean), -1), th.cat(
            (relu_var, neg_relu_var), -1
        )

    def forward(self, mu_h, var_h=None):
        if var_h is None:
            var_h = th.zeros_like(mu_h)

        mu_f = self.mean_layer(mu_h)
        var_f = F.linear(var_h, self.mean_layer.weight.pow(2))

        # compute relu moments if it is not an output layer
        if not self.isoutput:
            if self.act == "relu":
                return self.relu_moments(mu_f, var_f)
            elif self.act == "crelu":
                return self.crelu_moments(mu_f, var_f)
        else:
            return mu_f, var_f

    def __repr__(self):
        return (
            self.__class__.__name__
            + f" ({str(self.n_in)} -> {self.n_out}, isoutput={self.isoutput})"
        )


class ConvCLTLayer(CLTLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        alpha=10,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        isinput=False,
        isoutput=False,
    ):
        super().__init__()
        self.n_in = in_channels
        self.n_out = out_channels

        self.isinput = isinput
        self.isoutput = isoutput
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.alpha = alpha
        self.normal = True

        self.M = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.logS = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.Mbias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.n_in
        for k in range(1, self.kernel_size):
            n *= k
        self.M.data.normal_(0, 1.0 / math.sqrt(n))
        self.logS.data.zero_().normal_(-9, 0.001)
        self.Mbias.data.zero_()

    def forward(self, mu_h, var_h):
        if var_h is None:
            assert (
                self.isinput
            ), "We do not allow determinstic input to non-input layers."

        var_s = self.logS.clamp(-11, 11).exp()
        mu_f = F.conv2d(
            mu_h,
            self.M,
            self.Mbias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.isinput:
            var_f = F.conv2d(
                mu_h**2,
                var_s,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        else:
            var_f = F.conv2d(
                var_h + mu_h.pow(2),
                var_s,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            var_f += F.conv2d(
                var_h,
                self.M.pow(2),
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        if self.isoutput:
            return mu_f, var_f
        else:
            return self.relu_moments(mu_f, var_f.sqrt())

    def __repr__(self):
        s = "{name}({n_in}, {n_out}, kernel_size={kernel_size}" ", stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        s += ", isinput={isinput}"
        s += ")"

        return s.format(name=self.__class__.__name__, **self.__dict__)


class LeNet5Closed(nn.Module):
    def __init__(self, n_dim=1, n_classes=10, large=False, mode="maxent"):
        super().__init__()
        self.varfactor = 1.0
        self.beta = 1.0
        self.mode = mode
        if large:
            self.latdim = 4 * 4 * 192 if n_dim == 1 else 5 * 5 * 192
            self.conv1 = ConvCLTLayer(n_dim, 192, 5, stride=2, isinput=True)
            self.conv2 = ConvCLTLayer(192, 192, 5, stride=2)
            self.dense1 = CLTLayer(self.latdim, 1000)
            self.dense2 = CLTLayer(1000, n_classes, isoutput=True)
        else:
            self.latdim = 4 * 4 * 50 if n_dim == 1 else 5 * 5 * 50
            self.conv1 = ConvCLTLayer(n_dim, 20, 5, stride=2, isinput=True)
            self.conv2 = ConvCLTLayer(20, 50, 5, stride=2)
            self.dense1 = CLTLayer(self.latdim, 500)
            self.dense2 = CLTLayer(500, n_classes, isoutput=True)

    def reset_params(self):
        for l in [self.conv1, self.conv2, self.dense1, self.dense2]:
            l.reset_parameters()

    def forward(self, input):
        mu_h1, var_h1 = self.conv1(input, None)
        mu_h2, var_h2 = self.conv2(mu_h1, var_h1)

        mu_h2 = mu_h2.view(-1, self.latdim)
        var_h2 = var_h2.view(-1, self.latdim)

        mu_h3, var_h3 = self.dense1(mu_h2, var_h2)
        mu_pred, var_pred = self.dense2(mu_h3, var_h3)

        return mu_pred, var_pred

    def loss(self, data, target, N):
        mu_pred, var_pred = self.forward(data)
        KLsum = self.dense1.KL() + self.dense2.KL() + self.conv1.KL() + self.conv2.KL()

        return ProbitLoss_var(mu_pred, var_pred, target) + (KLsum / N)
