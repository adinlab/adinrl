import torch
from torch.nn.modules.loss import _Loss
from nets.layers.bayesian_layers import calculate_kl_terms
import math
import torch as th


#####################################################################
class LaplaceMLELoss(_Loss):

    def forward(self, mu, logvar, y):
        b = (2 * logvar.exp()).sqrt().view(-1, 1)  # .clamp(1e-14, None)
        return (torch.log(2.0 * b) + (y - mu).abs() / b).mean()


#####################################################################
class ProbabilisticLoss(_Loss):

    def forward(self, mu, logvar, y):
        pass


#####################################################################
class NormalMLELoss(_Loss):

    def forward(self, mu, logvar, y):
        logvar = logvar  # .clamp(-4.6, 4.6)  # log(-4.6) = 0.01
        var = logvar.exp().clamp(1e-6, 10)
        return (0.5 * logvar + 0.5 * ((mu - y).pow(2)) / var).mean()


#####################################################################
# Analytical loss for a last-layer BNN
class NormalLLLoss(_Loss):
    def forward(self, mu, lvar, y):
        var = lvar.exp()  # .clamp(-4.6, 4.6)
        return ((mu - y).pow(2) + var).mean()


#####################################################################
# Approx McAllester Bound (as used in the PAC4SAC draft)
# TODO: Generalize this to a general McAllester bound
class McAllester(_Loss):
    def forward(self, critic, N, delta=0.05):
        confidence_term = math.log(2.0 * math.sqrt(N) / delta)
        return (
            ((calculate_kl_terms(critic)[0] + confidence_term) / (2 * N)).sqrt().mean()
        )


#####################################################################
# Variational Free Energy Loss
class VFELossTotal(NormalMLELoss):
    def forward(self, mu, logvar, y, net, n_data):
        ENeglogLik = super().forward(mu, logvar, y)
        kl, _ = calculate_kl_terms(net)
        loss = ENeglogLik + kl / n_data
        return loss


#####################################################################
class VFELoss(_Loss):
    def forward(self, ypred, y, net, n_data):
        kl, n_w = calculate_kl_terms(net)
        loss = ((ypred - y) ** 2).mean() + kl / n_data
        return loss


#####################################################################
class LaplaceKLLoss(ProbabilisticLoss):

    def get_kl(self, mu1, b1, mu2, b2):
        mu_diff = (mu1 - mu2).abs()
        return (b1 * torch.exp(-mu_diff / b1) + mu_diff) / b2 + torch.log(b2 / b1)

    def forward(self, mu, logvar, y):
        b = (2 * logvar.exp()).sqrt().view(-1, 1)  # .clamp(1e-14, None)
        mu_t = y[:, 0].view(-1, 1)
        b_t = (2 * y[:, 1]).sqrt().view(-1, 1)  # target mean and scale
        mle = torch.log(2.0 * b) + (mu - mu_t).abs() / b
        kl = self.get_kl(mu, b, mu_t, b_t)
        return (mle * (b_t <= 0) + kl * (b_t > 0)).mean()


#####################################################################
class NormalKLLoss(ProbabilisticLoss):

    def get_kl(self, mu1, var1, mu2, var2):
        return var2.log() - var1.log() + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5

    def forward(self, mu, logvar, y):
        mu_t = y[:, 0].view(-1, 1)
        sig2_t = y[:, 1].view(-1, 1)  # .clamp(1e-14, None)
        sig2 = logvar.exp()  # .clamp(1e-14, None)
        return self.get_kl(mu_t, sig2_t, mu, sig2).mean()


#####################################################################
class NormalReverseKLLoss(NormalKLLoss):

    def forward(self, mu, logvar, y):
        mu_t = y[:, 0].view(-1, 1)
        sig2_t = y[:, 1].view(-1, 1)  # .clamp(1e-14, None)
        sig2 = logvar.exp()  # .clamp(-14, 5)
        return self.get_kl(mu, sig2, mu_t, sig2_t).mean()


#####################################################################
# @torch.jit.script
# def calculate_huber_loss(td_errors: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
#    return torch.where(
#        td_errors.abs() <= kappa,
#        0.5 * td_errors.pow(2),
#        kappa * (td_errors.abs() - 0.5 * kappa))


@torch.compile
def get_w2(mu1, std1, mu2, std2):
    return (mu1 - mu2) ** 2 + (std1 - std2) ** 2
    # return calculate_huber_loss(mu1 - mu2) + calculate_huber_loss(std1 - std2)


class WassersteinLoss(NormalMLELoss):
    def forward(self, mu, logvar, y):
        # shape(y) = n_ens x n_batch x (mu,var)
        mu_t = y[:, :, 0]
        sig_t = y[:, :, 1].sqrt()  # .clamp(1e-2, 1e2)
        sig = th.exp(0.5 * logvar)  # .clamp(1e-2, 1e2)
        # Take the mean per member not over everything
        return get_w2(mu, sig, mu_t, sig_t).mean(1).sum()


class NormalWassersteinLossLVar(NormalMLELoss):
    def forward(self, mu, logvar, y):
        mu_t = y[:, 0].view(-1, 1)
        sig_t = y[:, 1].view(-1, 1).sqrt()  # .clamp(1e-2, 1e2)
        # sig = logvar.clamp(-4.6, None).exp().sqrt()
        sig = th.exp(0.5 * logvar)  # .clamp(1e-2, 1e2)
        return get_w2(mu, sig, mu_t, sig_t).mean()


def calculate_huber_loss(td_errors: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    return torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa),
    )


#####################################################################
class NoisyNormalMLELoss(ProbabilisticLoss):

    def forward(self, mu, logvar, y):
        mu_t = y[:, 0].view(-1, 1)
        sig2_t = y[:, 1].view(-1, 1)
        varf = logvar.exp() + sig2_t  # .clamp(1e-14, None)
        return (varf.log() + ((mu - mu_t) ** 2) / varf).mean()


#####################################################################
class NormalLCBLoss(_Loss):

    def forward(self, mu, logvar, y):
        sig2 = logvar.exp()  # .clamp(1e-14, None)
        mu_t = y[:, 0].view(-1, 1)
        sig2_t = y[:, 1].view(-1, 1)
        y_lcb = mu_t - sig2_t.sqrt() * 1.96
        return (sig2.log() + ((mu - y_lcb) ** 2) / sig2).mean()
