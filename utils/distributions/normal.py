import math

import torch as th


def pdf_normal(x, mean, var):
    return th.exp(-0.5 * (x - mean).pow(2) / var) / math.sqrt(2 * math.pi)


def cdf_normal(x, mean, var):
    return 0.5 * (1 + th.erf((x - mean) / th.sqrt(2 * var)))


def cdfinv_standardnormal(p):
    return math.sqrt(2) * th.erfinv(2 * p - 1)


def pdf_truncated_normal(x, mean, var, low, high):
    return (
        1
        / var.sqrt()
        * pdf_normal(x, mean, var)
        / (cdf_normal(high, mean, var) - cdf_normal(low, mean, var))
    )


def cdf_truncated_normal(x, mean, var, low, high):
    return (cdf_normal(x, mean, var) - cdf_normal(low, mean, var)) / (
        cdf_normal(high, mean, var) - cdf_normal(low, mean, var)
    )


def mean_truncated_normal(mean, var, low, high):
    return (
        mean
        + (pdf_normal(low, mean, var) - pdf_normal(high, mean, var))
        / (cdf_normal(high, mean, var) - cdf_normal(low, mean, var))
        * var.sqrt()
    )


def var_truncated_normal(mean, var, low, high):
    Z = cdf_normal(high, mean, var) - cdf_normal(low, mean, var)
    phi_alpha = pdf_normal(low, mean, var)
    phi_beta = pdf_normal(high, mean, var)
    return var * (
        1
        - (high * phi_beta - low * phi_alpha) / Z
        - ((phi_alpha - phi_beta) / Z).pow(2)
    )


def sample_exact_truncated_normal(mean, var, low, high):
    U = th.rand_like(mean)
    cdf_alpha = cdf_normal(low, mean, var)
    cdf_beta = cdf_normal(high, mean, var)
    return (
        cdfinv_standardnormal(cdf_alpha + U * (cdf_beta - cdf_alpha)) * var.sqrt()
        + mean
    )
