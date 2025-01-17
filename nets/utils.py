import itertools

import torch
from torch import nn as nn
from torch.nn import functional as F


class CReLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.cat((x, -x), -1)
        return F.relu(x)


def create_net(d_in, d_out, depth, width, act="crelu", has_norm=True, n_elements=1):
    assert depth > 0, "Need at least one layer"

    double_width = False
    if act == "crelu":
        act = CReLU
        double_width = True
    elif act == "relu":
        act = nn.ReLU
    else:
        raise NotImplementedError(f"{act} is not implemented")

    if depth == 1:
        arch = nn.Linear(d_in, d_out)
    elif depth == 2:
        arch = nn.Sequential(
            nn.Linear(d_in, width),
            (
                nn.LayerNorm(width, elementwise_affine=False)
                if has_norm
                else nn.Identity()
            ),
            act(),
            nn.Linear(2 * width if double_width else width, d_out),
        )
    else:
        in_layer = nn.Linear(d_in, width)
        if n_elements > 1:
            out_layer = nn.Linear(
                2 * width if double_width else width, d_out, n_elements
            )
        else:
            out_layer = nn.Linear(2 * width if double_width else width, d_out)

        # This can probably be done in a more readable way, but it's fast and works...
        hidden = list(
            itertools.chain.from_iterable(
                [
                    [
                        (
                            nn.LayerNorm(width, elementwise_affine=False)
                            if has_norm
                            else nn.Identity()
                        ),
                        act(),
                        nn.Linear(2 * width if double_width else width, width),
                    ]
                    for _ in range(depth - 1)
                ]
            )
        )[:-1]
        arch = nn.Sequential(in_layer, *hidden, out_layer)

    return arch
