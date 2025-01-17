from abc import ABC, abstractmethod

import torch


class AbstractProblem(torch.nn.Module, ABC):
    def __init__(
        self,
    ):
        super().__init__()
        self.method = "CG"
        self._jitter = 1e-3
        self._rtol = 1e-2

    def forward(self, u):
        """Evaluate some random input"""
        pass

    @abstractmethod
    def solve(self):
        """Solve for optimal solution"""
        pass


class Argmin(torch.nn.Module):
    """Wrapping up the Actual argmin code in some PyTorch.nn Functionality"""

    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def forward(self, x=None):
        return Argmin_Function.apply(self.problem, x, *self.problem.parameters())


class Argmin_Function(torch.autograd.Function):
    """Providing backward via IFT
    f = problem instance
    x = optional inputs
    *args = f.parameters()
    """

    @staticmethod
    def forward(ctx, f, x, *args):
        # solve functionality is supplied by problem.
        # this function is critical when descent algorithm is called
        # --> work only with local copies of variables, untracked history!
        # no backward calls!! they are error prone
        u = f.solve(x)
        # store information in ctx for backward.
        # Detach everything in order avoid unwanted behaviour (e.g. extra grads)
        # all other tensors or python objects can be saved in the ctx as ctx.in1 = int1,
        # see https://discuss.pytorch.org/t/how-to-save-a-list-of-integers-for-backward-when-using-cpp-custom-layer/25483
        if x is not None:
            ctx.x = x.detach().clone().requires_grad_(x.requires_grad)
        else:
            ctx.x = None
        ctx.y_star = u.view(-1).detach().clone().requires_grad_(True)
        ctx.f = f

        return u  # this is y_star

    @staticmethod
    def _get_aug_grad_naive(D_y, y_star, grad_output):
        dim = grad_output.shape[0]

        D_yy = []
        for c in range(dim):
            # this loop is the bottleneck
            # maybe distributed pytorch can help here
            torch._C.set_grad_enabled(True)
            D_yy.append(torch.autograd.grad(D_y[c], y_star, retain_graph=True)[0])

        D_yy = torch.cat(D_yy).view(dim, dim)
        D_yy_inv = torch.inverse(D_yy)
        aug_grad = -grad_output @ D_yy_inv
        return aug_grad

    @staticmethod
    def _get_aug_grad_CG(D_y, y_star, grad_output, jitter, r_tol):
        x_cg = grad_output.clone().detach()
        x_cg = x_cg * torch.randn(grad_output.shape)  # x0= diagonal rnadom hessian
        # jitter = .01
        # r_tol = 1e-2

        r = grad_output - (
            torch.autograd.grad(D_y, y_star, x_cg, retain_graph=True)[0] + jitter * x_cg
        )
        p = r.clone()
        rs_old = (r**2).sum()
        num_elements = r.shape[0]

        for k in range(num_elements):
            Ap = torch.autograd.grad(D_y, y_star, p, retain_graph=True)[0] + jitter * p
            alpha = rs_old / (Ap * p).sum()
            x_cg = x_cg + alpha * p
            r = r - alpha * Ap
            rs_new = (r**2).sum()
            if rs_new / num_elements < r_tol:
                break
            beta = rs_new / rs_old
            p = r + beta * p
            rs_old = rs_new
        return -x_cg

    @staticmethod
    def backward(ctx, grad_output):
        # unpack stored information from forward call
        y_star = ctx.y_star
        x = ctx.x
        f = ctx.f
        # flatten grad_output
        grad_output = grad_output.view(-1)

        # problem input dependent?
        x_grad = False
        if x is not None:
            if x.requires_grad:
                x_grad = True

        # note that _C.enable_grad is necessary in between every grad call,
        # autograd behaves in for loops and list comprehensions not accordingly
        # to the with statement (documentation is actually wrong in pytorch)
        with torch.enable_grad():  # direct access C code
            f_eval = f(y_star, x)
            D_y, *_ = torch.autograd.grad(
                f_eval, y_star, create_graph=True, retain_graph=True
            )
            # calc grad_output@Hessian_inverse
            if f.method == "naive":
                aug_grad = Argmin_Function._get_aug_grad_naive(D_y, y_star, grad_output)
            elif f.method == "CG":
                aug_grad = Argmin_Function._get_aug_grad_CG(
                    D_y, y_star, grad_output, f._jitter, f._rtol
                )
            grad_theta = torch.autograd.grad(
                D_y, f.parameters(), aug_grad, retain_graph=x_grad
            )

            if x_grad:
                grad_x = torch.autograd.grad(D_y, x, aug_grad, retain_graph=False)[0]
            else:
                grad_x = None
            return (
                None,
                grad_x,
                *grad_theta,
            )
