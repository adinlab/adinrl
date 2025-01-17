import copy
import numpy
import torch
import torch as th
from torch import nn, func as thf



class Critic(nn.Module):
    def __init__(self, arch, args, n_state, n_action):
        super().__init__()
        self.args = args
        self.arch = arch
        self.model = arch(
            n_state,
            n_action,
            depth=args.depth_critic,
            width=args.width_critic,
            act=args.act_critic,
            has_norm=not args.no_norm_critic,
        ).to(args.device)
        self.target = arch(
            n_state,
            n_action,
            depth=args.depth_critic,
            width=args.width_critic,
            act=args.act_critic,
            has_norm=not args.no_norm_critic,
        ).to(args.device)
        self.init_target()
        # self.loss = nn.MSELoss()
        self.loss = nn.HuberLoss()
        self.optim = torch.optim.Adam(self.model.parameters(), args.learning_rate)
        self.iter = 0

    def set_writer(self, writer):
        self.writer = writer

    def init_target(self):
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(local_param.data)

    def update_target(self):
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters()
        ):
            target_param.data.mul_(1.0 - self.args.tau)
            target_param.data.add_(self.args.tau * local_param.data)

    def Q(self, s, a):
        if a.shape == ():
            a = a.view(1, 1)
        return self.model(th.cat((s, a), -1))

    def Q_t(self, s, a):
        if a.shape == ():
            a = a.view(1, 1)
        return self.target(th.cat((s, a), -1))

    def update(self, s, a, y):  # y denotes bellman target
        self.optim.zero_grad()
        loss = self.loss(self.Q(s, a), y)
        loss.backward()
        self.optim.step()
        self.iter += 1


class Critics(nn.Module):
    def __init__(self, arch, args, n_state, n_action, critictype=Critic):
        super().__init__()
        self.n_members = args.n_critics
        self.args = args
        self.arch = arch
        self.n_state = n_state
        self.n_action = n_action
        self.critictype = critictype
        self.iter = 0
        # self.loss = nn.MSELoss()
        self.loss = self.critictype(
            self.arch, self.args, self.n_state, self.n_action
        ).loss
        self.optim = self.critictype(
            self.arch, self.args, self.n_state, self.n_action
        ).optim

        # Helperfunctions
        self.expand = lambda x: (
            x.expand(self.n_members, *x.shape) if len(x.shape) < 3 else x
        )
        # self.reduce = lambda q_val: q_val.min(0)[0]

        self.reset()

    def reset(self):
        self.critics = [
            self.critictype(self.arch, self.args, self.n_state, self.n_action)
            for _ in range(self.n_members)
        ]

        self.critics_model = [
            self.critictype(self.arch, self.args, self.n_state, self.n_action).model
            for _ in range(self.n_members)
        ]
        self.critics_target = [
            self.critictype(self.arch, self.args, self.n_state, self.n_action).target
            for _ in range(self.n_members)
        ]

        self.params_model, self.buffers_model = thf.stack_module_state(
            self.critics_model
        )
        self.params_target, self.buffers_target = thf.stack_module_state(
            self.critics_target
        )

        self.base_model = copy.deepcopy(self.critics[0].model).to("meta")
        self.base_target = copy.deepcopy(self.critics[0].target).to("meta")

        def _fmodel(base_model, params, buffers, x):
            return thf.functional_call(base_model, (params, buffers), (x,))

        self.forward_model = thf.vmap(lambda p, b, x: _fmodel(self.base_model, p, b, x))
        self.forward_target = thf.vmap(
            lambda p, b, x: _fmodel(self.base_target, p, b, x)
        )
        self.optim = th.optim.Adam(
            self.params_model.values(), lr=self.args.learning_rate
        )

    def reduce(self, q_val):
        return q_val.min(0)[0]

    def __getitem__(self, item):
        return self.critics[item]

    def unstack(self, target=False, single=True, net_id=None):
        """
        Extract the single parameters back to the individual members
        target: whether the target ensemble should be extracted or not
        single: whether just the first member of the ensemble should be extracted
        """
        params = self.params_target if target else self.params_model
        if single and net_id is None:
            net_id = 0

        for key in params.keys():
            if single:
                tmp = (
                    self.critics[net_id].model
                    if not target
                    else self.critics[net_id].target
                )
                for name in key.split("."):
                    tmp = getattr(tmp, name)
                tmp.data.copy_(params[key][net_id])
            else:
                for net_id in range(self.n_members):
                    tmp = (
                        self.critics[net_id].model
                        if not target
                        else self.critics[net_id].target
                    )
                    for name in key.split("."):
                        tmp = getattr(tmp, name)
                    tmp.data.copy_(params[key][net_id])
                    if single:
                        break

    def set_writer(self, writer):
        assert (
            writer is None
        ), "For now nothing else is implemented for the parallel version"
        self.writer = writer
        [critic.set_writer(writer) for critic in self.critics]

    def Q(self, s, a):
        SA = self.expand(th.cat((s, a), -1))
        return self.forward_model(self.params_model, self.buffers_model, SA)

    @th.no_grad()
    def Q_t(self, s, a):
        SA = self.expand(th.cat((s, a), -1))
        return self.forward_target(self.params_target, self.buffers_target, SA)

    def update(self, s, a, y):  # y denotes bellman target
        self.optim.zero_grad()
        loss = self.loss(self.Q(s, a), self.expand(y))
        loss.backward()
        self.optim.step()
        self.iter += 1

    @torch.no_grad()
    def update_target(self):
        for key in self.params_model.keys():
            self.params_target[key].data.mul_(1.0 - self.args.tau)
            self.params_target[key].data.add_(
                self.args.tau * self.params_model[key].data
            )

    @torch.no_grad()
    def get_bellman_target(self, r, sp, done, actor):
        alpha = actor.log_alpha.exp().detach() if hasattr(actor, "log_alpha") else 0
        ap, ep = actor.act(sp)
        qp = self.Q_t(sp, ap)
        qp_t = self.reduce(qp) - alpha * (ep if ep is not None else 0)
        y = r.unsqueeze(-1) + (self.args.gamma * qp_t * (1 - done.unsqueeze(-1)))
        return y
    



    def save_params(self, path):
        self.unstack(target=False, single=True, net_id=None)
        self.unstack(target=True, single=True, net_id=None)
        params_list = []
        for i in range (len(self.critics)):
            params_list.append(self.load_params(self.critics[i]))
        torch.save(params_list, path )
            
    def load_params(self,critic):
        params = {
            "params_model": critic.model.state_dict(),   
            "params_target": critic.target.state_dict(),   
            "optim": self.optim.state_dict(),
        }
        params_th = {
        k: v if isinstance(v, torch.Tensor) else v  # Ensure the values are tensors
        for k, v in params.items()
}
        return params_th
        


