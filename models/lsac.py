import torch

from models.sequential.sequential_sac import SequentialSoftActorCritic


class LipschitzSoftActorCritic(SequentialSoftActorCritic):
    _agent_name = "LSAC"

    def __init__(self, env, args, actor_nn, critic_nn):
        super().__init__(env, args, actor_nn, critic_nn)

        # Lipschitz constraint
        self.L = 1.0
        # gradient penalty weight
        self.gradient_penalty_weight = 0.001

    def calculate_gradient_penalty(self, critic, s, a):
        # calculate the gradient of the critic with respect to the input
        batch_size = s.size(0)
        s_req_grad = torch.autograd.Variable(s * 1.0, requires_grad=True)
        a_req_grad = torch.autograd.Variable(a * 1.0, requires_grad=True)
        critic_out = critic(s_req_grad, a_req_grad)
        gradients = torch.autograd.grad(
            outputs=critic_out,
            inputs=(s_req_grad, a_req_grad),
            grad_outputs=torch.ones_like(critic_out),
            create_graph=True,
            retain_graph=True,
        )[0]
        # calculate the gradient penalty
        gradient_penalty = (gradients.norm(2, dim=1) - self.L) ** 2
        return gradient_penalty.mean()

    def learn(self, max_iter=1):
        if self.args.batch_size > len(self.experience_memory):
            return None

        for iteration in range(max_iter):
            s, a, r, sp, done, step = self.experience_memory.sample_random(
                self.args.batch_size
            )
            # generate q targets
            with torch.no_grad():
                up_pred, ep_pred = self._actor(sp)
                qp_1_target = self._critic_1_target(sp, up_pred)
                qp_2_target = self._critic_2_target(sp, up_pred)
                qp_target = torch.min(qp_1_target, qp_2_target) - self._alpha * ep_pred
                q_target = r.unsqueeze(-1) + (
                    self._gamma * qp_target * (1 - done.unsqueeze(-1))
                )

            # update critic 1
            self._critic_1_optim.zero_grad()
            q_1 = self._critic_1(s, a)
            q_1_loss = self._critic_loss_fcn(q_1, q_target)
            # calculate the gradient penalty Lipschitz constraint
            gradient_penalty_1 = self.calculate_gradient_penalty(self._critic_1, s, a)
            q_1_loss = q_1_loss + self.gradient_penalty_weight * gradient_penalty_1
            q_1_loss.backward()
            self._critic_1_optim.step()
            # update critic 2
            self._critic_2_optim.zero_grad()
            q_2 = self._critic_2(s, a)
            q_2_loss = self._critic_loss_fcn(q_2, q_target)
            # calculate the gradient penalty Lipschitz constraint
            gradient_penalty_2 = self.calculate_gradient_penalty(self._critic_2, s, a)
            q_2_loss = q_2_loss + self.gradient_penalty_weight * gradient_penalty_2
            q_2_loss.backward()
            self._critic_2_optim.step()
            # update actor
            self._actor_optim.zero_grad()
            a_pred, e_pred = self._actor(s)
            # ---
            q_pi = torch.min(self._critic_1(s, a_pred), self._critic_2(s, a_pred))
            pi_loss = (-q_pi + self._alpha * e_pred).mean()
            pi_loss.backward()
            self._actor_optim.step()
            # soft update of target critic networks
            self._soft_update(self._critic_1, self._critic_1_target)
            self._soft_update(self._critic_2, self._critic_2_target)

            # print(f"q_1_loss: {(q_1_loss.item() - self.gradient_penalty_weight*gradient_penalty_1.item()):10.5f}, q_2_loss: {(q_2_loss.item() - self.gradient_penalty_weight*gradient_penalty_2.item()):10.5f}, gradient_penalty_1: {gradient_penalty_1.item():10.5f}, gradient_penalty_2: {gradient_penalty_2.item():10.5f}")
