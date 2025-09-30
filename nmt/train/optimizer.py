"""adadelta optimizer per zeiler 2012 with rho 0.95 eps 1e-6."""

import torch


class Adadelta(torch.optim.Optimizer):
    """adadelta with the classic rms normalization of updates."""

    def __init__(self, params, rho: float = 0.95, eps: float = 1e-6):
        defaults = dict(rho=rho, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """one parameter update over all gradients."""
        for group in self.param_groups:
            rho = group["rho"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["acc_g"] = torch.zeros_like(p)
                    state["acc_dx"] = torch.zeros_like(p)
                acc_g = state["acc_g"]
                acc_dx = state["acc_dx"]
                acc_g.mul_(rho).addcmul_(grad, grad, value=1 - rho)
                scaled = grad.mul((acc_dx + eps).sqrt().div((acc_g + eps).sqrt()))
                p.add_(-scaled)
                acc_dx.mul_(rho).addcmul_(scaled, scaled, value=1 - rho)

    def statistics(self):
        """mean rms of the gradient and update accumulators."""
        rms_g = []
        rms_dx = []
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state:
                    continue
                rms_g.append(state["acc_g"].mean().sqrt().item())
                rms_dx.append(state["acc_dx"].mean().sqrt().item())
        if not rms_g:
            return 0.0, 0.0, 0.0
        g = sum(rms_g) / len(rms_g)
        dx = sum(rms_dx) / len(rms_dx)
        return g, dx, dx / max(g, 1e-12)