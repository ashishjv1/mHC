"""
Muon optimizer: Momentum Orthogonalized by Newton-Schulz.

For 2D+ weight matrices, the momentum buffer is orthogonalized via Newton-Schulz
before being used as the update direction. This equalizes singular values of the
update, providing better conditioning.

Non-matrix parameters (biases, norms, embeddings) are handled by a standard
AdamW update internally.

Reference: Keller Jordan, "Muon: An optimizer for hidden layers in neural networks"
"""
import torch
from torch.optim import Optimizer
from src.newton_schulz import newton_schulz_muon


class Muon(Optimizer):
    """
    Muon for 2D weight matrices. Pair with a separate AdamW for 1D params.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0,
                 ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_steps=ns_steps)
        super().__init__(params, defaults)
        self._spectrum_stats = {}

    @torch.no_grad()
    def step(self):
        self._spectrum_stats.clear()
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)

                if g.ndim >= 2:
                    update = newton_schulz_muon(buf, steps=ns_steps)
                    # Scale so the update Frobenius norm ~ sqrt(max(m,n))
                    scale = max(g.shape[0], g.shape[1]) ** 0.5
                    update = update * scale

                    if id(p) not in self._spectrum_stats:
                        svs = torch.linalg.svdvals(update.float()[:min(g.shape), :min(g.shape)])
                        self._spectrum_stats[id(p)] = {
                            "min": svs.min().item(),
                            "max": svs.max().item(),
                            "mean": svs.mean().item(),
                            "std": svs.std().item(),
                        }
                else:
                    update = buf

                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)

    def get_spectrum_stats(self):
        return self._spectrum_stats
