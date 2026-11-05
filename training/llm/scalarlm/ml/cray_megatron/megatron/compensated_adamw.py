import math

import torch
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer


def _local_tensor(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class CompensatedAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        chunk_size=1048576,
        maximize=False,
    ):
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            chunk_size=chunk_size,
            maximize=maximize,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            self._validate_group(group)

    @staticmethod
    def _validate_group(group):
        for name in ("lr", "eps", "weight_decay"):
            value = group[name]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}: {value}")
        if len(group["betas"]) != 2 or any(
            not math.isfinite(beta) or not 0 <= beta < 1 for beta in group["betas"]
        ):
            raise ValueError(f"Invalid betas: {group['betas']}")
        chunk_size = group["chunk_size"]
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            self._validate_group(group)
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                local_param = _local_tensor(param)
                grad = _local_tensor(param.grad)
                if grad.layout != torch.strided:
                    raise RuntimeError("CompensatedAdamW does not support sparse gradients")
                if local_param.dtype != torch.bfloat16 or grad.dtype != torch.bfloat16:
                    raise TypeError("CompensatedAdamW requires BF16 parameters and gradients")
                state = self.state[param]
                if not state:
                    state["step"] = torch.zeros((), dtype=torch.float32)
                    for name in ("exp_avg", "exp_avg_sq", "residual"):
                        state[name] = torch.zeros_like(param, memory_format=torch.contiguous_format)
                state["step"].add_(1)
                step = state["step"].item()
                correction1 = 1 - beta1 ** step
                correction2 = 1 - beta2 ** step
                param_flat = local_param.reshape(-1)
                grad_flat = grad.reshape(-1)
                avg_flat = _local_tensor(state["exp_avg"]).view(-1)
                sq_flat = _local_tensor(state["exp_avg_sq"]).view(-1)
                residual_flat = _local_tensor(state["residual"]).view(-1)
                for start in range(0, local_param.numel(), group["chunk_size"]):
                    end = start + group["chunk_size"]
                    p = param_flat[start:end]
                    m = avg_flat[start:end]
                    v = sq_flat[start:end]
                    r = residual_flat[start:end]
                    g = grad_flat[start:end].float()
                    if group["maximize"]:
                        g.neg_()
                    avg = m.float().mul_(beta1).add_(g, alpha=1 - beta1)
                    sq = v.float().mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    m.copy_(avg)
                    v.copy_(sq)
                    effective = p.float().add_(r.float())
                    effective.mul_(1 - group["lr"] * group["weight_decay"])
                    denom = sq.div_(correction2).sqrt_().add_(group["eps"])
                    effective.addcdiv_(avg, denom, value=-group["lr"] / correction1)
                    p.copy_(effective)
                    r.copy_(effective.sub_(p.float()))
                if not local_param.is_contiguous():
                    local_param.copy_(param_flat.view(local_param.shape))
        return loss
