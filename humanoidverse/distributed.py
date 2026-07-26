from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

_CONTROL_GROUP: Any | None = None


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def rank() -> int:
    return dist.get_rank() if is_distributed() else int(os.environ.get("RANK", "0"))


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else int(os.environ.get("WORLD_SIZE", "1"))


def initialize_control_group() -> None:
    """Create a CPU process group for long rank-asymmetric synchronization."""
    global _CONTROL_GROUP
    if not is_distributed() or _CONTROL_GROUP is not None:
        return
    if dist.get_backend() != "gloo":
        _CONTROL_GROUP = dist.new_group(backend="gloo", timeout=timedelta(hours=2))


def control_barrier() -> None:
    """Synchronize ranks without occupying or depending on their CUDA streams."""
    if not is_distributed():
        return
    if dist.get_backend() == "gloo":
        dist.barrier()
        return
    if _CONTROL_GROUP is None:
        raise RuntimeError("initialize_control_group() must be called on every rank before control_barrier()")
    dist.barrier(group=_CONTROL_GROUP)


def barrier() -> None:
    if is_distributed():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


@torch.compiler.disable
def average_gradients(parameters: Iterable[torch.nn.Parameter]) -> None:
    if not is_distributed():
        return
    scale = 1.0 / float(dist.get_world_size())
    for param in parameters:
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.mul_(scale)


@torch.compiler.disable
@torch.no_grad()
def clip_grad_norm_stable_(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> torch.Tensor:
    """Clip a global L2 gradient norm without overflowing float32 square sums."""
    if max_norm <= 0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")

    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return torch.zeros((), dtype=torch.float64)
    if any(grad.is_sparse for grad in grads):
        raise ValueError("clip_grad_norm_stable_ does not support sparse gradients")

    device = grads[0].device
    if any(grad.device != device for grad in grads):
        raise ValueError("clip_grad_norm_stable_ requires all gradients to be on the same device")

    max_abs = torch.zeros((), dtype=torch.float32, device=device)
    for index, grad in enumerate(grads):
        grad_max = grad.detach().abs().max().float()
        if not bool(torch.isfinite(grad_max).item()):
            raise RuntimeError(f"Gradient tensor {index} contains NaN or Inf values before clipping")
        max_abs = torch.maximum(max_abs, grad_max)

    if float(max_abs.item()) == 0.0:
        return torch.zeros((), dtype=torch.float64, device=device)

    scaled_square_sum = torch.zeros((), dtype=torch.float32, device=device)
    for grad in grads:
        scaled = grad.detach().abs().float() / max_abs
        scaled_square_sum += scaled.square().sum()

    total_norm = max_abs.double() * scaled_square_sum.double().sqrt()
    clip_coefficient = torch.clamp(torch.as_tensor(max_norm, dtype=torch.float64, device=device) / (total_norm + 1e-12), max=1.0)
    for grad in grads:
        grad.mul_(clip_coefficient.to(dtype=grad.dtype))
    return total_norm


@torch.compiler.disable
def broadcast_module_state(module: torch.nn.Module, src: int = 0) -> None:
    if not is_distributed():
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.data, src=src)


@torch.compiler.disable
def sync_floating_buffers(module: torch.nn.Module, src: int = 0) -> None:
    if not is_distributed():
        return
    scale = 1.0 / float(dist.get_world_size())
    for buffer in module.buffers():
        if buffer.dtype.is_floating_point or buffer.dtype.is_complex:
            dist.all_reduce(buffer.data, op=dist.ReduceOp.SUM)
            buffer.data.mul_(scale)
        else:
            dist.broadcast(buffer.data, src=src)


@torch.compiler.disable
def broadcast_optimizer_state(optimizer: torch.optim.Optimizer, src: int = 0) -> None:
    if not is_distributed():
        return
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                dist.broadcast(value.data, src=src)


def agent_optimizers(agent: Any) -> list[torch.optim.Optimizer]:
    names = [
        "actor_optimizer",
        "backward_optimizer",
        "forward_optimizer",
        "critic_optimizer",
        "discriminator_optimizer",
        "aux_critic_optimizer",
    ]
    return [getattr(agent, name) for name in names if hasattr(agent, name)]


def broadcast_agent_state(agent: Any, src: int = 0) -> None:
    if not is_distributed():
        return
    broadcast_module_state(agent._model, src=src)
    for optimizer in agent_optimizers(agent):
        broadcast_optimizer_state(optimizer, src=src)


@torch.compiler.disable
def broadcast_object(value: Any, src: int = 0) -> Any:
    if not is_distributed():
        return value
    objects = [value if dist.get_rank() == src else None]
    if dist.get_backend() == "gloo":
        dist.broadcast_object_list(objects, src=src)
    else:
        if _CONTROL_GROUP is None:
            raise RuntimeError("initialize_control_group() must be called on every rank before broadcast_object()")
        dist.broadcast_object_list(objects, src=src, group=_CONTROL_GROUP)
    return objects[0]


@torch.compiler.disable
def module_sync_report(module: torch.nn.Module, src: int = 0) -> dict[str, Any]:
    if not is_distributed():
        return {"world_size": 1, "rank": 0, "max_abs_diff_from_rank0": 0.0}

    tensors = list(module.parameters()) + [
        buffer for buffer in module.buffers() if buffer.dtype.is_floating_point or buffer.dtype.is_complex
    ]
    if tensors:
        device = tensors[0].device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_summary = torch.zeros(3, dtype=torch.float64, device=device)
    max_abs_diff = torch.zeros((), dtype=torch.float32, device=device)

    for tensor in tensors:
        detached = tensor.detach()
        local_summary[0] += detached.double().sum()
        local_summary[1] += detached.double().square().sum()
        local_summary[2] += detached.numel()
        reference = detached.clone()
        dist.broadcast(reference, src=src)
        if detached.numel() > 0:
            diff = (detached - reference).abs().max().float()
            max_abs_diff = torch.maximum(max_abs_diff, diff)

    dist.all_reduce(max_abs_diff, op=dist.ReduceOp.MAX)
    gathered = [torch.empty_like(local_summary) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_summary)
    summaries = torch.stack(gathered).cpu()
    return {
        "world_size": dist.get_world_size(),
        "rank": dist.get_rank(),
        "max_abs_diff_from_rank0": float(max_abs_diff.cpu().item()),
        "rank_param_buffer_sum": [float(x) for x in summaries[:, 0].tolist()],
        "rank_param_buffer_sqsum": [float(x) for x in summaries[:, 1].tolist()],
        "rank_param_buffer_numel": [int(x) for x in summaries[:, 2].tolist()],
    }


@torch.compiler.disable
def average_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    if not is_distributed():
        return dict(metrics)
    reduced: dict[str, Any] = {}
    scale = 1.0 / float(dist.get_world_size())
    for key, value in metrics.items():
        if torch.is_tensor(value):
            tensor = value.detach().clone()
            if not (tensor.dtype.is_floating_point or tensor.dtype.is_complex):
                tensor = tensor.float()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            reduced[key] = tensor * scale
        else:
            reduced[key] = value
    return reduced
