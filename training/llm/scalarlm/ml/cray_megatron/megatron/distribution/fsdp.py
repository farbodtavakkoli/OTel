import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from cray_infra.training.distributed import (
    get_size,
    get_rank,
    allgather,
    reduce_scatter,
    cuda_device,
)
from collections import defaultdict
from cray_infra.training.metrics import get_model_memory_footprint

import time
import logging

logger = logging.getLogger(__name__)


class FSDPLayer(nn.Module):
    def __init__(self, module, should_checkpoint=False, should_recurse=False):
        super().__init__()
        self.module = module

        self.module.register_full_backward_hook(self._full_backward_hook)

        self.should_checkpoint = should_checkpoint
        self.should_recurse = should_recurse

        self.perf_metrics = {
            "shard": {
                "time": 0.0,
            },
            "gather": {
                "time": 0.0,  # Placeholder for elapsed time in seconds
                "bytes": 0,  # Placeholder for bytes processed
            },
            "forward_pass": {"time": 0.0},  # Placeholder for elapsed time in seconds
            "free_params": {"time": 0.0},  # Placeholder for elapsed time in seconds
            "reduce_scatter": {"time": 0.0},  # Placeholder for elapsed time in seconds
        }

        # Whether this layer's parameters are currently all-gathered.
        self._params_gathered = False

        start = time.time()
        self.shard_parameters()
        end = time.time()
        self.perf_metrics["shard"]["time"] += end - start

    def _full_backward_hook(self, module, grad_input, grad_output):
        self.free_params()

    def shard_parameters(self):
        rank = get_rank()
        self.sharded_parameter_metadata = {}

        logger.debug(f"Rank {rank}: Sharding parameters for {self.module}")

        for name, param in list(
            self.module.named_parameters(recurse=self.should_recurse)
        ):
            local_name = name.split(".")[-1]
            shard, metadata_dict = shard_tensor(param)

            self.sharded_parameter_metadata[name] = metadata_dict

            logger.debug(
                f" Rank {rank}: Sharding parameter {name} with shape {param.shape} "
                f"into {metadata_dict}, new shape {shard.shape}"
            )

            parent_module = self.get_parent_module(self.module, param)

            setattr(
                parent_module,
                "shard_" + local_name,
                nn.Parameter(shard, requires_grad=param.requires_grad),
            )
            delattr(parent_module, local_name)
            setattr(parent_module, local_name, shard)

        logger.debug(
            f" Rank {rank}: Sharded parameters are "
            f"{[i[0] for i in self.module.named_parameters(recurse=self.should_recurse)]}"
        )

    def forward(self, *args, **kwargs):
        if self.should_checkpoint:
            return checkpoint(self.forward_op, *args, use_reentrant=False, **kwargs)
        else:
            return self.forward_op(*args, **kwargs)

    def forward_op(self, *args, **kwargs):
        # gather all params
        start = time.time()
        self.gather_all_parameters()
        end = time.time()
        self.perf_metrics["gather"]["time"] += end - start

        # run forward pass
        start = time.time()
        result = self.module(*args, **kwargs)
        end = time.time()
        self.perf_metrics["forward_pass"]["time"] += end - start

        # free params
        start = time.time()
        self.free_params()
        end = time.time()
        self.perf_metrics["free_params"]["time"] += end - start

        return result

    def gather_all_parameters(self):
        rank = get_rank()
        logger.debug(f"Rank {rank}: Gathering parameters for {self.module}")

        for name, param in self.module.named_parameters(recurse=self.should_recurse):

            local_name = name.split(".")[-1]

            prefix = name[: -len(local_name)]

            if not local_name.startswith("shard_"):
                logger.debug(f" Rank {rank}: Skipping parameter {name}")
                continue

            # Remove _shard_ prefix
            local_name = local_name[6:]

            full_unsharded_name = prefix + local_name

            parent_module = self.get_parent_module(self.module, param)

            # Get metadata for this parameter
            metadata_dict = self.sharded_parameter_metadata[full_unsharded_name]

            # Gather all shards and reconstruct the full tensor
            full_tensor = all_gather_op(param, metadata_dict, self.perf_metrics)

            logger.debug(
                f" Rank {rank}: Gathered parameter {full_unsharded_name} with shape {full_tensor.shape}"
            )

            # Copy the full tensor back to the original parameter
            setattr(parent_module, local_name, full_tensor)

        self._params_gathered = True

    def free_params(self):
        for name, param in self.module.named_parameters(recurse=self.should_recurse):
            local_name = name.split(".")[-1]

            if not local_name.startswith("shard_"):
                continue

            # Remove _shard_ prefix
            local_name = local_name[6:]

            parent_module = self.get_parent_module(self.module, param)

            if hasattr(parent_module, local_name):
                if (
                    getattr(parent_module, local_name).data_ptr()
                    != param.data.data_ptr()
                ):
                    delattr(parent_module, local_name)

            setattr(parent_module, local_name, param.data)

        self._params_gathered = False

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass

        # Read through to the wrapped module via __dict__ so a miss cannot recurse into __getattr__.
        module = (self.__dict__.get("_modules") or {}).get("module")
        if module is None:
            raise AttributeError(name)

        # Parameters unshard only inside forward, so an outside read must gather on demand.
        metadata = self.__dict__.get("sharded_parameter_metadata") or {}
        if name in metadata and not self.__dict__.get("_params_gathered", False):
            shard = getattr(module, "shard_" + name, None)
            if shard is not None:
                full_tensor = all_gather_op(
                    shard, metadata[name], self.__dict__.get("perf_metrics", {})
                )
                setattr(module, name, full_tensor)
                return full_tensor

        return getattr(module, name)

    def get_parent_module(self, module, param):
        for potential_param in module.parameters(recurse=False):
            if potential_param is param:
                return module

        for name, child in module.named_children():
            parent = self.get_parent_module(child, param)
            if parent is not None:
                return parent

        return None


class SimpleFSDP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model_memory_footprint = get_model_memory_footprint(self.model)
        self._wrap_layers(model)

    def _wrap_layers(self, module):

        for name, child in module.named_children():
            is_conditional_layer = self._get_is_conditional_layer(child)
            grand_children = list(child.children())

            has_grand_children = False
            if len(grand_children) > 0:
                has_grand_children = True

            all_params = list(child.parameters(recurse=is_conditional_layer))
            any_requires_grad = any(param.requires_grad for param in all_params)

            # checkpoint if model memory footprint is > 32GB and has grand children and has any requires_grad
            should_checkpoint = (
                self.model_memory_footprint > (32 * 1024**3)
                and has_grand_children
                and any_requires_grad
            )

            if len(all_params) > 0:
                wrapped = FSDPLayer(
                    child,
                    should_checkpoint=should_checkpoint,
                    should_recurse=is_conditional_layer,
                )
                setattr(module, name, wrapped)

            if has_grand_children and not is_conditional_layer:
                self._wrap_layers(child)

    def _get_is_conditional_layer(self, module):
        # Assume a module named experts is a conditional layer and do not wrap its children.
        for name, child in module.named_children():
            if "expert" in name.lower():
                return True

    def forward(self, *args, **kwargs):
        result = self.model(*args, **kwargs)

        rank = get_rank()
        if rank == 0:
            metrics_str = "\n"
            # Aggregate and print metrics
            aggregated = aggregate_perf_metrics(self.model)
            for op, metrics in aggregated.items():
                total_time = "{:.2f}".format(metrics["time"])
                metrics_str += f"{op}:\n  Total time: {total_time} s\n"

            logger.debug(metrics_str)

        return result

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def unwrap_model(self):
        unwrapped_state_dict = {}

        self.unwrap_layers(
            prefix="",
            module=self.model,
            unwrapped_state_dict=unwrapped_state_dict,
        )

        return unwrapped_state_dict

    def unwrap_layers(self, prefix, module, unwrapped_state_dict):
        rank = get_rank()
        for name, child in module.named_children():
            if isinstance(child, FSDPLayer):
                logger.debug(f" Rank {rank}: Unwrapping module {prefix}{name}")

                for param_name, param in child.module.named_parameters(recurse=False):
                    if param.requires_grad:
                        local_name = param_name.split(".")[-1]
                        name_prefix = param_name[: -len(local_name)]

                        if local_name.startswith("shard_"):
                            # Remove _shard_ prefix
                            local_name = param_name[len("shard_") :]
                            full_unsharded_name = name_prefix + local_name

                            # Get metadata for this parameter
                            metadata_dict = child.sharded_parameter_metadata[
                                full_unsharded_name
                            ]

                            # Gather all shards and reconstruct the full tensor
                            full_tensor = all_gather_op(param, metadata_dict)

                            unwrapped_state_dict[
                                f"{prefix}{name}.{full_unsharded_name}"
                            ] = full_tensor.to(torch.device("cpu"))

                        else:
                            unwrapped_state_dict[f"{prefix}{name}.{param_name}"] = (
                                param.to(torch.device("cpu"))
                            )

                        logger.debug(
                            f" Rank {rank}: Unwrapping parameter {prefix}{name}.{param_name}"
                        )

            self.unwrap_layers(
                prefix=prefix + name + ".",
                module=child,
                unwrapped_state_dict=unwrapped_state_dict,
            )

    def load_unwrapped_model(self, state_dict):
        """Inverse of unwrap_model: re-shard each full tensor and write this rank's slice."""
        missing: list[str] = []
        loaded = self._load_unwrapped_layers(
            prefix="", module=self.model, state_dict=state_dict, missing=missing
        )
        if loaded == 0:
            raise RuntimeError(
                "FSDP checkpoint restore matched no sharded parameters — the "
                "model_state_dict namespace does not line up with the live "
                "sharded model."
            )
        if missing:
            raise RuntimeError(
                f"FSDP checkpoint restore is missing {len(missing)} trained "
                f"parameter(s), e.g. {missing[:3]}. Refusing to continue with "
                "partially-restored weights."
            )
        logger.info(
            "Resharded %d trained parameter tensor(s) into the FSDP model on resume",
            loaded,
        )

    def _load_unwrapped_layers(self, prefix, module, state_dict, missing):
        rank = get_rank()
        world_size = get_size()
        loaded = 0
        for name, child in module.named_children():
            if isinstance(child, FSDPLayer):
                for param_name, param in child.module.named_parameters(recurse=False):
                    if not param.requires_grad:
                        continue
                    local_name = param_name.split(".")[-1]
                    name_prefix = param_name[: -len(local_name)]

                    if local_name.startswith("shard_"):
                        unsharded_local = param_name[len("shard_") :]
                        full_unsharded_name = name_prefix + unsharded_local
                        key = f"{prefix}{name}.{full_unsharded_name}"
                        if key not in state_dict:
                            missing.append(key)
                            continue
                        metadata_dict = child.sharded_parameter_metadata[
                            full_unsharded_name
                        ]
                        shard = shard_full_tensor(
                            state_dict[key], metadata_dict, rank, world_size
                        )
                        with torch.no_grad():
                            param.copy_(
                                shard.to(device=param.device, dtype=param.dtype)
                            )
                        loaded += 1
                    else:
                        key = f"{prefix}{name}.{param_name}"
                        if key not in state_dict:
                            missing.append(key)
                            continue
                        with torch.no_grad():
                            param.copy_(
                                state_dict[key].to(
                                    device=param.device, dtype=param.dtype
                                )
                            )
                        loaded += 1

            loaded += self._load_unwrapped_layers(
                prefix=prefix + name + ".",
                module=child,
                state_dict=state_dict,
                missing=missing,
            )
        return loaded

    def _sharded_metadata_by_param(self):
        """id(param) -> metadata_dict, for every `shard_` parameter."""
        out = {}

        def walk(module):
            for name, child in module.named_children():
                if isinstance(child, FSDPLayer):
                    for param_name, param in child.module.named_parameters(
                        recurse=False
                    ):
                        if not param.requires_grad:
                            continue
                        local_name = param_name.split(".")[-1]
                        name_prefix = param_name[: -len(local_name)]
                        if local_name.startswith("shard_"):
                            unsharded_local = param_name[len("shard_") :]
                            full_unsharded_name = name_prefix + unsharded_local
                            out[id(param)] = child.sharded_parameter_metadata[
                                full_unsharded_name
                            ]
                walk(child)

        walk(self.model)
        return out

    def unwrap_optimizer(self, optimizer):
        """Full, unsharded, CPU optimizer state - the counterpart to unwrap_model (collective)."""
        metadata_by_param = self._sharded_metadata_by_param()
        state_dict = optimizer.state_dict()
        ordered_params = [p for group in optimizer.param_groups for p in group["params"]]

        gathered_state = {}
        for index, entry in state_dict.get("state", {}).items():
            metadata = (
                metadata_by_param.get(id(ordered_params[index]))
                if index < len(ordered_params)
                else None
            )
            new_entry = {}
            for key, value in entry.items():
                if not torch.is_tensor(value):
                    new_entry[key] = value
                elif metadata is not None and value.dim() > 0:
                    # A per-shard moment: gather it back to the whole tensor.
                    new_entry[key] = (
                        collectives_all_gather(value, metadata).detach().cpu()
                    )
                else:
                    # Replicated, or a 0-d scalar like `step`.
                    new_entry[key] = value.detach().cpu()
            gathered_state[index] = new_entry

        return {
            "state": gathered_state,
            "param_groups": state_dict.get("param_groups", []),
        }

    def load_unwrapped_optimizer(self, optimizer, state_dict):
        """Inverse of unwrap_optimizer: re-shard full moments into this rank (collective)."""
        rank = get_rank()
        world_size = get_size()
        metadata_by_param = self._sharded_metadata_by_param()
        ordered_params = [p for group in optimizer.param_groups for p in group["params"]]

        resharded_state = {}
        for index, entry in state_dict.get("state", {}).items():
            index = int(index)
            param = ordered_params[index] if index < len(ordered_params) else None
            metadata = metadata_by_param.get(id(param)) if param is not None else None
            new_entry = {}
            for key, value in entry.items():
                if not torch.is_tensor(value):
                    new_entry[key] = value
                elif metadata is not None and value.dim() > 0:
                    shard = shard_full_tensor(value, metadata, rank, world_size)
                    new_entry[key] = shard.to(
                        device=param.device, dtype=param.dtype
                    )
                elif param is not None:
                    new_entry[key] = value.to(device=param.device)
                else:
                    new_entry[key] = value
            resharded_state[index] = new_entry

        optimizer.load_state_dict(
            {
                "state": resharded_state,
                "param_groups": state_dict.get(
                    "param_groups", optimizer.state_dict()["param_groups"]
                ),
            }
        )

    def backward_sync(self):
        pass


def get_fsdp_layers(module):
    """Recursively collect all FSDPLayer instances"""
    fsdp_layers = []
    for child in module.children():
        if isinstance(child, FSDPLayer):
            fsdp_layers.append(child)
        fsdp_layers.extend(get_fsdp_layers(child))
    return fsdp_layers


def aggregate_perf_metrics(module):

    fsdp_layers = get_fsdp_layers(module)

    """Sum metrics across all FSDP layers"""
    aggregated = defaultdict(lambda: {"time": 0, "bytes": 0})
    for layer in fsdp_layers:
        for op, metrics in layer.perf_metrics.items():
            aggregated[op]["time"] += metrics.get("time", 0)
            if "bytes" in aggregated[op]:
                aggregated[op]["bytes"] += metrics.get("bytes", 0)
    return dict(aggregated)


def shard_tensor(tensor):
    """Evenly shard tensor across ranks, returning (shard, metadata_dict)."""
    world_size = get_size()
    rank = get_rank()

    original_shape = tensor.shape
    original_numel = tensor.numel()

    # Calculate padding for equal division
    padded_numel = ((original_numel + world_size - 1) // world_size) * world_size
    padding = padded_numel - original_numel

    # Pad tensor if needed
    if padding > 0:
        tensor_padded = torch.cat(
            [
                tensor.view(-1),
                torch.zeros(padding, device=tensor.device, dtype=tensor.dtype),
            ]
        )
    else:
        tensor_padded = tensor.view(-1)

    # Split into equal shards
    shard_size = padded_numel // world_size
    start = rank * shard_size
    shard = tensor_padded[start : start + shard_size].clone()

    # NCCL/RCCL collectives need CUDA tensors, so metadata buffers are allocated on-device.
    device = cuda_device()
    local_metadata = torch.tensor(
        [original_numel, *original_shape, shard_size, padding],
        dtype=torch.long,
        device=device,
    )
    all_metadata = torch.zeros(
        (world_size, local_metadata.numel()), dtype=torch.long, device=device
    )
    allgather(local_metadata, all_metadata.view(-1))

    # Reshape all_metadata back to 2D after allgather
    all_metadata = all_metadata.view(world_size, -1)

    # Create a dictionary of metadata keyed by rank
    metadata_dict = {rank: all_metadata[rank].tolist() for rank in range(world_size)}

    # Convert metadata back to original format
    metadata_dict = {
        rank: (meta[0], tuple(meta[1:-2]), meta[-2], meta[-1])
        for rank, meta in metadata_dict.items()
    }

    return shard, metadata_dict


def shard_full_tensor(full_tensor, metadata_dict, rank, world_size):
    """Inverse of shard_tensor() for a single rank."""
    shard_size = metadata_dict[rank][2]
    flat = full_tensor.reshape(-1)
    padded_numel = shard_size * world_size
    if flat.numel() < padded_numel:
        flat = torch.cat([flat, flat.new_zeros(padded_numel - flat.numel())])
    start = rank * shard_size
    return flat[start : start + shard_size].clone()


def trim_padding(all_tensors, rank, world_size, metadata_dict):

    original_numel, _, shard_size, padding = metadata_dict[rank]

    if padding == 0:
        return all_tensors

    # all_tensors is a list of tensors that have been collected
    if padding > shard_size:
        # Calculate the number of fully padded tensors
        fully_padded_tensors = padding // shard_size

        # Remove fully padded tensors
        all_tensors = all_tensors[:-fully_padded_tensors]

        # Calculate the remaining padding in the last tensor
        remaining_padding = padding % shard_size

        # Trim the remaining padding from the last tensor
        if remaining_padding > 0 and len(all_tensors) > 0:
            last_tensor = all_tensors[-1]
            all_tensors[-1] = last_tensor[:-remaining_padding]
    else:
        # trim padding for last tensor
        valid_elements = original_numel - (world_size - 1) * shard_size
        all_tensors[-1] = all_tensors[-1][:valid_elements]

    return all_tensors


def collectives_all_gather(shard, metadata_dict):
    """Gather shards and reconstruct the full tensor using metadata."""
    world_size = get_size()
    rank = get_rank()

    # fp32 on-device buffers keep the reduction safe regardless of the parameter dtype.
    device = cuda_device()
    orig_dtype = shard.dtype
    shard = shard.to(torch.float32).to(device)
    gathered = torch.zeros(
        shard.numel() * world_size, device=device, dtype=torch.float32
    )

    # Collective operation in float32
    allgather(shard, gathered)

    # Convert gathered result back to original dtype
    gathered = gathered.to(orig_dtype)

    # Reconstruct the full tensor using metadata
    all_tensors = []
    offset = 0
    for r in range(world_size):
        shard_size = metadata_dict[r][2]
        rank_shard_flattened = gathered[offset : offset + shard_size]
        all_tensors.append(rank_shard_flattened)
        offset += shard_size

    all_tensors = trim_padding(all_tensors, rank, world_size, metadata_dict)
    concatenated = torch.cat(all_tensors)
    original_shape = metadata_dict[rank][1]

    return concatenated.reshape(original_shape)


def collectives_reduce_scatter(tensor, metadata_dict):
    """Reduce-scatter with even sharding. Returns local shard trimmed to original size."""
    world_size = get_size()
    rank = get_rank()

    # Save original dtype
    orig_dtype = tensor.dtype

    original_numel, _, shard_size, padding = metadata_dict[rank]

    # Pad tensor if needed. Buffers live on-device for NCCL/RCCL.
    device = cuda_device()
    tensor_padded = tensor.reshape(-1).to(device)
    if padding > 0:
        tensor_padded = torch.concatenate(
            [
                tensor_padded,
                torch.zeros(padding, device=device, dtype=tensor_padded.dtype),
            ]
        )

    # Convert to float32 for the collective
    tensor_padded = tensor_padded.to(torch.float32)
    local_shard = torch.zeros(shard_size, device=device, dtype=torch.float32)

    # Collective operation in float32
    reduce_scatter(tensor_padded, local_shard)

    # reduce_scatter() sums, but gradients must be the MEAN across ranks.
    if world_size > 1:
        local_shard /= world_size

    # Convert result back to original dtype
    local_shard = local_shard.to(orig_dtype)

    # Trim padding on last rank using its original size from metadata_dict
    if rank == world_size - 1:
        valid_elements = original_numel - padding
        local_shard = local_shard[:valid_elements]

    return local_shard


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shard, metadata_dict, metrics={}):
        ctx.metadata_dict = metadata_dict
        ctx.metrics = metrics
        return collectives_all_gather(shard, metadata_dict)

    @staticmethod
    def backward(ctx, grad_output):
        metadata_dict = ctx.metadata_dict
        metrics = ctx.metrics

        start = time.time()
        result = collectives_reduce_scatter(grad_output, metadata_dict), None, None
        end = time.time()
        if "reduce_scatter" in metrics:
            metrics["reduce_scatter"]["time"] += end - start

        return result


all_gather_op = _AllGather.apply
