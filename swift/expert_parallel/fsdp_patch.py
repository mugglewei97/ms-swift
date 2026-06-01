# Copyright (c) ModelScope Contributors. All rights reserved.
"""Patch FSDP2 so it ignores expert parameters already sharded by expert parallel.

When ``--expert_parallel true`` is used together with FSDP2, the MoE expert
weights are physically partitioned across ranks by ``expert_parallel`` (each
rank keeps a disjoint slice of experts). FSDP must NOT try to shard those
parameters again -- doing so would both double-shard them and break the
All-to-All routing that assumes each rank holds whole local experts.

FSDP2 exposes the ``ignored_params`` argument on ``fully_shard``. We wrap
``fully_shard`` so that, for every module being wrapped, any parameter carrying
the ``EP_SHARDED_FLAG`` is added to ``ignored_params``. All non-expert
parameters are still sharded by FSDP2 as usual.
"""
import torch

from swift.utils import get_logger
from .ep import EP_SHARDED_FLAG

logger = get_logger()

_patched = False


def _collect_ep_params(module: torch.nn.Module):
    """Collect parameters under ``module`` that were sharded by expert parallel."""
    ep_params = set()
    for param in module.parameters(recurse=True):
        if getattr(param, EP_SHARDED_FLAG, False):
            ep_params.add(param)
    return ep_params


def patch_fsdp_ignore_ep_params():
    """Wrap ``fully_shard`` to exclude expert-parallel sharded parameters."""
    global _patched
    if _patched:
        return

    try:
        import torch.distributed.fsdp as fsdp_module
    except ImportError:
        raise RuntimeError(
            'expert_parallel with FSDP2 requires a torch version that provides '
            f'torch.distributed.fsdp.fully_shard. Current torch version: {torch.__version__}.')

    if not hasattr(fsdp_module, 'fully_shard'):
        raise RuntimeError(
            'expert_parallel with FSDP2 requires torch.distributed.fsdp.fully_shard, '
            f'which is unavailable in torch {torch.__version__}.')

    original_fully_shard = fsdp_module.fully_shard

    from functools import wraps

    @wraps(original_fully_shard)
    def wrapped_fully_shard(module, *args, **kwargs):
        ep_params = _collect_ep_params(module)
        if ep_params:
            existing = kwargs.get('ignored_params', None)
            existing = set() if existing is None else set(existing)
            existing |= ep_params
            kwargs['ignored_params'] = existing
        return original_fully_shard(module, *args, **kwargs)

    fsdp_module.fully_shard = wrapped_fully_shard

    # accelerate imports `fully_shard` into its own namespace; patch that reference too.
    try:
        import accelerate.utils.fsdp_utils as accel_fsdp
        if hasattr(accel_fsdp, 'fully_shard'):
            accel_fsdp.fully_shard = wrapped_fully_shard
    except ImportError:
        pass

    _patched = True
    logger.info('Patched FSDP2 fully_shard to ignore expert_parallel sharded params.')
