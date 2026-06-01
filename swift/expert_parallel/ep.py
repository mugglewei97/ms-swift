# Copyright (c) ModelScope Contributors. All rights reserved.
"""Expert Parallel (EP) for MoE models.

Design mirrors the structure of ``swift/sequence_parallel/ulysses.py``:
- A global singleton ``expert_parallel`` holds the EP device mesh / process group.
- ``prepare(...)`` is called once on the raw model (before FSDP wrapping). It:
    1. builds the EP process group (MVP: ``ep_size == world_size``);
    2. physically shards each MoE experts module so every rank keeps only
       ``num_experts / ep_size`` experts (this is where the real memory saving
       comes from -- each rank only stores/optimizes its own experts);
    3. marks the sharded expert parameters so the FSDP auto-wrap policy skips
       them (they are already partitioned across ranks and must NOT be
       re-sharded by FSDP);
    4. patches the MoE block forward with All-to-All dispatch/combine so each
       token is routed to the rank that owns its target expert.

Currently only DeepSeek-V4 (``DeepseekV4SparseMoeBlock`` + ``DeepseekV4Experts``)
is supported. The expert weights there are 3D grouped Parameters::

    gate_up_proj: [num_experts, 2 * intermediate, hidden]
    down_proj:    [num_experts, hidden, intermediate]

so sharding is a slice along dim 0 (the expert dimension).
"""
import types
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import init_device_mesh

from swift.model import get_llm_model
from swift.utils import get_device, get_dist_setting, get_logger
from .all_to_all import all_to_all_single

logger = get_logger()

# Attribute set on every expert parameter that has been physically sharded by EP.
# The FSDP auto-wrap policy must skip modules whose parameters carry this flag.
EP_SHARDED_FLAG = '_ep_sharded'


class ExpertParallel:

    _global_inited: bool = False

    def __init__(self):
        self.ep_size: Optional[int] = None
        self.device_mesh = None
        self.num_experts: Optional[int] = None
        self.num_local_experts: Optional[int] = None

    @property
    def ep_group(self):
        return self.device_mesh['expert'].get_group() if self.device_mesh else None

    @property
    def ep_rank(self) -> int:
        return dist.get_rank(self.ep_group) if self.device_mesh else 0

    def _init_device_mesh(self):
        rank, local_rank, world_size, local_world_size = get_dist_setting()
        # MVP: ep_size == world_size, so the whole world is a single EP group.
        assert self.ep_size == world_size, (
            f'expert_parallel currently only supports ep_size == world_size, '
            f'got ep_size={self.ep_size}, world_size={world_size}')
        self.device_mesh = init_device_mesh(
            get_device().split(':')[0], mesh_shape=(self.ep_size, ), mesh_dim_names=('expert', ))

    def prepare(self, ep_size: int, model: torch.nn.Module):
        """Shard experts and patch MoE forward. Call BEFORE FSDP wrapping."""
        self.ep_size = ep_size
        if not ExpertParallel._global_inited:
            self._init_device_mesh()
            ExpertParallel._global_inited = True

        llm_model = get_llm_model(model)
        num_moe_blocks = self._shard_and_patch(llm_model)
        if num_moe_blocks == 0:
            raise RuntimeError(
                'expert_parallel did not find any supported MoE block in the model. '
                'Currently only DeepSeek-V4 (DeepseekV4SparseMoeBlock) is supported.')

        # FSDP2 must skip the already-sharded expert params; patch it before wrapping.
        from .fsdp_patch import patch_fsdp_ignore_ep_params
        patch_fsdp_ignore_ep_params()
        logger.info(f'expert_parallel prepared: ep_size={self.ep_size}, '
                    f'num_experts={self.num_experts}, num_local_experts={self.num_local_experts}, '
                    f'patched_moe_blocks={num_moe_blocks}')

    def _shard_and_patch(self, llm_model: torch.nn.Module) -> int:
        num_patched = 0
        for module in llm_model.modules():
            if type(module).__name__ == 'DeepseekV4SparseMoeBlock':
                self._shard_deepseek_v4_experts(module.experts)
                self._patch_deepseek_v4_moe_forward(module)
                num_patched += 1
        return num_patched

    def _shard_deepseek_v4_experts(self, experts: torch.nn.Module):
        """Slice the 3D grouped expert Parameters along the expert dim (dim 0)."""
        num_experts = experts.num_experts
        assert num_experts % self.ep_size == 0, (
            f'num_experts ({num_experts}) must be divisible by ep_size ({self.ep_size})')
        num_local_experts = num_experts // self.ep_size
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts

        start = self.ep_rank * num_local_experts
        end = start + num_local_experts

        with torch.no_grad():
            gate_up = experts.gate_up_proj.data[start:end].clone()
            down = experts.down_proj.data[start:end].clone()

        # Replace the Parameters with the local shard. requires_grad follows the original.
        gate_up_param = torch.nn.Parameter(gate_up, requires_grad=experts.gate_up_proj.requires_grad)
        down_param = torch.nn.Parameter(down, requires_grad=experts.down_proj.requires_grad)
        setattr(gate_up_param, EP_SHARDED_FLAG, True)
        setattr(down_param, EP_SHARDED_FLAG, True)
        experts.gate_up_proj = gate_up_param
        experts.down_proj = down_param

        # The local module now only owns its own experts.
        experts.num_experts = num_local_experts
        experts._ep_num_local_experts = num_local_experts
        experts._ep_expert_offset = start

    def _patch_deepseek_v4_moe_forward(self, moe_block: torch.nn.Module):
        ep = self

        def ep_moe_forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor = None) -> torch.Tensor:
            batch, seq_len, hidden_dim = hidden_states.shape
            residual = hidden_states
            flat = hidden_states.view(-1, hidden_dim)
            if getattr(self, 'is_hash', False):
                _, weights, indices = self.gate(hidden_states, input_ids)
            else:
                _, weights, indices = self.gate(hidden_states)
            routed = ep._ep_experts_forward(self.experts, flat, indices, weights)
            routed = routed.view(batch, seq_len, hidden_dim)
            return routed + self.shared_experts(residual)

        moe_block.forward = types.MethodType(ep_moe_forward, moe_block)

    def _ep_experts_forward(self, experts: torch.nn.Module, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                            top_k_weights: torch.Tensor) -> torch.Tensor:
        """All-to-All dispatch -> local expert compute -> All-to-All combine.

        hidden_states: [num_tokens, hidden]
        top_k_index:   [num_tokens, top_k] global expert ids
        top_k_weights: [num_tokens, top_k]
        """
        group = self.ep_group
        num_local_experts = self.num_local_experts
        num_tokens = hidden_states.shape[0]
        top_k = top_k_index.shape[1]
        device = hidden_states.device

        # Flatten the (token, top_k) pairs; each pair is an independent routed slot.
        flat_expert_ids = top_k_index.reshape(-1)  # [num_slots]
        flat_weights = top_k_weights.reshape(-1)  # [num_slots]
        # Which rank owns the expert each slot is routed to.
        dest_rank = flat_expert_ids // num_local_experts  # [num_slots]

        # Sort slots by destination rank so they form contiguous send buffers.
        sort_order = torch.argsort(dest_rank, stable=True)
        sorted_dest = dest_rank[sort_order]
        sorted_expert_ids = flat_expert_ids[sort_order]
        sorted_weights = flat_weights[sort_order]
        # The token index each slot belongs to (used to scatter results back).
        slot_token_idx = (torch.arange(num_tokens, device=device).repeat_interleave(top_k))[sort_order]

        # How many rows we send to each rank, and (after exchange) how many we receive.
        input_split_sizes = torch.bincount(sorted_dest, minlength=self.ep_size).tolist()
        output_split_sizes = self._exchange_split_sizes(group, input_split_sizes, device)

        # Dispatch: send each slot's hidden state to the rank owning its expert.
        send_hidden = hidden_states.index_select(0, slot_token_idx)
        send_local_expert = (sorted_expert_ids % num_local_experts).long()

        recv_hidden = all_to_all_single(group, send_hidden, output_split_sizes, input_split_sizes)
        recv_local_expert = self._all_to_all_meta(group, send_local_expert, output_split_sizes, input_split_sizes)

        # Local expert computation on the received tokens.
        recv_out = self._compute_local_experts(experts, recv_hidden, recv_local_expert)

        # Combine: send the expert outputs back to the source ranks (split sizes swapped).
        combined = all_to_all_single(group, recv_out, input_split_sizes, output_split_sizes)

        # Weight each slot output and scatter-add back to the per-token result.
        combined = combined * sorted_weights.unsqueeze(-1).to(combined.dtype)
        final = torch.zeros_like(hidden_states)
        final.index_add_(0, slot_token_idx, combined.to(final.dtype))
        return final

    @staticmethod
    def _exchange_split_sizes(group, input_split_sizes, device):
        """Tell every rank how many rows it will receive from each peer."""
        send = torch.tensor(input_split_sizes, device=device, dtype=torch.long)
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        return recv.tolist()

    @staticmethod
    def _all_to_all_meta(group, tensor_1d, output_split_sizes, input_split_sizes):
        """All-to-All for an integer metadata tensor (no autograd needed)."""
        tensor_1d = tensor_1d.contiguous()
        output = tensor_1d.new_empty([sum(output_split_sizes)])
        dist.all_to_all_single(
            output,
            tensor_1d,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group)
        return output

    @staticmethod
    def _compute_local_experts(experts: torch.nn.Module, hidden_states: torch.Tensor,
                               local_expert_ids: torch.Tensor) -> torch.Tensor:
        """Run the local experts on received tokens, grouped by local expert id.

        Mirrors ``DeepseekV4Experts.forward`` math but indexed by local expert id.
        """
        out = torch.zeros_like(hidden_states)
        num_local_experts = experts._ep_num_local_experts
        for local_idx in range(num_local_experts):
            token_mask = local_expert_ids == local_idx
            if not torch.any(token_mask):
                continue
            token_idx = token_mask.nonzero(as_tuple=True)[0]
            cur = F.linear(hidden_states[token_idx], experts.gate_up_proj[local_idx])
            cur = experts._apply_gate(cur)
            cur = F.linear(cur, experts.down_proj[local_idx])
            out.index_copy_(0, token_idx, cur.to(out.dtype))
        return out


expert_parallel = ExpertParallel()
