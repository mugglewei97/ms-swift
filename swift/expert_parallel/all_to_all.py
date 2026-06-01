# Copyright (c) ModelScope Contributors. All rights reserved.
"""All-to-All primitives for expert parallel (EP).

The dispatch (scatter tokens to the rank that owns the target expert) and the
combine (gather expert outputs back to the source rank) are symmetric: the
forward of one is the backward of the other. Wrapping the collective in an
autograd ``Function`` lets gradients flow through the dispatch/combine round
trip exactly as they would on a single device.
"""
from typing import Any, List, Tuple

import torch
import torch.distributed as dist


class _AllToAllSingle(torch.autograd.Function):
    """All-to-All over dim 0 with uneven, per-rank split sizes.

    Args:
        group: the expert-parallel process group.
        input: a 2D tensor ``[num_rows, hidden]`` to redistribute.
        output_split_sizes: number of rows this rank receives from every rank.
        input_split_sizes: number of rows this rank sends to every rank.

    The backward swaps ``input_split_sizes`` and ``output_split_sizes`` so the
    gradient of the dispatch is exactly the combine (and vice versa).
    """

    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: torch.Tensor,
        output_split_sizes: List[int],
        input_split_sizes: List[int],
    ) -> torch.Tensor:
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        input = input.contiguous()
        output = input.new_empty([sum(output_split_sizes)] + list(input.shape[1:]))
        dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None, None]:
        grad_output = grad_output.contiguous()
        # The backward redistribution is the mirror of the forward: rows that were
        # sent are now received and vice versa, so the split sizes are swapped.
        grad_input = grad_output.new_empty([sum(ctx.input_split_sizes)] + list(grad_output.shape[1:]))
        dist.all_to_all_single(
            grad_input,
            grad_output,
            output_split_sizes=ctx.input_split_sizes,
            input_split_sizes=ctx.output_split_sizes,
            group=ctx.group)
        return None, grad_input, None, None


def all_to_all_single(
    group: dist.ProcessGroup,
    input: torch.Tensor,
    output_split_sizes: List[int],
    input_split_sizes: List[int],
) -> torch.Tensor:
    """Autograd-aware All-to-All with uneven split sizes along dim 0."""
    return _AllToAllSingle.apply(group, input, output_split_sizes, input_split_sizes)
