import torch
import triton

from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
from liger_kernel.ops.utils import (
    amp_custom_bwd,
    amp_custom_fwd,
    element_mul_kernel,
    is_hip,
)

MAX_FUSED_SIZE = 65536 // 2


def _flce_forward(_input, weight, target, ignore_index, chunk_size):
    BT, H = _input.shape
    V = weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    num_chunks = triton.cdiv(BT, chunk_size)
    num_warps = 32 if not is_hip() else 16

    requires_grad = _input.requires_grad
    grad_input = torch.zeros_like(_input) if requires_grad else None
    grad_weight = (
        torch.zeros_like(weight, dtype=torch.float32)
        if requires_grad and weight.requires_grad
        else None
    )
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=_input.device)

    total_n_non_ignore = int((target != ignore_index).sum().item())

    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = min(start + chunk_size, BT)
        input_chunk = _input[start:end]  # [rows, H]
        logits_chunk = (input_chunk @ weight.t()).contiguous()  # [rows, V]
        target_chunk = target[start:end].contiguous()
        loss_slice = loss_1d[start:end]

        liger_cross_entropy_kernel[(end - start,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),
            weight_ptr=None,
            loss_ptr=loss_slice,
            z_loss_ptr=None,
            loss_stride=loss_slice.stride(-1),
            token_accuracy_ptr=None,
            token_accuracy_stride=0,
            predicted_tokens_ptr=None,
            predicted_tokens_stride=0,
            n_cols=V,
            n_non_ignore=total_n_non_ignore,
            sum_non_ignore_weight=total_n_non_ignore,
            weight_sum=0.0,
            ignore_index=ignore_index,
            lse_square_scale=0.0,
            label_smoothing=0.0,
            reduction="mean",
            softcap=None,
            RETURN_Z_LOSS=False,
            RETURN_TOKEN_ACCURACY=False,
            RETURN_PREDICTED_TOKENS=False,
            HAS_WEIGHT=False,
            HAS_SOFTCAPPING=False,
            HAS_GRADIENTS=requires_grad,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        if requires_grad:
            grad_logits_chunk = logits_chunk
            grad_input[start:end] = grad_logits_chunk @ weight
            if grad_weight is not None:
                grad_weight += torch.mm(grad_logits_chunk.t(), input_chunk).float()

    loss = torch.sum(loss_1d)
    if grad_weight is not None:
        grad_weight = grad_weight.to(weight.dtype)
    return loss, grad_input, grad_weight


class FusedLinearCrossEntropyLargeChunk(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    def forward(ctx, _input, weight, target, ignore_index, chunk_size):
        loss, grad_input, grad_weight = _flce_forward(
            _input, weight, target, ignore_index, chunk_size
        )
        ctx.save_for_backward(
            grad_input.detach() if grad_input is not None else None,
            grad_weight.detach() if grad_weight is not None else None,
        )
        return loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        num_warps = 32 if not is_hip() else 16

        if grad_input is not None:
            BT, H = grad_input.shape
            BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))
            element_mul_kernel[(BT,)](
                grad_input,
                grad_input.stride(-2),
                grad_output,
                H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
            if grad_weight is not None:
                V, _ = grad_weight.shape
                element_mul_kernel[(V,)](
                    grad_weight,
                    grad_weight.stride(-2),
                    grad_output,
                    H,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=num_warps,
                )
        return grad_input, grad_weight, None, None, None


def flce_large_chunk(h_flat, weight, targets_flat, ignore_index, chunk_size):
    return FusedLinearCrossEntropyLargeChunk.apply(
        h_flat, weight, targets_flat, ignore_index, chunk_size
    )
