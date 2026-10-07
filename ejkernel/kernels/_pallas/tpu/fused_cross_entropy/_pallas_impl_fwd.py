# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TPU Pallas fused cross-entropy kernels.

The replicated-vocab path streams logits from HBM into VMEM with 2D BlockSpec
double-buffering, computes sparse cross-entropy row by row in a single online
softmax pass, and saves global ``lse`` for the analytic backward in
``_pallas_impl_bwd``.

The vocab-parallel path is designed for ``shard_map`` with a partition spec
like ``P((dp, fsdp), sp, tp)``. Each TP shard runs a hardware-pipelined 2D
Pallas online-softmax stats kernel over ``(M // block_m, V_local // block_v)``,
and fused JAX collectives merge row-wise softmax statistics across ``tp``.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ejkernel.kernels._xla.fused_cross_entropy._xla_impl_fwd import fused_cross_entropy as _xla_fused_ce

from ._pallas_impl_bwd import _ce_bwd_pallas


def _default_block_v(vocab_size: int) -> int:
    """Choose the default TPU vocab tile width for CE forward kernels."""
    if vocab_size <= 256:
        return 256
    if vocab_size <= 1024:
        return 1024
    if vocab_size <= 4096:
        return 4096
    if vocab_size <= 16384:
        return 4096
    if vocab_size <= 65536:
        return 4096
    return 4096


def _default_block_m() -> int:
    """Return the row tile size used by the sparse-row TPU CE kernels."""
    return 256


def _pallas_out_shape(shape: tuple[int, ...], dtype: jnp.dtype) -> jax.ShapeDtypeStruct:
    """Build a Pallas ``out_shape`` entry that varies over the active manual mesh axes."""
    abstract_mesh = jax.sharding.get_abstract_mesh()
    axis_sizes = getattr(abstract_mesh, "axis_sizes", None) or getattr(abstract_mesh, "shape", {})
    axis_size = getattr(axis_sizes, "get", lambda _axis, default: default)
    manual_axes = frozenset(axis for axis in abstract_mesh.manual_axes if int(axis_size(axis, 1)) > 1)
    return jax.ShapeDtypeStruct(
        shape,
        dtype,
        manual_axis_type=jax.sharding.ManualAxisType(varying=manual_axes),
    )


def _pad_rows_2d(x: jax.Array, pad_rows: int, pad_value: float = 0.0) -> jax.Array:
    """Pad a rank-2 tensor along rows so Pallas grid blocks are rectangular."""
    if pad_rows == 0:
        return x
    return jnp.pad(x, ((0, pad_rows), (0, 0)), constant_values=pad_value)


def _pad_rows_1d(x: jax.Array, pad_rows: int, pad_value: float = 0.0) -> jax.Array:
    """Pad a rank-1 row tensor with the value expected by the caller."""
    if pad_rows == 0:
        return x
    return jnp.pad(x, (0, pad_rows), constant_values=pad_value)


def _flatten_logits(logits: jax.Array) -> tuple[jax.Array, tuple[int, ...]]:
    """Flatten leading dimensions into rows while preserving output shape."""
    if logits.ndim < 2:
        raise ValueError(f"fused_cross_entropy expects rank>=2 logits; got shape {logits.shape}")
    return logits.reshape(-1, logits.shape[-1]), logits.shape[:-1]


def _per_token_weights(targets: jax.Array, weights: jax.Array | None, ignore_index: int) -> jax.Array:
    """Return float32 row weights, deriving them from ``ignore_index`` if absent."""
    if weights is None:
        return (targets != ignore_index).astype(jnp.float32)
    return weights.astype(jnp.float32)


def _online_row_stats_2d_kernel(
    x_ref,
    lbl_ref,
    m_out_ref,
    d_out_ref,
    s_out_ref,
    m_scratch,
    d_scratch,
    s_scratch,
    *,
    bk: int,
):
    """Hardware-pipelined 2D BlockSpec online-softmax + target gather kernel.

    Streams ``(bm, bk)`` tiles across ``grid = (M // bm, V // bk)`` with
    ``dimension_semantics=("parallel", "arbitrary")`` so Mosaic double-buffers
    HBM-to-VMEM DMA transfers while the VPU computes online ``max``, ``sum_exp``,
    and target logit extraction in a single HBM pass.
    """
    j = pl.program_id(1)

    @pl.when(j == 0)
    def _init():
        m_scratch[...] = jnp.full_like(m_scratch, -1e30)
        d_scratch[...] = jnp.zeros_like(d_scratch)
        s_scratch[...] = jnp.zeros_like(s_scratch)

    x_tile = x_ref[...]

    col_start = j * bk
    target_offset = lbl_ref[...] - col_start
    cols = jax.lax.broadcasted_iota(jnp.int32, (1, bk), 1)
    is_target = cols == target_offset
    matched_val = jnp.where(is_target, x_tile, jnp.zeros_like(x_tile))
    tile_sel = jnp.sum(matched_val.astype(jnp.float32), axis=-1, keepdims=True)
    s_scratch[...] = s_scratch[...] + tile_sel

    tile_max = jnp.max(x_tile, axis=-1, keepdims=True).astype(jnp.float32)
    old_m = m_scratch[...]
    new_m = jnp.maximum(old_m, tile_max)
    alpha = jnp.where(old_m <= -1e29, 0.0, jnp.exp(old_m - new_m))
    tile_sum = jnp.sum(jnp.exp(x_tile.astype(jnp.float32) - new_m), axis=-1, keepdims=True)
    d_scratch[...] = d_scratch[...] * alpha + tile_sum
    m_scratch[...] = new_m

    @pl.when(j == pl.num_programs(1) - 1)
    def _store():
        m_out_ref[...] = m_scratch[...]
        d_out_ref[...] = d_scratch[...]
        s_out_ref[...] = s_scratch[...]


def _run_online_row_stats_2d(
    logits_2d: jax.Array,
    local_targets_1d: jax.Array,
    *,
    block_m: int,
    block_v: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run the 2D BlockSpec-pipelined online-softmax statistics kernel."""
    m, k = logits_2d.shape
    bm = min(int(block_m), m) if m >= 128 else int(block_m)
    bk = min(int(block_v), k) if k >= 128 else int(block_v)

    pad_m = (bm - (m % bm)) % bm
    pad_k = (bk - (k % bk)) % bk
    if pad_m > 0 or pad_k > 0:
        x_pad = jnp.pad(logits_2d, ((0, pad_m), (0, pad_k)), constant_values=-1e30)
    else:
        x_pad = logits_2d

    if pad_m > 0:
        lbl_pad = jnp.pad(local_targets_1d[:, None], ((0, pad_m), (0, 0)), constant_values=-1)
    else:
        lbl_pad = local_targets_1d[:, None]

    m_pad, k_pad = x_pad.shape
    grid = (m_pad // bm, k_pad // bk)

    m_out, d_out, s_out = pl.pallas_call(
        functools.partial(_online_row_stats_2d_kernel, bk=bk),
        out_shape=(
            _pallas_out_shape((m_pad, 1), jnp.float32),
            _pallas_out_shape((m_pad, 1), jnp.float32),
            _pallas_out_shape((m_pad, 1), jnp.float32),
        ),
        grid=grid,
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j: (i, j)),
            pl.BlockSpec((bm, 1), lambda i, j: (i, 0)),
        ],
        out_specs=[
            pl.BlockSpec((bm, 1), lambda i, j: (i, 0)),
            pl.BlockSpec((bm, 1), lambda i, j: (i, 0)),
            pl.BlockSpec((bm, 1), lambda i, j: (i, 0)),
        ],
        scratch_shapes=[
            pltpu.VMEM((bm, 1), jnp.float32),
            pltpu.VMEM((bm, 1), jnp.float32),
            pltpu.VMEM((bm, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=96 * 1024 * 1024,
        ),
    )(x_pad, lbl_pad.astype(jnp.int32))

    return m_out[:m, 0], d_out[:m, 0], s_out[:m, 0]


def _copy_hbm_to_vmem(src_ref, dst_ref, sem_ref, row_start, col_start: int, block_m: int, size: int):
    """DMA-copy one ``(rows, vocab)`` tile from HBM into VMEM scratch."""
    copy = pltpu.make_async_copy(
        src_ref=src_ref.at[pl.ds(row_start, block_m), pl.ds(col_start, size)],
        dst_ref=dst_ref.at[pl.ds(0, block_m), pl.ds(0, size)],
        sem=sem_ref.at[0],
    )
    copy.start()
    copy.wait()


def _copy_rows_hbm_to_vmem(src_ref, dst_ref, sem_ref, row_start, block_m: int):
    """DMA-copy one row-vector block from HBM into VMEM scratch."""
    copy = pltpu.make_async_copy(
        src_ref=src_ref.at[pl.ds(row_start, block_m)],
        dst_ref=dst_ref.at[pl.ds(0, block_m)],
        sem=sem_ref.at[0],
    )
    copy.start()
    return copy


def _ce_fwd_kernel(
    logits_ref,
    targets_ref,
    weights_ref,
    loss_ref,
    lse_ref,
    logits_tile_ref,
    target_ref,
    weight_ref,
    dma_sem_ref,
    *,
    ignore_index: int,
    label_smoothing: float,
    z_loss: float,
    normalizing_constant: float,
    block_v: int,
    block_m: int,
):
    """Fallback replicated-vocab CE forward kernel when label_smoothing > 0."""
    row_start = pl.program_id(0) * block_m
    _, vocab_size = logits_ref.shape
    offsets = jnp.arange(block_v)
    row_offsets = jnp.arange(block_m)
    row_active = (row_start + row_offsets) < logits_ref.shape[0]
    cc1 = _copy_rows_hbm_to_vmem(targets_ref, target_ref, dma_sem_ref, row_start, block_m)
    cc2 = _copy_rows_hbm_to_vmem(weights_ref, weight_ref, dma_sem_ref, row_start, block_m)
    cc1.wait()
    cc2.wait()
    target = target_ref[...].astype(jnp.int32)
    safe_target = jnp.clip(target, 0, vocab_size - 1)
    weight = weight_ref[...].astype(jnp.float32)
    weight_abs = jnp.abs(weight)
    valid = row_active & (target != ignore_index) & (weight != 0.0)

    loss_ref[...] = jnp.zeros((block_m,), dtype=jnp.float32)
    lse_ref[...] = jnp.zeros((block_m,), dtype=jnp.float32)

    @pl.when(jnp.any(valid))
    def _compute_active_block():
        max_val = jnp.full((block_m,), -jnp.inf, dtype=jnp.float32)
        sum_exp = jnp.zeros((block_m,), dtype=jnp.float32)
        sum_logits = jnp.zeros((block_m,), dtype=jnp.float32)
        target_logit = jnp.zeros((block_m,), dtype=jnp.float32)
        num_blocks = pl.cdiv(vocab_size, block_v)

        for block_idx in range(num_blocks):
            start = block_idx * block_v
            size = min(block_v, vocab_size - start)
            _copy_hbm_to_vmem(logits_ref, logits_tile_ref, dma_sem_ref, row_start, start, block_m, size)
            tile = logits_tile_ref[...].astype(jnp.float32)
            vocab_idx = start + offsets
            in_vocab = offsets < size
            masked = jnp.where(in_vocab[None, :], tile, -jnp.inf)
            tile_max = jnp.max(masked, axis=1)
            new_max = jnp.maximum(max_val, tile_max)
            correction = jnp.exp(max_val - new_max)
            tile_sum_exp = jnp.sum(jnp.where(in_vocab[None, :], jnp.exp(tile - new_max[:, None]), 0.0), axis=1)
            sum_exp = sum_exp * correction + tile_sum_exp
            max_val = new_max
            sum_logits = sum_logits + jnp.sum(jnp.where(in_vocab[None, :], tile, 0.0), axis=1)
            target_logit = target_logit + jnp.sum(
                jnp.where(in_vocab[None, :] & (vocab_idx[None, :] == safe_target[:, None]), tile, 0.0),
                axis=1,
            )

        lse = jnp.log(sum_exp) + max_val
        confidence = 1.0 - float(label_smoothing)
        low_conf = float(label_smoothing) / float(vocab_size - 1) if vocab_size > 1 and label_smoothing > 0.0 else 0.0
        eff_target_w = confidence - low_conf
        base = lse - eff_target_w * target_logit - low_conf * sum_logits - float(normalizing_constant)
        per_row = weight_abs * (base + float(z_loss) * lse * lse)

        loss_ref[...] = jnp.where(valid, per_row, 0.0).astype(jnp.float32)
        lse_ref[...] = jnp.where(row_active, lse, 0.0).astype(jnp.float32)


def _ce_fwd_pallas(logits_2d, targets_1d, weights_1d, *, ignore_index, label_smoothing, z_loss, block_v, block_m):
    """Launch replicated-vocab TPU Pallas CE forward and trim padded rows."""
    n_rows, vocab_size = logits_2d.shape
    if float(label_smoothing) == 0.0:
        targets_i32 = targets_1d.astype(jnp.int32)
        weights_f32 = weights_1d.astype(jnp.float32)
        valid = (targets_i32 != int(ignore_index)) & (weights_f32 != 0.0)
        safe_targets = jnp.where(valid, targets_i32, -1)
        m_out, d_out, s_out = _run_online_row_stats_2d(
            logits_2d,
            safe_targets,
            block_m=int(block_m),
            block_v=int(block_v),
        )
        lse = m_out + jnp.log(d_out)
        base = lse - s_out
        if float(z_loss) != 0.0:
            base = base + float(z_loss) * lse * lse
        loss = jnp.where(valid, jnp.abs(weights_f32) * base, 0.0).astype(jnp.float32)
        return loss, lse.astype(jnp.float32)

    n_rows_pad = pl.cdiv(n_rows, int(block_m)) * int(block_m)
    pad_rows = n_rows_pad - n_rows
    logits_pad = _pad_rows_2d(logits_2d, pad_rows)
    targets_pad = _pad_rows_1d(targets_1d, pad_rows, ignore_index)
    weights_pad = _pad_rows_1d(weights_1d, pad_rows)
    confidence = 1.0 - float(label_smoothing)
    low_conf = float(label_smoothing) / float(vocab_size - 1) if vocab_size > 1 and label_smoothing > 0.0 else 0.0
    normalizing_constant = 0.0
    if label_smoothing > 0.0:
        normalizing_constant = -(
            confidence * math.log(max(confidence, 1e-20)) + (vocab_size - 1) * low_conf * math.log(max(low_conf, 1e-20))
        )
    loss, lse = pl.pallas_call(
        functools.partial(
            _ce_fwd_kernel,
            ignore_index=int(ignore_index),
            label_smoothing=float(label_smoothing),
            z_loss=float(z_loss),
            normalizing_constant=float(normalizing_constant),
            block_v=int(block_v),
            block_m=int(block_m),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=[
                pl.BlockSpec((int(block_m),), lambda row_block: (row_block,)),
                pl.BlockSpec((int(block_m),), lambda row_block: (row_block,)),
            ],
            scratch_shapes=[
                pltpu.VMEM((int(block_m), int(block_v)), logits_2d.dtype),
                pltpu.VMEM((int(block_m),), targets_1d.dtype),
                pltpu.VMEM((int(block_m),), weights_1d.dtype),
                pltpu.SemaphoreType.DMA((1,)),
            ],
            grid=(n_rows_pad // int(block_m),),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        out_shape=[
            _pallas_out_shape((n_rows_pad,), jnp.float32),
            _pallas_out_shape((n_rows_pad,), jnp.float32),
        ],
    )(logits_pad, targets_pad.astype(jnp.int32), weights_pad.astype(jnp.float32))
    return loss[:n_rows], lse[:n_rows]


def _ce_tp_stats_kernel(
    logits_ref,
    targets_ref,
    weights_ref,
    max_ref,
    sum_exp_ref,
    target_logit_ref,
    sum_logits_ref,
    logits_tile_ref,
    target_ref,
    weight_ref,
    dma_sem_ref,
    *,
    ignore_index: int,
    block_v: int,
    block_m: int,
):
    """Fallback 1D local-vocab CE statistics kernel."""
    row_start = pl.program_id(0) * block_m
    _, local_vocab_size = logits_ref.shape
    offsets = jnp.arange(block_v)
    row_offsets = jnp.arange(block_m)
    row_active = (row_start + row_offsets) < logits_ref.shape[0]
    cc1 = _copy_rows_hbm_to_vmem(targets_ref, target_ref, dma_sem_ref, row_start, block_m)
    cc2 = _copy_rows_hbm_to_vmem(weights_ref, weight_ref, dma_sem_ref, row_start, block_m)
    cc1.wait()
    cc2.wait()
    target = target_ref[...].astype(jnp.int32)
    weight = weight_ref[...].astype(jnp.float32)
    valid = row_active & (target != ignore_index) & (weight != 0.0)

    max_ref[...] = jnp.full((block_m,), -jnp.inf, dtype=jnp.float32)
    sum_exp_ref[...] = jnp.zeros((block_m,), dtype=jnp.float32)
    target_logit_ref[...] = jnp.zeros((block_m,), dtype=jnp.float32)
    sum_logits_ref[...] = jnp.zeros((block_m,), dtype=jnp.float32)

    @pl.when(jnp.any(valid))
    def _compute_active_block():
        max_val = jnp.full((block_m,), -jnp.inf, dtype=jnp.float32)
        sum_exp = jnp.zeros((block_m,), dtype=jnp.float32)
        sum_logits = jnp.zeros((block_m,), dtype=jnp.float32)
        target_logit = jnp.zeros((block_m,), dtype=jnp.float32)
        num_blocks = pl.cdiv(local_vocab_size, block_v)

        for block_idx in range(num_blocks):
            start = block_idx * block_v
            size = min(block_v, local_vocab_size - start)
            _copy_hbm_to_vmem(logits_ref, logits_tile_ref, dma_sem_ref, row_start, start, block_m, size)
            tile = logits_tile_ref[...].astype(jnp.float32)
            local_vocab_idx = start + offsets
            in_vocab = offsets < size
            masked = jnp.where(in_vocab[None, :], tile, -jnp.inf)
            tile_max = jnp.max(masked, axis=1)
            new_max = jnp.maximum(max_val, tile_max)
            correction = jnp.exp(max_val - new_max)
            tile_sum_exp = jnp.sum(
                jnp.where(in_vocab[None, :], jnp.exp(tile - new_max[:, None]), 0.0),
                axis=1,
            )
            sum_exp = sum_exp * correction + tile_sum_exp
            max_val = new_max
            sum_logits = sum_logits + jnp.sum(jnp.where(in_vocab[None, :], tile, 0.0), axis=1)
            target_logit = target_logit + jnp.sum(
                jnp.where(in_vocab[None, :] & (local_vocab_idx[None, :] == target[:, None]), tile, 0.0),
                axis=1,
            )

        max_ref[...] = jnp.where(valid, max_val, -jnp.inf).astype(jnp.float32)
        sum_exp_ref[...] = jnp.where(valid, sum_exp, 0.0).astype(jnp.float32)
        target_logit_ref[...] = jnp.where(valid, target_logit, 0.0).astype(jnp.float32)
        sum_logits_ref[...] = jnp.where(valid, sum_logits, 0.0).astype(jnp.float32)


def _ce_tp_stats_pallas(logits_2d, targets_1d, weights_1d, *, ignore_index, block_v, block_m):
    """Launch the 2D BlockSpec-pipelined local-vocab CE stats kernel used by the TP path.

    Returns per-row ``(local_max, local_sum_exp, local_target_logit,
    local_sum_logits)``.
    """
    n_rows = logits_2d.shape[0]
    targets_i32 = targets_1d.astype(jnp.int32)
    weights_f32 = weights_1d.astype(jnp.float32)
    valid = (targets_i32 != int(ignore_index)) & (weights_f32 != 0.0)
    safe_targets = jnp.where(valid, targets_i32, -1)

    max_val, sum_exp, target_logit = _run_online_row_stats_2d(
        logits_2d,
        safe_targets,
        block_m=int(block_m),
        block_v=int(block_v),
    )
    max_val = jnp.where(valid, max_val, -jnp.inf).astype(jnp.float32)
    sum_exp = jnp.where(valid, sum_exp, 0.0).astype(jnp.float32)
    target_logit = jnp.where(valid, target_logit, 0.0).astype(jnp.float32)
    sum_logits = jnp.zeros((n_rows,), dtype=jnp.float32)
    return max_val, sum_exp, target_logit, sum_logits


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def _fused_ce_loss_pallas(logits_2d, targets_1d, weights_1d, ignore_index, label_smoothing, z_loss, block_v, block_m):
    """Differentiable replicated-vocab CE loss wrapper."""
    loss, _lse = _ce_fwd_pallas(
        logits_2d,
        targets_1d,
        weights_1d,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        block_v=block_v,
        block_m=block_m,
    )
    return loss


def _ce_loss_fwd(logits_2d, targets_1d, weights_1d, ignore_index, label_smoothing, z_loss, block_v, block_m):
    """Forward rule for replicated-vocab CE, saving tensors for backward."""
    loss, lse = _ce_fwd_pallas(
        logits_2d,
        targets_1d,
        weights_1d,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        block_v=block_v,
        block_m=block_m,
    )
    return loss, (logits_2d, lse, targets_1d, weights_1d)


def _ce_loss_bwd(ignore_index, label_smoothing, z_loss, block_v, block_m, residual, dy):
    """Backward rule for replicated-vocab CE using the saved global LSE."""
    logits_2d, lse, targets_1d, weights_1d = residual
    dlogits = _ce_bwd_pallas(
        logits_2d,
        lse,
        targets_1d,
        weights_1d,
        dy,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        block_v=block_v,
        block_m=block_m,
    )
    return dlogits, None, None


_fused_ce_loss_pallas.defvjp(_ce_loss_fwd, _ce_loss_bwd)


def _ce_tp_loss_and_lse(
    logits_2d,
    targets_1d,
    weights_1d,
    *,
    ignore_index,
    label_smoothing,
    z_loss,
    block_v,
    block_m,
    vocab_parallel_axis,
):
    """Build TP-vocab sparse CE loss from 2D-pipelined Pallas stats plus fused collectives."""
    n_rows, local_vocab_size = int(logits_2d.shape[0]), int(logits_2d.shape[-1])
    axis_idx = jax.lax.axis_index(vocab_parallel_axis)
    vocab_start = axis_idx * local_vocab_size
    targets_i32 = targets_1d.astype(jnp.int32)
    weights_f32 = weights_1d.astype(jnp.float32)
    valid = (targets_i32 != int(ignore_index)) & (weights_f32 != 0.0)
    local_targets = jnp.where(
        valid,
        targets_i32 - vocab_start,
        jnp.array(int(ignore_index), dtype=jnp.int32),
    )

    local_max, local_sum_exp, local_target_logit = _run_online_row_stats_2d(
        logits_2d,
        jnp.where(valid, local_targets, -1),
        block_m=int(block_m),
        block_v=int(block_v),
    )

    if n_rows >= 8192:
        # Single-barrier AllGather of packed (3, M) local statistics for large row counts.
        stats = jnp.stack([local_max, local_sum_exp, local_target_logit], axis=0)
        gathered = jax.lax.all_gather(stats, vocab_parallel_axis)
        all_m = gathered[:, 0, :]
        all_d = gathered[:, 1, :]
        all_s = gathered[:, 2, :]
        global_max = jnp.max(all_m, axis=0)
        global_sum_exp = jnp.sum(all_d * jnp.exp(all_m - global_max), axis=0)
        target_logit = jnp.sum(all_s, axis=0)
    else:
        # Two-barrier reduction with packed (2, M) psum for scaled_sum and target_logit.
        global_max = jax.lax.pmax(local_max, vocab_parallel_axis)
        scaled_sum = local_sum_exp * jnp.exp(local_max - global_max)
        packed = jnp.stack([scaled_sum, local_target_logit], axis=0)
        summed = jax.lax.psum(packed, vocab_parallel_axis)
        global_sum_exp = summed[0]
        target_logit = summed[1]

    lse = jnp.where(valid, global_max + jnp.log(global_sum_exp), 0.0)
    base = lse - target_logit
    if float(z_loss) != 0.0:
        base = base + float(z_loss) * lse * lse
    per_row = jnp.where(valid, jnp.abs(weights_f32) * base, 0.0)
    return per_row.astype(jnp.float32), lse.astype(jnp.float32), local_targets.astype(jnp.int32)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8))
def _fused_ce_loss_pallas_tp(
    logits_2d,
    targets_1d,
    weights_1d,
    ignore_index,
    label_smoothing,
    z_loss,
    block_v,
    block_m,
    vocab_parallel_axis,
):
    """Differentiable TP-vocab CE loss wrapper backed by Pallas kernels."""
    loss, _lse, _local_targets = _ce_tp_loss_and_lse(
        logits_2d,
        targets_1d,
        weights_1d,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        block_v=block_v,
        block_m=block_m,
        vocab_parallel_axis=vocab_parallel_axis,
    )
    return loss


def _ce_loss_tp_fwd(
    logits_2d,
    targets_1d,
    weights_1d,
    ignore_index,
    label_smoothing,
    z_loss,
    block_v,
    block_m,
    vocab_parallel_axis,
):
    """Forward rule for TP-vocab CE, saving global LSE and local targets."""
    loss, lse, local_targets = _ce_tp_loss_and_lse(
        logits_2d,
        targets_1d,
        weights_1d,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        block_v=block_v,
        block_m=block_m,
        vocab_parallel_axis=vocab_parallel_axis,
    )
    return loss, (logits_2d, lse, local_targets, weights_1d)


def _ce_loss_tp_bwd(ignore_index, label_smoothing, z_loss, block_v, block_m, vocab_parallel_axis, residual, dy):
    """Backward rule for TP-vocab CE."""
    logits_2d, lse, local_targets, weights_1d = residual
    dlogits = _ce_bwd_pallas(
        logits_2d,
        lse,
        local_targets,
        weights_1d,
        dy,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        block_v=block_v,
        block_m=block_m,
    )
    return dlogits, None, None


_fused_ce_loss_pallas_tp.defvjp(_ce_loss_tp_fwd, _ce_loss_tp_bwd)


def _ce_correct_jax(logits_2d, targets_1d, weights_1d, ignore_index: int):
    """Compute sparse per-row accuracy outside the loss Pallas kernel."""
    preds = jnp.argmax(logits_2d.astype(jnp.float32), axis=-1).astype(jnp.int32)
    active = (targets_1d != int(ignore_index)) & (weights_1d != 0.0)
    return jnp.where(active & (preds == targets_1d), 1.0, 0.0).astype(jnp.float32)


def fused_cross_entropy_pallas(
    logits,
    targets=None,
    weights=None,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    soft_targets=None,
    reduction: str = "mean",
    vocab_parallel_axis: str | None = None,
    block_v: int = 0,
    block_m: int = 0,
):
    """Run TPU Pallas fused sparse cross-entropy."""
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"Invalid reduction '{reduction}'; expected one of none/sum/mean.")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1); got {label_smoothing}")
    if z_loss < 0.0:
        raise ValueError(f"z_loss must be non-negative; got {z_loss}")
    if soft_targets is not None:
        return _xla_fused_ce(
            logits,
            targets,
            weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            z_loss=z_loss,
            soft_targets=soft_targets,
            reduction=reduction,
            vocab_parallel_axis=vocab_parallel_axis,
        )
    if vocab_parallel_axis is not None and label_smoothing > 0.0:
        raise NotImplementedError("TPU Pallas vocab-parallel fused_cross_entropy does not support label_smoothing yet.")
    if targets is None:
        raise ValueError("either `targets` (sparse) or `soft_targets` (dense) must be provided.")
    if targets.shape != logits.shape[:-1]:
        raise ValueError(
            f"fused_cross_entropy: targets.shape={targets.shape} must equal logits.shape[:-1]={logits.shape[:-1]}"
        )
    if weights is not None and weights.shape != targets.shape:
        raise ValueError(f"fused_cross_entropy: weights.shape={weights.shape} must equal targets.shape={targets.shape}")

    flat_logits, leading = _flatten_logits(logits)
    flat_targets = targets.reshape(-1).astype(jnp.int32)
    flat_weights = _per_token_weights(flat_targets, None if weights is None else weights.reshape(-1), ignore_index)
    default_bv = _default_block_v(int(flat_logits.shape[-1]))
    bv = max(default_bv, int(block_v)) if int(block_v) > 0 else default_bv
    bm = int(block_m) if int(block_m) > 0 else _default_block_m()

    if vocab_parallel_axis is None:
        per_row = _fused_ce_loss_pallas(
            flat_logits,
            flat_targets,
            flat_weights,
            int(ignore_index),
            float(label_smoothing),
            float(z_loss),
            bv,
            bm,
        )
        correct = _ce_correct_jax(flat_logits, flat_targets, flat_weights, int(ignore_index))
    else:
        per_row = _fused_ce_loss_pallas_tp(
            flat_logits,
            flat_targets,
            flat_weights,
            int(ignore_index),
            float(label_smoothing),
            float(z_loss),
            bv,
            bm,
            vocab_parallel_axis,
        )
        correct = jnp.full(flat_targets.shape, -1.0, dtype=jnp.float32)
    if reduction == "none":
        return per_row.reshape(leading), correct.reshape(leading)
    total = jnp.sum(per_row)
    if reduction == "sum":
        return total, correct.reshape(leading)
    denom = jnp.maximum(jnp.sum(flat_weights), 1e-8)
    return total / denom, correct.reshape(leading)
