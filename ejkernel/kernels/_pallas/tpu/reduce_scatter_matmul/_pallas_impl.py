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

"""Pipelined Reduce-Scatter Matmul with Staggered N-Split Ring Overlap on TPU.

Replaces the unpipelined per-tile synchronous DMA loops (`tiled_matmul_hbm` and
`tiled_add_hbm`) and redundant HBM scratchpads (`computation_scratch_ref` +
`scratch_ref`) with:
1. A hardware-pipelined 3D Pallas `BlockSpec` matmul kernel with on-chip VMEM
   `float32` accumulation (`dimension_semantics=("parallel", "parallel", "arbitrary")`)
   and native `bfloat16` MXU dot products.
2. N-split (column-partitioned) staggered `lax.ppermute` ring steps that keep
   the full per-rank `M_block = M // tp_size` tile height intact, read weights `y`
   only once per ring step instead of twice, and overlap ICI ring communication of
   one N-partition with TensorCore matmul computation of the other N-partition.
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

Ref = Any


def _infer_axis_size(axis_name: str) -> int | None:
    """Infer collective axis size from the active mapped context when available."""
    try:
        return jax.core.concrete_or_error(
            int,
            lax.psum(jnp.array(1, dtype=jnp.int32), axis_name=axis_name),
            f"collective axis '{axis_name}' size must be static.",
        )
    except Exception:
        return None


def _resolve_tp_size(tp_size: int | None, axis_name: str) -> int:
    """Resolve tensor-parallel world size using explicit value, axis context, then global device count."""
    resolved = (
        int(tp_size)
        if tp_size is not None
        else (_infer_axis_size(axis_name) or int(jax.device_count()))
    )
    if resolved < 1:
        raise ValueError(f"tp_size must be >= 1, got {resolved}.")
    return resolved


def mod(x: jax.Array, n: int) -> jax.Array:
    """Modulo operation that works with JAX arrays."""
    return lax.rem(x + n, n)


class KernelConfig(NamedTuple):
    """Configuration for the reduce-scatter matmul kernel."""

    num_devices: int
    m_block: int
    m_half_block: int
    bm: int = 512
    bn: int = 1024
    bk: int = 512
    rhs_transpose: bool = True


def _select_block_size(dim_size: int, requested: int, default_target: int) -> int:
    """Choose a largest power-of-two multiple of 128 dividing `dim_size`."""
    target = default_target if requested == 128 else requested
    b = min(int(target), dim_size)
    while dim_size % b != 0 and b > 128:
        b //= 2
    return b


def _matmul_kernel(
    x_ref: Ref,
    y_ref: Ref,
    o_ref: Ref,
    acc_ref: Ref,
    *,
    rhs_transpose: bool,
    dot_prec: lax.Precision,
):
    k_id = pl.program_id(2)

    @pl.when(k_id == 0)
    def _zero():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    lhs = x_ref[...]
    rhs = y_ref[...].T if rhs_transpose else y_ref[...]
    acc_ref[...] += jnp.dot(
        lhs,
        rhs,
        preferred_element_type=jnp.float32,
        precision=dot_prec,
    )

    @pl.when(k_id == pl.num_programs(2) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _pallas_matmul_part(
    x_slice: jax.Array,
    y: jax.Array,
    part_idx: int,
    num_parts: int,
    bm: int,
    bn: int,
    bk: int,
    *,
    rhs_transpose: bool = True,
) -> jax.Array:
    """Hardware-pipelined Pallas matmul for a single M-block and N-partition."""
    s, k_shard = x_slice.shape
    n_total = y.shape[0] if rhs_transpose else y.shape[1]
    part_n = n_total // num_parts
    part_block_offset = part_idx * (part_n // bn)
    dot_prec = (
        lax.Precision.HIGHEST
        if x_slice.dtype == jnp.float32
        else lax.Precision.DEFAULT
    )

    if rhs_transpose:
        y_spec = pl.BlockSpec(
            (bn, bk), lambda i, j, p: (part_block_offset + j, p)
        )
    else:
        y_spec = pl.BlockSpec(
            (bk, bn), lambda i, j, p: (p, part_block_offset + j)
        )

    return pl.pallas_call(
        functools.partial(
            _matmul_kernel,
            rhs_transpose=rhs_transpose,
            dot_prec=dot_prec,
        ),
        grid=(s // bm, part_n // bn, k_shard // bk),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, p: (i, p)),
            y_spec,
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, p: (i, j)),
        out_shape=jax.ShapeDtypeStruct((s, part_n), jnp.float32),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
    )(x_slice, y)


def reduce_scatter_matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    axis_name: str = "x",
    tp_size: int | None = None,
    collective_id: int | None = 0,
    bm: int = 128,
    bn: int = 128,
    bk: int = 128,
    rhs_transpose: bool = True,
) -> jax.Array:
    """Pipelined reduce-scatter matmul with staggered N-split ring overlap.

    Computes `reduce_scatter(x @ y.T, scatter_dim=0)` (or `x @ y` when
    `rhs_transpose=False`) across `axis_name` in `float32` ring accumulation
    and casts the scattered output to `x.dtype`.
    """
    del collective_id
    tp_size = _resolve_tp_size(tp_size, axis_name)
    if tp_size == 1:
        rhs = y.T if rhs_transpose else y
        return jnp.dot(x, rhs, preferred_element_type=jnp.float32).astype(x.dtype)

    num_ranks = int(tp_size)
    m_total, k_shard = x.shape
    n_total = y.shape[0] if rhs_transpose else y.shape[1]

    if m_total % num_ranks != 0:
        raise ValueError(
            f"M ({m_total}) must be divisible by num_devices ({num_ranks})."
        )

    s = m_total // num_ranks
    me = lax.axis_index(axis_name)
    perm = [(i, (i + 1) % num_ranks) for i in range(num_ranks)]
    x_chunks = x.reshape((num_ranks, s, k_shard))

    # Choose 4-way N-split on large shapes (e.g. M_block >= 1024, N >= 8192)
    # and 2-way N-split on medium/small shapes to maximize MXU/ICI overlap.
    if s >= 1024 and n_total >= 4096 and (n_total // 4) % 128 == 0:
        quarter_n = n_total // 4
        eff_bm = _select_block_size(s, bm, 1024)
        eff_bn = _select_block_size(quarter_n, bn, 2048)
        eff_bk = _select_block_size(k_shard, bk, 512)

        c0 = (me + num_ranks - 1) % num_ranks
        x0 = x_chunks[c0]

        buf0 = _pallas_matmul_part(
            x0, y, 0, 4, eff_bm, eff_bn, eff_bk, rhs_transpose=rhs_transpose
        )
        buf1 = _pallas_matmul_part(
            x0, y, 1, 4, eff_bm, eff_bn, eff_bk, rhs_transpose=rhs_transpose
        )
        buf2 = _pallas_matmul_part(
            x0, y, 2, 4, eff_bm, eff_bn, eff_bk, rhs_transpose=rhs_transpose
        )
        buf3 = _pallas_matmul_part(
            x0, y, 3, 4, eff_bm, eff_bn, eff_bk, rhs_transpose=rhs_transpose
        )

        for t in range(num_ranks - 1):
            c_next = (me + 2 * num_ranks - 2 - t) % num_ranks
            x_next = x_chunks[c_next]

            send0 = lax.ppermute(buf0, axis_name, perm)
            send1 = lax.ppermute(buf1, axis_name, perm)
            next_y2 = _pallas_matmul_part(
                x_next, y, 2, 4, eff_bm, eff_bn, eff_bk, rhs_transpose=rhs_transpose
            )
            next_y3 = _pallas_matmul_part(
                x_next, y, 3, 4, eff_bm, eff_bn, eff_bk, rhs_transpose=rhs_transpose
            )

            send2 = lax.ppermute(buf2, axis_name, perm)
            send3 = lax.ppermute(buf3, axis_name, perm)
            next_y0 = _pallas_matmul_part(
                x_next, y, 0, 4, eff_bm, eff_bn, eff_bk, rhs_transpose=rhs_transpose
            )
            next_y1 = _pallas_matmul_part(
                x_next, y, 1, 4, eff_bm, eff_bn, eff_bk, rhs_transpose=rhs_transpose
            )

            buf0 = send0 + next_y0
            buf1 = send1 + next_y1
            buf2 = send2 + next_y2
            buf3 = send3 + next_y3

        return jnp.concatenate(
            [
                buf0.astype(x.dtype),
                buf1.astype(x.dtype),
                buf2.astype(x.dtype),
                buf3.astype(x.dtype),
            ],
            axis=1,
        )

    if n_total % 2 == 0 and (n_total // 2) % 128 == 0:
        half_n = n_total // 2
        eff_bm = _select_block_size(s, bm, 512)
        eff_bn = _select_block_size(half_n, bn, 1024)
        eff_bk = _select_block_size(k_shard, bk, 512)

        c0 = (me + num_ranks - 1) % num_ranks
        x0 = x_chunks[c0]

        buf0 = _pallas_matmul_part(
            x0, y, 0, 2, eff_bm, eff_bn, eff_bk, rhs_transpose=rhs_transpose
        )
        buf1 = _pallas_matmul_part(
            x0, y, 1, 2, eff_bm, eff_bn, eff_bk, rhs_transpose=rhs_transpose
        )

        for t in range(num_ranks - 1):
            c_next = (me + 2 * num_ranks - 2 - t) % num_ranks
            x_next = x_chunks[c_next]

            send0 = lax.ppermute(buf0, axis_name, perm)
            next_y1 = _pallas_matmul_part(
                x_next, y, 1, 2, eff_bm, eff_bn, eff_bk, rhs_transpose=rhs_transpose
            )

            send1 = lax.ppermute(buf1, axis_name, perm)
            next_y0 = _pallas_matmul_part(
                x_next, y, 0, 2, eff_bm, eff_bn, eff_bk, rhs_transpose=rhs_transpose
            )

            buf0 = send0 + next_y0
            buf1 = send1 + next_y1

        return jnp.concatenate(
            [buf0.astype(x.dtype), buf1.astype(x.dtype)], axis=1
        )

    eff_bm = _select_block_size(s, bm, 512)
    eff_bn = _select_block_size(n_total, bn, 1024)
    eff_bk = _select_block_size(k_shard, bk, 512)

    c0 = (me + num_ranks - 1) % num_ranks
    buf = _pallas_matmul_part(
        x_chunks[c0], y, 0, 1, eff_bm, eff_bn, eff_bk, rhs_transpose=rhs_transpose
    )
    for t in range(num_ranks - 1):
        send_buf = lax.ppermute(buf, axis_name, perm)
        c_next = (me + 2 * num_ranks - 2 - t) % num_ranks
        next_y = _pallas_matmul_part(
            x_chunks[c_next],
            y,
            0,
            1,
            eff_bm,
            eff_bn,
            eff_bk,
            rhs_transpose=rhs_transpose,
        )
        buf = send_buf + next_y

    return buf.astype(x.dtype)
