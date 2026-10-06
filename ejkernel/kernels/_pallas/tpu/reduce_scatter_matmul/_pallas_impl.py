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

"""Bidirectional Reduce-Scatter Matmul with M-Split Algorithm & Fused Sequence-Parallel MLP."""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import dtypes
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
    resolved = int(tp_size) if tp_size is not None else (_infer_axis_size(axis_name) or int(jax.device_count()))
    if resolved < 1:
        raise ValueError(f"tp_size must be >= 1, got {resolved}.")
    return resolved


def mod(x: jax.Array, n: int) -> jax.Array:
    """Modulo operation that works with JAX arrays."""
    return lax.rem(x + n, n)


class KernelConfig(NamedTuple):
    """Configuration for the kernel."""

    num_devices: int
    m_block: int
    m_half_block: int
    bm: int = 128
    bn: int = 128
    bk: int = 128
    rhs_transpose: bool = False


def get_rs_vmem_estimate_bytes(
    bm: int,
    bn: int,
    bk: int,
    x_dtype: jnp.dtype,
    y_dtype: jnp.dtype,
    ring_dtype: jnp.dtype,
) -> int:
    """Estimate total scoped VMEM bytes required by reduce_scatter_matmul scratch buffers."""
    x_bytes = bm * bk * dtypes.itemsize_bits(x_dtype) // 8
    y_bytes = bk * bn * dtypes.itemsize_bits(y_dtype) // 8
    acc_bytes = bm * bn * dtypes.itemsize_bits(jnp.float32) // 8
    comp_bytes = bm * bn * dtypes.itemsize_bits(ring_dtype) // 8
    add_bytes = bm * bn * dtypes.itemsize_bits(ring_dtype) // 8
    out_bytes = bm * bn * dtypes.itemsize_bits(x_dtype) // 8
    return x_bytes + y_bytes + acc_bytes + comp_bytes + add_bytes + out_bytes


def tiled_matmul_hbm(
    x_hbm_ref: Ref,
    y_hbm_ref: Ref,
    out_hbm_ref: Ref,
    x_vmem_ref: Ref,
    y_vmem_ref: Ref,
    acc_vmem_ref: Ref,
    out_vmem_ref: Ref,
    copy_sem: Ref,
    *,
    m_block_idx: int | jax.Array,
    m_size: int,
    bm: int,
    bn: int,
    bk: int,
    rhs_transpose: bool = False,
):
    """Tiled matmul with concurrent LHS/RHS DMA copies and native BF16 MXU accumulation."""
    _, k_shard = x_hbm_ref.shape
    n_total = y_hbm_ref.shape[0] if rhs_transpose else y_hbm_ref.shape[1]
    num_m_tiles = m_size // bm
    num_n_tiles = n_total // bn
    num_k_tiles = k_shard // bk
    dot_prec = lax.Precision.HIGHEST if x_hbm_ref.dtype == jnp.float32 else lax.Precision.DEFAULT

    for m_tile in range(num_m_tiles):
        global_m_tile = m_block_idx + m_tile
        for n_tile in range(num_n_tiles):
            n_start = n_tile * bn
            for k_tile in range(num_k_tiles):
                k_start = k_tile * bk
                x_copy = pltpu.make_async_copy(
                    src_ref=x_hbm_ref.at[pl.ds(global_m_tile * bm, bm), pl.ds(k_start, bk)],
                    dst_ref=x_vmem_ref,
                    sem=copy_sem,
                )
                x_copy.start()
                if rhs_transpose:
                    y_copy = pltpu.make_async_copy(
                        src_ref=y_hbm_ref.at[pl.ds(n_start, bn), pl.ds(k_start, bk)],
                        dst_ref=y_vmem_ref,
                        sem=copy_sem,
                    )
                else:
                    y_copy = pltpu.make_async_copy(
                        src_ref=y_hbm_ref.at[pl.ds(k_start, bk), pl.ds(n_start, bn)],
                        dst_ref=y_vmem_ref,
                        sem=copy_sem,
                    )
                y_copy.start()
                x_copy.wait()
                y_copy.wait()
                lhs = x_vmem_ref[...]
                rhs = y_vmem_ref[...].T if rhs_transpose else y_vmem_ref[...]
                prod = jnp.dot(
                    lhs,
                    rhs,
                    preferred_element_type=jnp.float32,
                    precision=dot_prec,
                )
                if k_tile == 0:
                    acc_vmem_ref[...] = prod
                else:
                    acc_vmem_ref[...] = acc_vmem_ref[...] + prod
            out_vmem_ref[...] = acc_vmem_ref[...].astype(out_hbm_ref.dtype)
            out_copy = pltpu.make_async_copy(
                src_ref=out_vmem_ref,
                dst_ref=out_hbm_ref.at[pl.ds(m_tile * bm, bm), pl.ds(n_start, bn)],
                sem=copy_sem,
            )
            out_copy.start()
            out_copy.wait()


def tiled_add_hbm(
    src_hbm_ref: Ref,
    dst_hbm_ref: Ref,
    out_hbm_ref: Ref,
    src_vmem_ref: Ref,
    dst_vmem_ref: Ref,
    out_vmem_ref: Ref,
    copy_sem: Ref,
    *,
    bm: int,
    bn: int,
    m_out_offset: int = 0,
):
    m_size, n_total = src_hbm_ref.shape
    num_m_tiles = m_size // bm
    num_n_tiles = n_total // bn
    for m_tile in range(num_m_tiles):
        m_start = m_tile * bm
        for n_tile in range(num_n_tiles):
            n_start = n_tile * bn
            src_copy = pltpu.make_async_copy(
                src_ref=src_hbm_ref.at[pl.ds(m_start, bm), pl.ds(n_start, bn)],
                dst_ref=src_vmem_ref,
                sem=copy_sem,
            )
            src_copy.start()
            dst_copy = pltpu.make_async_copy(
                src_ref=dst_hbm_ref.at[pl.ds(m_start, bm), pl.ds(n_start, bn)],
                dst_ref=dst_vmem_ref,
                sem=copy_sem,
            )
            dst_copy.start()
            src_copy.wait()
            dst_copy.wait()
            result = src_vmem_ref[...].astype(jnp.float32) + dst_vmem_ref[...].astype(jnp.float32)
            out_vmem_ref[...] = result.astype(out_hbm_ref.dtype)
            out_copy = pltpu.make_async_copy(
                src_ref=out_vmem_ref,
                dst_ref=out_hbm_ref.at[pl.ds(m_out_offset + m_start, bm), pl.ds(n_start, bn)],
                sem=copy_sem,
            )
            out_copy.start()
            out_copy.wait()


def _kernel(
    x_ref: Ref,
    y_ref: Ref,
    out_ref: Ref,
    scratch_ref: Ref,
    computation_scratch_ref: Ref,
    x_vmem_ref: Ref,
    y_vmem_ref: Ref,
    acc_vmem_ref: Ref,
    comp_vmem_ref: Ref,
    add_vmem_ref: Ref,
    out_vmem_ref: Ref,
    send_left_sem: Ref,
    recv_left_sem: Ref,
    send_right_sem: Ref,
    recv_right_sem: Ref,
    copy_sem: Ref,
    left_capacity_sem: Ref,
    right_capacity_sem: Ref,
    *,
    config: KernelConfig,
    axis_name: str,
):
    num_devices = config.num_devices
    m_block = config.m_block
    m_half_block = config.m_half_block
    bm, bn, bk = config.bm, config.bn, config.bk
    rhs_transpose = config.rhs_transpose
    ring_step = pl.program_id(0)
    my_id = lax.axis_index(axis_name)
    left_neighbor = mod(my_id - 1, num_devices)
    right_neighbor = mod(my_id + 1, num_devices)
    left_working_slot = lax.rem(ring_step, 2)
    left_receiving_slot = 1 - left_working_slot
    right_working_slot = 2 + lax.rem(ring_step, 2)
    right_receiving_slot = 5 - right_working_slot
    left_compute_slot = 0
    right_compute_slot = 1
    num_steps = num_devices
    is_first_step = ring_step == 0
    is_last_step = ring_step == num_steps - 1

    def get_left_target_block(step):
        return mod(my_id + step + 1, num_devices)

    def get_right_target_block(step):
        return mod(my_id - step - 1, num_devices)

    def compute_matmul_top_half(block_idx, out_slot):
        m_block_idx = block_idx * (m_block // bm)
        tiled_matmul_hbm(
            x_hbm_ref=x_ref,
            y_hbm_ref=y_ref,
            out_hbm_ref=computation_scratch_ref.at[out_slot],
            x_vmem_ref=x_vmem_ref,
            y_vmem_ref=y_vmem_ref,
            acc_vmem_ref=acc_vmem_ref,
            out_vmem_ref=comp_vmem_ref,
            copy_sem=copy_sem,
            m_block_idx=m_block_idx,
            m_size=m_half_block,
            bm=bm,
            bn=bn,
            bk=bk,
            rhs_transpose=rhs_transpose,
        )

    def compute_matmul_bot_half(block_idx, out_slot):
        m_block_idx = block_idx * (m_block // bm) + m_half_block // bm
        tiled_matmul_hbm(
            x_hbm_ref=x_ref,
            y_hbm_ref=y_ref,
            out_hbm_ref=computation_scratch_ref.at[out_slot],
            x_vmem_ref=x_vmem_ref,
            y_vmem_ref=y_vmem_ref,
            acc_vmem_ref=acc_vmem_ref,
            out_vmem_ref=comp_vmem_ref,
            copy_sem=copy_sem,
            m_block_idx=m_block_idx,
            m_size=m_half_block,
            bm=bm,
            bn=bn,
            bk=bk,
            rhs_transpose=rhs_transpose,
        )

    def accumulate_computation_to_slot(compute_slot, dst_slot):
        tiled_add_hbm(
            src_hbm_ref=computation_scratch_ref.at[compute_slot],
            dst_hbm_ref=scratch_ref.at[dst_slot],
            out_hbm_ref=scratch_ref.at[dst_slot],
            src_vmem_ref=add_vmem_ref,
            dst_vmem_ref=comp_vmem_ref,
            out_vmem_ref=comp_vmem_ref,
            copy_sem=copy_sem,
            bm=bm,
            bn=bn,
        )

    def accumulate_computation_to_out(compute_slot, dst_slot, m_out_offset):
        tiled_add_hbm(
            src_hbm_ref=computation_scratch_ref.at[compute_slot],
            dst_hbm_ref=scratch_ref.at[dst_slot],
            out_hbm_ref=out_ref,
            src_vmem_ref=add_vmem_ref,
            dst_vmem_ref=comp_vmem_ref,
            out_vmem_ref=out_vmem_ref,
            copy_sem=copy_sem,
            bm=bm,
            bn=bn,
            m_out_offset=m_out_offset,
        )

    def copy_computation_to_slot(compute_slot, dst_slot):
        local_copy = pltpu.make_async_copy(
            src_ref=computation_scratch_ref.at[compute_slot],
            dst_ref=scratch_ref.at[dst_slot],
            sem=copy_sem,
        )
        local_copy.start()
        local_copy.wait()

    def local_barrier():
        barrier_sem = pltpu.get_barrier_semaphore()
        pl.semaphore_signal(barrier_sem, inc=1, device_id=(left_neighbor,), device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_signal(barrier_sem, inc=1, device_id=(right_neighbor,), device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(barrier_sem, 2)

        @functools.partial(pl.run_scoped, second_barrier=pltpu.SemaphoreType.REGULAR)
        def _(second_barrier):
            pl.semaphore_signal(
                second_barrier, inc=1, device_id=(left_neighbor,), device_id_type=pl.DeviceIdType.MESH
            )
            pl.semaphore_signal(
                second_barrier, inc=1, device_id=(right_neighbor,), device_id_type=pl.DeviceIdType.MESH
            )
            pl.semaphore_wait(second_barrier, 2)

    def signal_left_neighbor():
        pl.semaphore_signal(
            left_capacity_sem, inc=1, device_id=(left_neighbor,), device_id_type=pl.DeviceIdType.MESH
        )

    def signal_right_neighbor():
        pl.semaphore_signal(
            right_capacity_sem, inc=1, device_id=(right_neighbor,), device_id_type=pl.DeviceIdType.MESH
        )

    left_target_block = get_left_target_block(ring_step)
    right_target_block = get_right_target_block(ring_step)

    @pl.when(is_first_step)
    def _prologue():
        local_barrier()
        compute_matmul_top_half(left_target_block, left_compute_slot)
        compute_matmul_bot_half(right_target_block, right_compute_slot)
        copy_computation_to_slot(left_compute_slot, left_working_slot)
        copy_computation_to_slot(right_compute_slot, right_working_slot)

    @pl.when(~is_first_step)
    def _main_loop():
        signal_left_neighbor()
        signal_right_neighbor()
        pl.semaphore_wait(left_capacity_sem, 1)
        pl.semaphore_wait(right_capacity_sem, 1)
        remote_copy_to_left = pltpu.make_async_remote_copy(
            src_ref=scratch_ref.at[left_receiving_slot],
            dst_ref=scratch_ref.at[left_working_slot],
            send_sem=send_left_sem,
            recv_sem=recv_left_sem,
            device_id=(left_neighbor,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        remote_copy_to_left.start()
        remote_copy_to_right = pltpu.make_async_remote_copy(
            src_ref=scratch_ref.at[right_receiving_slot],
            dst_ref=scratch_ref.at[right_working_slot],
            send_sem=send_right_sem,
            recv_sem=recv_right_sem,
            device_id=(right_neighbor,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        remote_copy_to_right.start()
        compute_matmul_top_half(left_target_block, left_compute_slot)
        compute_matmul_bot_half(right_target_block, right_compute_slot)
        remote_copy_to_left.wait()
        remote_copy_to_right.wait()

        @pl.when(is_last_step)
        def _epilogue():
            accumulate_computation_to_out(left_compute_slot, left_working_slot, 0)
            accumulate_computation_to_out(right_compute_slot, right_working_slot, m_half_block)

        @pl.when(~is_last_step)
        def _accumulate():
            accumulate_computation_to_slot(left_compute_slot, left_working_slot)
            accumulate_computation_to_slot(right_compute_slot, right_working_slot)


def reduce_scatter_matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    axis_name: str = "x",
    tp_size: int | None = None,
    collective_id: int | None = 0,
    bm: int = 512,
    bn: int = 1024,
    bk: int = 1024,
    rhs_transpose: bool = False,
    ring_dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Bidirectional reduce-scatter matmul with M-split algorithm and explicit VMEM budget."""
    num_devices = _resolve_tp_size(tp_size, axis_name)
    m_total, k_shard = x.shape
    n_total = y.shape[0] if rhs_transpose else y.shape[1]
    m_block = m_total // num_devices
    m_half_block = m_block // 2
    bm = min(int(bm), m_half_block)
    bn = min(int(bn), n_total)
    bk = min(int(bk), k_shard)
    while m_half_block % bm != 0 and bm > 128:
        bm //= 2
    while n_total % bn != 0 and bn > 128:
        bn //= 2
    while k_shard % bk != 0 and bk > 128:
        bk //= 2

    estimated_vmem_bytes = get_rs_vmem_estimate_bytes(bm, bn, bk, x.dtype, y.dtype, ring_dtype)
    vmem_limit_bytes = max(64 * 1024 * 1024, estimated_vmem_bytes + 16 * 1024 * 1024)

    config = KernelConfig(
        num_devices=num_devices,
        m_block=m_block,
        m_half_block=m_half_block,
        bm=bm,
        bn=bn,
        bk=bk,
        rhs_transpose=rhs_transpose,
    )
    out_shape = jax.ShapeDtypeStruct((m_block, n_total), x.dtype)
    scratch_shape = jax.ShapeDtypeStruct((4, m_half_block, n_total), ring_dtype)
    computation_scratch_shape = jax.ShapeDtypeStruct((2, m_half_block, n_total), ring_dtype)
    x_vmem_shape = pltpu.VMEM((bm, bk), x.dtype)
    y_vmem_shape = pltpu.VMEM((bn, bk) if rhs_transpose else (bk, bn), y.dtype)
    acc_vmem_shape = pltpu.VMEM((bm, bn), jnp.float32)
    comp_vmem_shape = pltpu.VMEM((bm, bn), ring_dtype)
    add_vmem_shape = pltpu.VMEM((bm, bn), ring_dtype)
    out_vmem_shape = pltpu.VMEM((bm, bn), x.dtype)
    grid = (num_devices,)

    def kernel_fn(
        x_ref,
        y_ref,
        out_ref,
        scratch_ref,
        computation_scratch_ref,
        x_vmem_ref,
        y_vmem_ref,
        acc_vmem_ref,
        comp_vmem_ref,
        add_vmem_ref,
        out_vmem_ref,
        send_left_sem,
        recv_left_sem,
        send_right_sem,
        recv_right_sem,
        copy_sem,
        left_capacity_sem,
        right_capacity_sem,
    ):
        _kernel(
            x_ref=x_ref,
            y_ref=y_ref,
            out_ref=out_ref,
            scratch_ref=scratch_ref,
            computation_scratch_ref=computation_scratch_ref,
            x_vmem_ref=x_vmem_ref,
            y_vmem_ref=y_vmem_ref,
            acc_vmem_ref=acc_vmem_ref,
            comp_vmem_ref=comp_vmem_ref,
            add_vmem_ref=add_vmem_ref,
            out_vmem_ref=out_vmem_ref,
            send_left_sem=send_left_sem,
            recv_left_sem=recv_left_sem,
            send_right_sem=send_right_sem,
            recv_right_sem=recv_right_sem,
            copy_sem=copy_sem,
            left_capacity_sem=left_capacity_sem,
            right_capacity_sem=right_capacity_sem,
            config=config,
            axis_name=axis_name,
        )

    out, _, _ = pl.pallas_call(
        kernel_fn,
        out_shape=(out_shape, scratch_shape, computation_scratch_shape),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec(memory_space=pl.ANY), pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=[
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY),
            ],
            scratch_shapes=[
                x_vmem_shape,
                y_vmem_shape,
                acc_vmem_shape,
                comp_vmem_shape,
                add_vmem_shape,
                out_vmem_shape,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.REGULAR,
                pltpu.SemaphoreType.REGULAR,
            ],
            grid=grid,
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        name=f"reduce_scatter_matmul_bm_{bm}_bn_{bn}_bk_{bk}",
    )(x, y)
    return out


def _matmul_gelu_kernel(x_ref, w_ref, o_ref, acc_ref):
    """Fused Matmul + tanh-GELU epilogue in VMEM using pipelined BlockSpecs."""
    k_idx = pl.program_id(2)

    @pl.when(k_idx == 0)
    def _():
        acc_ref[...] = jnp.dot(
            x_ref[...], w_ref[...], preferred_element_type=jnp.float32
        )

    @pl.when(k_idx != 0)
    def _():
        acc_ref[...] += jnp.dot(
            x_ref[...], w_ref[...], preferred_element_type=jnp.float32
        )

    @pl.when(k_idx == pl.num_programs(2) - 1)
    def _():
        val = acc_ref[...]
        val_sq = val * val
        poly = val * (1.0 + 0.044715 * val_sq)
        g = 0.5 * val * (1.0 + jnp.tanh(0.7978845608028654 * poly))
        o_ref[...] = g.astype(o_ref.dtype)


def pallas_matmul_gelu(
    x: jax.Array,
    w: jax.Array,
    bm: int = 1024,
    bn: int = 1024,
    bk: int = 1024,
) -> jax.Array:
    """Fused X @ W1 + GELU in a single TensorCore Pallas kernel without HBM round-trip."""
    m, k = x.shape
    _, n = w.shape
    bm = min(bm, m)
    bn = min(bn, n)
    bk = min(bk, k)
    while m % bm != 0 and bm > 128:
        bm //= 2
    while n % bn != 0 and bn > 128:
        bn //= 2
    while k % bk != 0 and bk > 128:
        bk //= 2

    return pl.pallas_call(
        _matmul_gelu_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=(m // bm, n // bn, k // bk),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, p: (i, p)),
            pl.BlockSpec((bk, bn), lambda i, j, p: (p, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, p: (i, j)),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=96 * 1024 * 1024,
        ),
    )(x, w)


def sequence_mlp(
    x: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    *,
    axis_name: str = "rank",
    bm: int = 1024,
    bn: int = 1024,
    bk: int = 1024,
    num_chunks: int = 8,
) -> jax.Array:
    """Sequence-parallel MLP with chunked async ICI overlap and VMEM-fused Matmul1+GELU."""
    m_local, _ = x.shape
    while num_chunks > 1 and m_local % num_chunks != 0:
        num_chunks //= 2

    if num_chunks > 1:
        chunks = jnp.split(x, num_chunks, axis=0)
        x_fulls = [
            lax.all_gather(chunk, axis_name, axis=0, tiled=True)
            for chunk in chunks
        ]
        outs = []
        for x_full in x_fulls:
            z = pallas_matmul_gelu(x_full, w1, bm=bm, bn=bn, bk=bk)
            y = jnp.dot(z, w2, preferred_element_type=jnp.float32)
            out = lax.psum_scatter(y, axis_name, scatter_dimension=0, tiled=True)
            outs.append(out)
        return jnp.concatenate(outs, axis=0).astype(x.dtype)

    x_full = lax.all_gather(x, axis_name, axis=0, tiled=True)
    z = pallas_matmul_gelu(x_full, w1, bm=bm, bn=bn, bk=bk)
    y = jnp.dot(z, w2, preferred_element_type=jnp.float32)
    return lax.psum_scatter(y, axis_name, scatter_dimension=0, tiled=True).astype(x.dtype)
