# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from builtin.io import _printf
from gpu import barrier
from gpu.host import DeviceContext
from gpu.host._compile import _get_gpu_target
from gpu.host._nvidia_cuda import TMADescriptor, create_tma_descriptor
from gpu.id import block_idx, thread_idx
from gpu.memory import (
    _GPUAddressSpace,
    cp_async_bulk_tensor_shared_cluster_global,
    cp_async_bulk_tensor_global_shared_cta,
    tma_store_fence,
)
from gpu.sync import (
    mbarrier_arrive_expect_tx_shared,
    mbarrier_init,
    mbarrier_try_wait_parity_shared,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
)
from memory import UnsafePointer, stack_allocation
from testing import assert_equal

from utils.index import Index
from utils.static_tuple import StaticTuple

alias GMEM_HEIGHT = 4096
alias GMEM_WIDTH = 4096
alias BLOCK_SIZE = 32
alias SMEM_HEIGHT = BLOCK_SIZE
alias SMEM_WIDTH = BLOCK_SIZE


@__llvm_arg_metadata(descriptor, `nvvm.grid_constant`)
fn kernel_copy_async_tma[block_size: Int](descriptor: TMADescriptor):
    var shmem = stack_allocation[
        block_size * block_size,
        DType.float32,
        alignment=1024,
        address_space = _GPUAddressSpace.SHARED,
    ]()
    var mbar = stack_allocation[
        1, Int64, address_space = _GPUAddressSpace.SHARED
    ]()
    var descriptor_ptr = UnsafePointer(to=descriptor).bitcast[NoneType]()

    x = block_idx.x * block_size
    y = block_idx.y * block_size

    col = thread_idx.x % block_size
    row = thread_idx.x // block_size

    # LOAD
    if thread_idx.x == 0:
        mbarrier_init(mbar, 1)
        mbarrier_arrive_expect_tx_shared(mbar, block_size * block_size * 4)
        cp_async_bulk_tensor_shared_cluster_global(
            shmem, descriptor_ptr, mbar, Index(x, y)
        )
    barrier()
    mbarrier_try_wait_parity_shared(mbar, 0, 10000000)

    # COMPUTE
    shmem[row * block_size + col] += 1

    # FENCE
    barrier()
    tma_store_fence()

    # STORE
    if thread_idx.x == 0:
        cp_async_bulk_tensor_global_shared_cta(
            shmem, descriptor_ptr, Index(x, y)
        )
        cp_async_bulk_commit_group()

    cp_async_bulk_wait_group[0]()


def test_tma_tile_copy(ctx: DeviceContext):
    # Initialize input matrix
    var gmem_host = UnsafePointer[Float32].alloc(GMEM_HEIGHT * GMEM_WIDTH)
    for i in range(GMEM_HEIGHT * GMEM_WIDTH):
        gmem_host[i] = i

    # Create device buffer and copy data
    var gmem_dev = ctx.enqueue_create_buffer[DType.float32](
        GMEM_HEIGHT * GMEM_WIDTH
    )
    ctx.enqueue_copy(gmem_dev, gmem_host)

    # Create TMA descriptor
    var descriptor = create_tma_descriptor[DType.float32, 2](
        gmem_dev,
        (GMEM_HEIGHT, GMEM_WIDTH),
        (GMEM_WIDTH, 1),
        (SMEM_HEIGHT, SMEM_WIDTH),
    )

    # Run kernel
    ctx.enqueue_function[kernel_copy_async_tma[BLOCK_SIZE]](
        descriptor,
        grid_dim=(GMEM_HEIGHT // SMEM_HEIGHT, GMEM_WIDTH // SMEM_WIDTH, 1),
        block_dim=(SMEM_HEIGHT * SMEM_WIDTH, 1, 1),
    )

    # Copy result back to host
    ctx.enqueue_copy(gmem_host, gmem_dev)
    ctx.synchronize()

    # Verify results
    for matrix_row in range(GMEM_HEIGHT // SMEM_HEIGHT):
        for matrix_col in range(GMEM_WIDTH // SMEM_WIDTH):
            for row in range(SMEM_HEIGHT):
                for col in range(SMEM_WIDTH):
                    idx = (matrix_row * SMEM_HEIGHT + row) * GMEM_WIDTH + (
                        matrix_col * SMEM_WIDTH + col
                    )
                    expected = idx + 1
                    assert_equal(
                        gmem_host[idx], expected, "TMA copy verification failed"
                    )

    gmem_host.free()


def main():
    with DeviceContext() as ctx:
        test_tma_tile_copy(ctx)
