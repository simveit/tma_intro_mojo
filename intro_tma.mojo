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
from gpu.id import block_idx
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

from utils.index import Index
from utils.static_tuple import StaticTuple


@__llvm_arg_metadata(descriptor, `nvvm.grid_constant`)
fn kernel_copy_async_tma(descriptor: TMADescriptor):
    var shmem = stack_allocation[
        16, DType.float32, alignment=16, address_space = _GPUAddressSpace.SHARED
    ]()
    var mbar = stack_allocation[
        1, Int64, address_space = _GPUAddressSpace.SHARED
    ]()
    var descriptor_ptr = UnsafePointer(to=descriptor).bitcast[NoneType]()
    mbarrier_init(mbar, 1)

    mbarrier_arrive_expect_tx_shared(mbar, 64)
    cp_async_bulk_tensor_shared_cluster_global(
        shmem, descriptor_ptr, mbar, Index(block_idx.x * 4, block_idx.y * 4)
    )
    mbarrier_try_wait_parity_shared(mbar, 0, 10000000)

    @parameter
    for i in range(16):
        shmem[i] += block_idx.x + block_idx.y

    ### Barrier
    barrier()
    tma_store_fence()

    ### Additional code to copy back
    cp_async_bulk_tensor_global_shared_cta(
        shmem, descriptor_ptr, Index(block_idx.x * 4, block_idx.y * 4)
    )
    # Note: Normally only do for first thread, here we only have one.
    cp_async_bulk_commit_group()
    cp_async_bulk_wait_group[0]()


def test_tma_tile_copy(ctx: DeviceContext):
    print("== test_tma_tile_copy")
    var gmem_host = UnsafePointer[Float32].alloc(8 * 8)
    for i in range(64):
        gmem_host[i] = i

    print("Initial 4x4 matrices:")
    for matrix_row in range(2):
        for matrix_col in range(2):
            print("\nMatrix at position (", matrix_row, ",", matrix_col, "):")
            for row in range(4):
                for col in range(4):
                    idx = (matrix_row * 4 + row) * 8 + (matrix_col * 4 + col)
                    print(String(gmem_host[idx]).ljust(4), end=" ")
                print()
    print()

    var gmem_dev = ctx.enqueue_create_buffer[DType.float32](8 * 8)

    ctx.enqueue_copy(gmem_dev, gmem_host)

    var descriptor = create_tma_descriptor[DType.float32, 2](
        gmem_dev, (8, 8), (8, 1), (4, 4)
    )

    ctx.enqueue_function[kernel_copy_async_tma](
        descriptor, grid_dim=(2, 2), block_dim=(1)
    )
    ctx.enqueue_copy(gmem_host, gmem_dev)
    ctx.synchronize()

    print("Final 4x4 matrices:")
    for matrix_row in range(2):
        for matrix_col in range(2):
            print("\nMatrix at position (", matrix_row, ",", matrix_col, "):")
            for row in range(4):
                for col in range(4):
                    idx = (matrix_row * 4 + row) * 8 + (matrix_col * 4 + col)
                    print(String(gmem_host[idx]).ljust(4), end=" ")
                print()
    print()
    gmem_host.free()


def main():
    with DeviceContext() as ctx:
        test_tma_tile_copy(ctx)
