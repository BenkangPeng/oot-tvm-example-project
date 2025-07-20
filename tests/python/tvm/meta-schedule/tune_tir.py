
import torch
import torch.nn as nn
import torch.fx as fx
import torch.nn.functional as F

import numpy as np

import tvm
import tvm.script.tir as T
import tvm.script.relax as R
import tvm.ir.module as IRModule
import tvm.relax as relax

import tvm.te as te

import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"


def showmod(mod: IRModule, show_meta=True):
    mod.show(
        black_format=True,
        show_meta=show_meta,
        verbose_expr=True,
        show_object_address=False,
        show_all_struct_info=True,
    )


def test_vec_add():
    @tvm.script.ir_module
    class MyModuleVecAdd:
        @T.prim_func
        def main(A: T.Buffer((1024,), "float32"),  # type: ignore
                 B: T.Buffer((1024,), "float32"),  # type: ignore
                 C: T.Buffer((1024,), "float32")) -> None:  # type: ignore
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            for i in T.grid(1024):
                with T.block("C"):
                    vi = T.axis.remap("S", [i])
                    C[vi] = A[vi] + B[vi]

    mod = MyModuleVecAdd
    showmod(mod)

    sch = tvm.tir.Schedule(MyModuleVecAdd)
    block_C = sch.get_block("C")
    i, = sch.get_loops(block=block_C)
    i0, i1 = sch.split(i, [None, 128])
    showmod(sch.mod)

    # GPU Thread Blocks
    sch.bind(i0, "blockIdx.x")
    sch.bind(i1, "threadIdx.x")
    showmod(sch.mod)

    # Build and Run the TensorIR Function on GPU
    rt_mod = tvm.build(sch.mod, target="cuda")
    print(rt_mod.imported_modules[0].get_source())

    A_np = np.random.uniform(size=(1024,)).astype("float32")
    B_np = np.random.uniform(size=(1024,)).astype("float32")
    A_nd = tvm.nd.array(A_np, tvm.cuda(0))
    B_nd = tvm.nd.array(B_np, tvm.cuda(0))
    C_nd = tvm.nd.array(np.zeros((1024,), dtype="float32"), tvm.cuda(0))
    C_np_gold = A_np + B_np

    rt_mod["main"](A_nd, B_nd, C_nd)
    np.testing.assert_allclose(C_nd.numpy(), C_np_gold, rtol=1e-5, atol=1e-5)


def test_window_sum():
    # Window Sum Example
    @tvm.script.ir_module
    class MyModuleWindowSum:
        @T.prim_func
        def main(A: T.Buffer((1027,), "float32"),  # type: ignore
                 B: T.Buffer((1024,), "float32")) -> None:  # type: ignore
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            for i in T.grid(1024):
                with T.block("C"):
                    vi = T.axis.remap("S", [i])
                    B[vi] = A[vi] + A[vi + 1] + A[vi + 2]

    sch = tvm.tir.Schedule(MyModuleWindowSum)
    nthread = 128
    block_C = sch.get_block("C")
    i,  = sch.get_loops(block=block_C)
    i0, i1 = sch.split(i, [None, nthread])
    sch.bind(i0, "blockIdx.x")
    sch.bind(i1, "threadIdx.x")
    showmod(sch.mod)

    A_shared = sch.cache_read(
        block_C, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(A_shared, i1)
    sch.mod.show()

    ax = sch.get_loops(A_shared)[-1]
    ax0, ax1 = sch.split(ax, [None, nthread])
    sch.bind(ax1, "threadIdx.x")
    showmod(sch.mod)

    rt_mod = tvm.build(sch.mod, target="cuda")
    print(rt_mod.imported_modules[0].get_source())

    A_np = np.random.uniform(size=(1027,)).astype("float32")
    B_np_gold = [A_np[i] + A_np[i + 1] + A_np[i + 2] for i in range(1024)]
    A_nd = tvm.nd.array(A_np, tvm.cuda(0))
    B_nd = tvm.nd.array(np.zeros((1024,), dtype="float32"), tvm.cuda(0))
    rt_mod["main"](A_nd, B_nd)
    np.testing.assert_allclose(B_nd.numpy(), B_np_gold, rtol=1e-5, atol=1e-5)


def blocking(sch,
             tile_local_y,
             tile_local_x,
             tile_block_y,
             tile_block_x,
             tile_k):
    block_C = sch.get_block("C")
    C_local = sch.cache_write(block_C, 0, "local")

    i, j, k = sch.get_loops(block=block_C)

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])
    sch.unroll(k1)
    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)

    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")

    sch.bind(i1, "threadIdx.y")
    sch.bind(j1, "threadIdx.x")
    sch.decompose_reduction(block_C, k0)

    return sch


# Matrix Multiplication
@tvm.script.ir_module
class MyModuleMatmul:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"),  # type: ignore
             B: T.Buffer((1024, 1024), "float32"),  # type: ignore
             C: T.Buffer((1024, 1024), "float32")) -> None:  # type: ignore
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def test_matmul_blocking():

    mod = MyModuleMatmul
    # showmod(mod)

    sch = tvm.tir.Schedule(MyModuleMatmul)
    sch = blocking(sch, 8, 8, 8, 8, 4)
    showmod(sch.mod)

    rt_mod = tvm.build(sch.mod, target="cuda")
    dev = tvm.cuda(0)
    A_np = np.random.uniform(size=(1024, 1024)).astype("float32")
    B_np = np.random.uniform(size=(1024, 1024)).astype("float32")
    A_nd = tvm.nd.array(A_np, dev)
    B_nd = tvm.nd.array(B_np, dev)
    C_nd = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), dev)

    C_np_gold = np.matmul(A_np, B_np)
    rt_mod["main"](A_nd, B_nd, C_nd)
    np.testing.assert_allclose(C_nd.numpy(), C_np_gold, rtol=1e-5, atol=1e-5)

    num_flop = 2 * 1024 * 1024 * 1024
    evaluator = rt_mod.time_evaluator("main", dev, number=10)

    print("GEMM-Blocking: %f GFLOPS" %
          (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))


def cache_read_and_coop_fetch(sch, block, nthread, read_idx, read_loc):
    read_cache = sch.cache_read(
        block=block, read_buffer_index=read_idx, storage_scope="shared")
    sch.compute_at(block=read_cache, loop=read_loc)
    # vectorized cooperative fetch
    inner0, inner1 = sch.get_loops(block=read_cache)[-2:]
    inner = sch.fuse(inner0, inner1)
    _, tx, vec = sch.split(loop=inner, factors=[None, nthread, 4])
    sch.vectorize(vec)
    sch.bind(tx, "threadIdx.x")


def blocking_with_shared(
        sch,
        tile_local_y,
        tile_local_x,
        tile_block_y,
        tile_block_x,
        tile_k):
    block_C = sch.get_block("C")
    C_local = sch.cache_write(block_C, 0, "local")

    i, j, k = sch.get_loops(block=block_C)

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])

    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)

    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")

    tx = sch.fuse(i1, j1)
    sch.bind(tx, "threadIdx.x")
    nthread = tile_block_y * tile_block_x
    cache_read_and_coop_fetch(sch, block_C, nthread, 0, k0)
    cache_read_and_coop_fetch(sch, block_C, nthread, 1, k0)
    sch.decompose_reduction(block_C, k0)

    return sch


def test_matmul_blocking_with_shared():
    sch = tvm.tir.Schedule(MyModuleMatmul)
    sch = blocking_with_shared(sch, 8, 8, 8, 8, 8)
    showmod(sch.mod)

    rt_mod = tvm.build(sch.mod, target="cuda")
    print(rt_mod.imported_modules[0].get_source())
    dev = tvm.cuda(0)
    evaluator = rt_mod.time_evaluator("main", dev, number=10)
    num_flop = 2 * 1024 * 1024 * 1024

    A_np = np.random.uniform(size=(1024, 1024)).astype("float32")
    B_np = np.random.uniform(size=(1024, 1024)).astype("float32")
    A_nd = tvm.nd.array(A_np, dev)
    B_nd = tvm.nd.array(B_np, dev)

    C_np_gold = np.matmul(A_np, B_np)
    C_nd = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), dev)
    rt_mod["main"](A_nd, B_nd, C_nd)
    print("GEMM-Blocking: %f GFLOPS" %
          (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))
    np.testing.assert_allclose(C_nd.numpy(), C_np_gold, rtol=1e-5, atol=1e-5)


def test_matmul_meta_schedule():
    import tvm.meta_schedule as ms
    target = "nvidia/geforce-rtx-4090"

    database = ms.tune_tir(
        mod=MyModuleMatmul,
        target=target,
        max_trials_global=64,
        num_trials_per_iter=64,
        work_dir="./tune_tmp",
    )
    sch = ms.tir_integration.compile_tir(database, MyModuleMatmul, target)
    sch.mod.show()

    rt_mod = tvm.build(sch.mod, target="cuda")
    print(rt_mod.imported_modules[0].get_source())


if __name__ == "__main__":
    test_vec_add()
    test_window_sum()
    test_matmul_blocking()
    test_matmul_blocking_with_shared()
    test_matmul_meta_schedule()
