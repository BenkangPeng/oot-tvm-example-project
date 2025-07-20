import tvm
import torch
import tvm.script
import tvm.script.tir as T
import tvm.script.relax as R
from tvm.script import ir as I
from tvm.relax.frontend.torch import from_exported_program
import tvm.meta_schedule as ms
import tvm.relax as rx
import tvm.testing
import numpy as np

import os

os.environ["OPENBLAS_NUM_THREADS"] = "4"


def test_tir_tune():

    M, K, N = 128, 128, 128
    dtype = "float32"

    @tvm.script.ir_module
    class MyTirModule:
        @T.prim_func
        def matmul(A: T.Buffer((M, K), dtype), B: T.Buffer((K, N), dtype), C: T.Buffer((M, N), dtype)):  # type: ignore
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            for i, j, k in T.grid(M, N, K):
                with T.block("C"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = 0
                    C[vi, vj] += A[vi, vk] * B[vk, vj]

    # Remember to check the variable `$CUDA_VISIBLE_DEVICES` in the terminal
    target = "nvidia/geforce-rtx-4090"
    # It would be better to use full path instead of relative path due to some cases of environment variables.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.join(this_dir, "tune_tmp")
    database = ms.tune_tir(
        mod=MyTirModule,
        target=target,
        max_trials_global=32,
        num_trials_per_iter=32,
        work_dir=work_dir,
    )

    sch = ms.tir_integration.compile_tir(database, MyTirModule, target)

    if sch is None:
        print("No schedule found in the database")
        return

    print("✅✅✅Schedule found in the database✅✅✅")
    sch.mod.show()

    a_np = np.random.rand(M, K).astype(dtype)
    b_np = np.random.rand(K, N).astype(dtype)
    c_np_gold = np.matmul(a_np, b_np)

    a_nd = tvm.nd.array(a_np, device=tvm.cuda(0))
    b_nd = tvm.nd.array(b_np, device=tvm.cuda(0))
    c_nd = tvm.nd.array(np.zeros((M, N), dtype=dtype), device=tvm.cuda(0))

    rt_mod = tvm.build(sch.mod, target="cuda")
    print(rt_mod.imported_modules[0].get_source())

    rt_mod["main"](a_nd, b_nd, c_nd)

    tvm.testing.assert_allclose(c_nd.numpy(), c_np_gold, atol=1e-5, rtol=1e-5)

    print("✅✅✅Test TIR Tuning Success✅✅✅")


if __name__ == '__main__':
    test_tir_tune()
