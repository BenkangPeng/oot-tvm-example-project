import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import tvm
import numpy as np
import tvm.meta_schedule as ms

from tvm.script import relax as R
from tvm.script import tir as T
from tvm.script import ir as I

M, K, N = 32, 64, 128
dtype = "float32"

USE_TIR_OPT_ONLY = True


@I.ir_module
class Module:
    @T.prim_func
    def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [M, K], dtype=dtype)
        B = T.match_buffer(b, [K, N], dtype=dtype)
        C = T.match_buffer(c, [M, N], dtype=dtype)
        for i, j, k in T.grid(M, N, K):  # type: ignore
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])  # type: ignore
                with T.init():
                    C[vi, vj] = T.cast(0.0, dtype=dtype)  # type: ignore
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    @T.prim_func
    def relu(c: T.handle, y: T.handle) -> None:
        C = T.match_buffer(c, [M, N], dtype=dtype)
        Y = T.match_buffer(y, [M, N], dtype=dtype)
        for i, j in T.grid(M, N):
            with T.block("Y"):
                vi, vj = T.axis.remap("SS", [i, j])
                Y[vi, vj] = T.max(C[vi, vj], T.cast(0.0, dtype=dtype))

    @R.function
    def main(
        a: R.Tensor(shape=[M, K], dtype=dtype), b: R.Tensor(shape=[K, N], dtype=dtype)  # type: ignore
    ) -> R.Tensor(shape=[M, N], dtype=dtype):  # type: ignore
        cls = Module
        with R.dataflow():
            lv0 = R.call_tir(cls.matmul, (a, b), out_sinfo=R.Tensor(shape=[M, N], dtype=dtype))
            out = R.call_tir(cls.relu, (lv0), out_sinfo=R.Tensor(shape=[M, N], dtype=dtype))
            R.output(out)
        return out


mod = Module
mod.show()


######################################################################
# Define the target
# Remember to check the variable `$CUDA_VISIBLE_DEVICES` in the terminal
target = tvm.target.Target("nvidia/geforce-rtx-4090")  # Change to your target device
total_trials = 1000
this_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.join(this_dir, "tune_tmp")


######################################################################
# Use relax.get_pipeline
# BUG `static_shape_tuning` will return a not-tuned mod.
# In `evolutionary_search.cc` Line521, `results` still be empty after running `support::parallel_for_dynamic`.
tvm_mod = tvm.relax.get_pipeline("static_shape_tuning", target=target, total_trials=total_trials, work_dir=work_dir)(mod)
tvm_mod.show()
tvm_mod = tvm.relax.transform.LegalizeOps()(tvm_mod)
tvm_mod.show()

######################################################################
# Create device
dev = tvm.cuda()

# # Build
# ex = tvm.relax.build(tvm_mod, target=target)
# vm = tvm.relax.VirtualMachine(ex, dev)

# # Prepare Data
# a_np = np.random.rand(M, K).astype(dtype)
# b_np = np.random.rand(K, N).astype(dtype)
# a_tvm = tvm.nd.array(a_np, device=dev)
# b_tvm = tvm.nd.array(b_np, device=dev)

# # Execute
# result = vm["main"](a_tvm, b_tvm)

# # Testing
# np_result = np.maximum(np.dot(a_np, b_np), 0.0)
# np.testing.assert_allclose(result.numpy(), np_result, rtol=1e-3)
# print("Pass!")

# # Output
# print("TVM output:")
# print(result.numpy()[:5, :5])
# print("NumPy output:")
# print(np_result[:5, :5])