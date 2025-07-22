from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
import tvm.meta_schedule as ms
import numpy as np
import tvm
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"


M, K, N = 32, 64, 128
dtype = "float32"

USE_TIR_OPT_ONLY = True


@I.ir_module
class SingleTirModule:
    @T.prim_func
    def matmul_relu(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "matmul_relu", "tir.noalias": True})
        A = T.match_buffer(a, [M, K], dtype=dtype)
        B = T.match_buffer(b, [K, N], dtype=dtype)
        Y = T.alloc_buffer([M, N], dtype=dtype)
        C = T.match_buffer(c, [M, N], dtype=dtype)
        for i, j, k in T.grid(M, N, K):  # type: ignore
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])  # type: ignore
                with T.init():
                    Y[vi, vj] = T.cast(0.0, dtype=dtype)  # type: ignore
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]

        for i, j in T.grid(M, N):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))

    @R.function
    def main(a: R.Tensor(shape=[M, K], dtype=dtype), b: R.Tensor(shape=[K, N], dtype=dtype)) -> R.Tensor(shape=[M, N], dtype=dtype):  # type: ignore
        cls = SingleTirModule
        with R.dataflow():
            lv0 = R.call_tir(cls.matmul_relu, (a, b),
                             out_sinfo=R.Tensor(shape=[M, N], dtype=dtype))
            R.output(lv0)
        return lv0


@I.ir_module
class MultiTirModule:
    @T.prim_func
    def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
        A = T.match_buffer(a, [M, K], dtype=dtype)
        B = T.match_buffer(b, [K, N], dtype=dtype)
        C = T.match_buffer(c, [M, N], dtype=dtype)
        for i, j, k in T.grid(M, N, K):  # type: ignore
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])  # type: ignore
                with T.init():
                    C[vi, vj] = T.cast(0.0, dtype=dtype)  # type: ignore
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    @T.prim_func
    def relu(c: T.handle, y: T.handle) -> None:
        T.func_attr({"global_symbol": "relu", "tir.noalias": True})
        C = T.match_buffer(c, [M, N], dtype=dtype)
        Y = T.match_buffer(y, [M, N], dtype=dtype)
        for i, j in T.grid(M, N):
            with T.block("Y"):
                vi, vj = T.axis.remap("SS", [i, j])
                Y[vi, vj] = T.max(C[vi, vj], T.cast(0.0, dtype=dtype))

    @R.function
    def main(
        # type: ignore
        a: R.Tensor(shape=[M, K], dtype=dtype), b: R.Tensor(shape=[K, N], dtype=dtype)
    ) -> R.Tensor(shape=[M, N], dtype=dtype):  # type: ignore
        # R.func_attr({"global_symbol": "main"})
        cls = MultiTirModule
        with R.dataflow():
            lv0 = R.call_tir(cls.matmul, (a, b), out_sinfo=R.Tensor(
                shape=[M, N], dtype=dtype))
            lv1 = R.call_tir(cls.relu, (lv0), out_sinfo=R.Tensor(
                shape=[M, N], dtype=dtype))
            R.output(lv1)
        return lv1


mod = SingleTirModule
# mod = MultiTirModule
# target = tvm.target.Target("nvidia/geforce-rtx-4090")
# mod = tvm.relax.get_default_pipeline(target)(mod)
# mod.show()
# mod = tvm.relax.get_pipeline("zero")(mod)
# mod.show()
# exit()


######################################################################
# Define the target
# Remember to check the variable `$CUDA_VISIBLE_DEVICES` in the terminal
# Change to your target device
target = tvm.target.Target("nvidia/geforce-rtx-4090")
total_trials = 16
max_trials_per_task = 4
this_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.join(this_dir, "tuning_logs")


######################################################################
# Use relax.get_pipeline
# BUG `static_shape_tuning` will return a not-tuned mod.
# In `evolutionary_search.cc` Line521, `results` still be empty after running `support::parallel_for_dynamic`.
tvm_mod = tvm.relax.get_pipeline("static_shape_tuning", target=target, total_trials=total_trials,
                                 max_trials_per_task=max_trials_per_task, work_dir=work_dir)(mod)
tvm_mod.show()

######################################################################
# Build
dev = tvm.device("cuda", 0)
ex = tvm.relax.build(tvm_mod, "cuda")
vm = tvm.relax.VirtualMachine(ex, dev)

# Prepare Data
a_np = np.random.rand(M, K).astype(dtype)
b_np = np.random.rand(K, N).astype(dtype)
a_tvm = tvm.nd.array(a_np, device=dev)
b_tvm = tvm.nd.array(b_np, device=dev)

# Execute
result = vm["main"](a_tvm, b_tvm)

# Testing
np_result = np.maximum(np.matmul(a_np, b_np), 0.0)
np.testing.assert_allclose(result.numpy(), np_result, rtol=1e-3)
print("Pass!")

# # Output
# print("TVM output:")
# print(result.numpy()[:5, :5])
# print("NumPy output:")
# print(np_result[:5, :5])
