from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
import tvm.meta_schedule as ms
from tvm.ir import IRModule
import numpy as np
import tvm
from tvm.relax import VirtualMachine
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"


M, K, N = 32, 64, 128
dtype = "float32"


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
        R.func_attr({"global_symbol": "main"})
        cls = MultiTirModule
        with R.dataflow():
            lv0 = R.call_tir(cls.matmul, (a, b), out_sinfo=R.Tensor(
                shape=[M, N], dtype=dtype))
            lv1 = R.call_tir(cls.relu, (lv0), out_sinfo=R.Tensor(
                shape=[M, N], dtype=dtype))
            R.output(lv1)
        return lv1


mod = MultiTirModule
target = tvm.target.Target("nvidia/geforce-rtx-4090")
this_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.join(this_dir, "tuning_logs")

database = ms.relax_integration.tune_relax(
    mod=mod,
    params={},
    target=target,
    max_trials_global=32,
    max_trials_per_task=4,
    work_dir=work_dir,
)

dev = tvm.device("cuda", 0)
vm_ex = ms.relax_integration.compile_relax(database, mod, target, {})
vm = VirtualMachine(vm_ex, dev)

a_np = np.random.rand(32, 64).astype("float32")
b_np = np.random.rand(64, 128).astype("float32")

a_tvm = tvm.nd.array(a_np, device=dev)
b_tvm = tvm.nd.array(b_np, device=dev)

c_tvm = vm["main"](a_tvm, b_tvm)
c_np = np.maximum(a_np @ b_np, 0)

np.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5, atol=1e-5)

sch_matmul = database.query_schedule(IRModule({"main": mod["matmul"]}), target, "main")
sch_matmul.mod.show()

sch_relu = database.query_schedule(IRModule({"main": mod["relu"]}), target, "main")
sch_relu.mod.show()
