import tvm
import tvm.meta_schedule as ms
import tvm.script.tir as T
from tvm.tir import Schedule


@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(  # type: ignore
        a: T.handle,
        b: T.handle,
        c: T.handle,
    ) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0  # type: ignore
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

def manual_schedule(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block=block)
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=[2, 4, 64, 2])
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=[4, 64, 2, 2])
    k_0, k_1 = sch.split(loop=k, factors=[32, 32])
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)


def test_round_robin():
    db = ms.database.MemoryDatabase()
    round_robin = ms.task_scheduler.RoundRobin()
    tune_ctx = ms.TuneContext(
        MatmulModule,
        target=tvm.target.Target("llvm"),
        space_generator=manual_schedule,
        search_strategy=,
    )
    round_robin.tune(
        [
            ms.TuneContext(
                MatmulModule,
                target=tvm.target.Target("llvm"),
                space_generator=_schedule_matmul,
                search_strategy=ms.search_strategy.ReplayTrace(),
