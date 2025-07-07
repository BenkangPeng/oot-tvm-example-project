import tvm
import tvm.meta_schedule as ms
from tvm.script import tir as T
from tvm.tir import Schedule
from tvm.meta_schedule.space_generator import ScheduleFn, SpaceGeneratorUnion

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument
# fmt: off

@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

def manual_schedule(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block=block)
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=[2, 4, 64, 2])
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=[4, 64, 2, 2])
    k_0, k_1 = sch.split(loop=k, factors=[32, 32])
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)


def test_schedule_fn():
    mod = Matmul
    schFn = ScheduleFn(sch_fn=manual_schedule)

    # design space is a list of schedules
    design_spaces = schFn.generate_design_space(mod)
    print(design_spaces)
    print(design_spaces[0].trace)

def test_space_generator_union():
    mod = Matmul
    schFn = ScheduleFn(sch_fn=manual_schedule)
    schFn2 = ScheduleFn(sch_fn=manual_schedule)
    space_generator_union = SpaceGeneratorUnion([schFn, schFn2])
    design_spaces = space_generator_union.generate_design_space(mod)
    print(design_spaces)

if __name__ == "__main__":
    test_schedule_fn()
    test_space_generator_union()