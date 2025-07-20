import tvm
import tvm.meta_schedule as ms
from tvm.meta_schedule.space_generator import ScheduleFn
from tvm.meta_schedule.search_strategy import ReplayFunc, ReplayTrace, EvolutionarySearch
from tvm.tir import Schedule
from tvm.script import tir as T
from tvm.tir.schedule import Trace
from typing import Optional, List


@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:  # type: ignore
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (32, 32), "float32")
        B = T.match_buffer(b, (32, 32), "float32")
        C = T.match_buffer(c, (32, 32), "float32")
        for i, j, k in T.grid(32, 32, 32):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0  # type: ignore
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def _schedule_matmul(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block=block)
    # get the random tile sizes
    i_0, i_1, i_2, i_3 = sch.split(i, sch.sample_perfect_tile(i, n=4))
    j_0, j_1, j_2, j_3 = sch.split(j, sch.sample_perfect_tile(j, n=4))
    k_0, k_1 = sch.split(k, sch.sample_perfect_tile(k, n=2))
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)


def test_replay_func():
    context = ms.TuneContext(
        mod=Matmul,
        space_generator=ms.space_generator.ScheduleFn(
            sch_fn=_schedule_matmul, postprocs=[]),
        search_strategy=ReplayFunc(),
    )
    strategy: ReplayFunc = context.search_strategy
    spaces = context.space_generator.generate_design_space(context.mod)
    strategy.pre_tuning(
        max_trials=10,
        num_trials_per_iter=5,
        design_spaces=spaces,
    )
    candidates = strategy.generate_measure_candidates()
    # 5, replay `_schedule_matmul` on IRModule 5 times(5 times per iterator) to get 5 candidates
    print(len(candidates))
    # print(candidates[0].sch.mod)
    # print(candidates[1].sch.mod)

    # 2nd iter(generate 5 candidates)
    candidates_2nd_iter = strategy.generate_measure_candidates()
    print(len(candidates_2nd_iter))  # 5, replay `_schedule_matmul` on IR

@ms.utils.derived_object
class DummyMutator(ms.mutator.PyMutator):
    """Dummy Mutator for testing"""

    def _initialize_with_tune_context(self, context: "ms.TuneContext") -> None:
        pass

    def apply(self, trace: Trace, _) -> Optional[Trace]:
        return Trace(trace.insts, {})

    def clone(self):
        return DummyMutator()


def test_evolutionary_search():
    def _schedule_matmul_small(sch: Schedule):
        block = sch.get_block("matmul")
        _, j, k = sch.get_loops(block=block)
        _, _ = sch.split(j, sch.sample_perfect_tile(j, n=2))
        _, _ = sch.split(k, sch.sample_perfect_tile(k, n=2))
    
    max_trials_per_task = 2000
    num_trials_per_iter = 10

    context = ms.TuneContext(
        mod=Matmul,
        space_generator=ms.space_generator.ScheduleFn(
            sch_fn=_schedule_matmul_small,
            sch_rules=[],
            postprocs=[],
            mutator_probs={
                DummyMutator(): 1.0,
            },
        ),
        search_strategy=ms.search_strategy.EvolutionarySearch(
            population_size=5,
            init_measured_ratio=0.1,
            init_min_unmeasured=50,
            genetic_num_iters=3,
            genetic_mutate_prob=0.5,
            genetic_max_fail_count=10,
            eps_greedy=0.9,
        ),
        target=tvm.target.Target("llvm"),
        num_threads=1,  # because we are using a mutator from the python side
    )
    strategy: EvolutionarySearch = context.search_strategy
    strategy.pre_tuning(
        max_trials=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        design_spaces=context.space_generator.generate_design_space(context.mod),
        database=ms.database.MemoryDatabase(),
        cost_model=ms.cost_model.RandomModel(),
    )

    num_trials_each_iter: List[int] = []
    candidates = strategy.generate_measure_candidates()
    while candidates is not None:
        num_trials_each_iter.append(len(candidates))
        runner_results: List[ms.runner.RunnerResult] = []
        for candidate in candidates:
            runner_results.append(
                ms.runner.RunnerResult(
                    run_secs=[0.11, 0.41, 0.54],
                    error_msg=None,
                )
            )
        strategy.notify_runner_results(candidates, runner_results)
        candidates = strategy.generate_measure_candidates()
    strategy.post_tuning()
    print(num_trials_each_iter)


if __name__ == "__main__":
    # test_replay_func()
    test_evolutionary_search()