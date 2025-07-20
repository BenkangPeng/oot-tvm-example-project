import tvm
import torch
from tvm.relax.frontend.torch import from_fx
import numpy as np
import time
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda", 0)
        self.linear = torch.nn.Linear(
            in_features=128, out_features=128, bias=True, device=self.device)
        self.relu = torch.nn.ReLU()

        self.linear2 = torch.nn.Linear(
            in_features=128, out_features=10, bias=True, device=self.device)
        self.relu2 = torch.nn.ReLU()

    def forward(self, input):
        x1 = self.relu(self.linear(input))
        x2 = self.relu2(self.linear2(x1))
        return x2


torch_model = MyModule()
input_info = [((10, 128), torch.float32)]
input_tensors = [
    torch.randn(*shape, dtype=dtype)
    for shape, dtype in input_info
]

# Use FX tracer to trace the PyTorch model.
graph_module = torch.fx.symbolic_trace(torch_model)

mod = from_fx(graph_module, input_info)
mod.show()

# relax tuning
target = tvm.target.Target("nvidia/geforce-rtx-4090")
TOTAL_TRIALS = 0
this_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.join(this_dir, "tune_tmp")
mod = tvm.relax.get_pipeline("static_shape_tuning", target=target, total_trials=TOTAL_TRIALS, work_dir=work_dir)(mod)

mod.show()


##############Test Performance########################
ex = tvm.compile(mod, target="cuda")
dev = tvm.device("cuda", 0)
vm = tvm.relax.VirtualMachine(ex, dev)

evaluator = vm.time_evaluator("main", dev, number=10)
gpu_data = tvm.nd.array(input_tensors[0], dev)
print(f'TVM cost: {(evaluator(gpu_data).mean * 1000):.4f} ms')


torch_model.eval()
dev = torch.device("cuda", 0)
input_tensor = input_tensors[0].to(dev)
# Warm up
with torch.no_grad():
    for _ in range(10):
        _ = torch_model(input_tensor)

# Measure execution time
num_runs = 10
start_time = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        _ = torch_model(input_tensor)
end_time = time.time()
avg_time_ms = (end_time - start_time) * 1000 / num_runs
print(f"Torch cost: {avg_time_ms:.4f} ms")
