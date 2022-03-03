import os
import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
import torch

N = 2048
input_shape = (2, N)

@auto_scheduler.register_workload
def softmax_layer():
    data = te.placeholder(input_shape, name='data')
    out = topi.nn.softmax(data)
    return [data, out]

target = tvm.target.Target("cuda")

task = auto_scheduler.SearchTask(
    func=softmax_layer, target=target
)

# Inspect the computational graph
# print("Computational DAG:")
# print(task.compute_dag)

s = "-".join(map(str, input_shape))
log_file = f"softmax-{s}.json"

if not os.path.exists(log_file):
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=10,  # change this to 1000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    # Run auto-tuning (search)
    task.tune(tune_option)
    del measure_ctx

# Apply the best schedule
sch, args = task.apply_best(log_file)
func = tvm.build(sch, args, target)

# Check correctness
data_np = np.random.uniform(size=input_shape).astype(np.float32)
ctx = tvm.cuda()
data_tvm = tvm.nd.array(data_np, device=ctx)
out_tvm = tvm.nd.array(np.ones_like(data_np), device=ctx)
func(data_tvm, out_tvm)

# Check results
data_pt = torch.from_numpy(data_np)
net = torch.nn.Softmax()
out_pt = net(data_pt)
np.testing.assert_allclose(out_pt.numpy(), out_tvm.asnumpy(), rtol=1e-3)

# Evaluate execution time
evaluator = func.time_evaluator(func.entry_name, ctx, number=100, repeat=10)
time = np.median(evaluator(data_tvm, out_tvm).results)
print("shape", data_np.shape)
print("Execution time of this operator: %.3f ms" % (time * 1000))
print("Speed: %.3f TFLOPS" % (2 * (N**2) / time / 1e12))
