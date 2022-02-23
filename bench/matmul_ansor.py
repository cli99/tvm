import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi

M = 2048
K = 2048
N = 2048

@auto_scheduler.register_workload
def dense_layer(M, K, N):
    data = te.placeholder((M, K), name='A')
    kernel = te.placeholder((K, N), name="kernel")
    out = topi.nn.dense(data, kernel)
    return [data, kernel, out]

target = tvm.target.Target("cuda")
task = auto_scheduler.SearchTask(
    func=dense_layer, args=(M, K, N), target=target
)

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

log_file = f"matmul-{M}-{K}-{N}.json"
if not os.path.exists(log_file):
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000,  # change this to 1000 to achieve the best performance
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
data_np = np.random.uniform(size=(M, K)).astype(np.float32)
weight_np = np.random.uniform(size=(K, N)).astype(np.float32)
out_np = np.matmul(data_np, weight_np.T)
ctx = tvm.cuda()
data_tvm = tvm.nd.array(data_np, device=ctx)
weight_tvm = tvm.nd.array(weight_np, device=ctx)
out_tvm = tvm.nd.array(np.ones_like(data_np), device=ctx)
func(data_tvm, weight_tvm, out_tvm)

# Check results
np.testing.assert_allclose(out_np, out_tvm.asnumpy(), rtol=1e-3)

# Evaluate execution time
evaluator = func.time_evaluator(func.entry_name, ctx, number=100, repeat=10)
time = np.median(evaluator(data_tvm, weight_tvm, out_tvm).results)
print("shape", data_np.shape, weight_np.shape)
print("Execution time of this operator: %.3f ms" % (time * 1000))
print("Speed: %.3f TFLOPS" % (2 * (M*K*N) / time / 1e12))
