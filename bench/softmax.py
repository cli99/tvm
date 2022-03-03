import tvm
from tvm import relay, auto_scheduler
import torch
from torch import nn
from tvm import te
from tvm.contrib import graph_executor
import numpy as np
from tvm.contrib import utils
import os
from time import perf_counter
import time
from tvm.driver import tvmc

USE_ANSOR=True
dtype = torch.float32
if dtype == torch.float16:
    tp_dtype = "float16"
else:
    tp_dtype = "float32"

def auto_scheduler_tune(mod, params, target, log_file):
    if os.path.exists(log_file):
        os.remove(log_file)

    layout = "NHWC"
    n_trials = 1000

    if "cpu" in target.keys:
        tuning_opt = auto_scheduler.TuningOptions(
            num_measure_trials=n_trials,
            runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
    else:
        min_repeat_ms = 0
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(
            repeat=1, min_repeat_ms=min_repeat_ms, timeout=10
        )
        tuning_opt = auto_scheduler.TuningOptions(
            num_measure_trials=n_trials,
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )

    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    for idx, task in enumerate(tasks):
        print(
            "========== Task %d  (workload key: %s) =========="
            % (idx, task.workload_key)
        )
        print(task.compute_dag)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tuner.tune(tuning_opt)

input_shape = (2, 2048, 2048)
np_input = (np.random.uniform(size=input_shape)).astype(tp_dtype)
pt_input = torch.from_numpy(np_input)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = nn.Softmax(dim=-1)
if dtype == torch.float16:
    net = net.half()
net.to(device)
net.eval()

pt_input = pt_input.to(device)
script_module = torch.jit.trace(net, pt_input)

input_name = "input"
input_shapes = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(script_module, input_shapes)

target = tvm.target.cuda(arch="sm_80")

s = "-".join(map(str, input_shape))
log_file = f"softmax-{s}.json"

if USE_ANSOR:
    if not os.path.exists(log_file):
        auto_scheduler_tune(mod, params, target, log_file=log_file)

    with auto_scheduler.ApplyHistoryBest(log_file):
        print("Apply history best")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)
else:
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build_module.build(mod, target=target, params=params)

# Create graph executor
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
tvm_input = tvm.nd.array(np_input)
module.set_input("input", tvm_input)

n = 30

# Evaluate
print("Evaluate inference time cost...")
print(module.benchmark(dev, repeat=n))

# save the graph, lib and params into separate files
if False:
    path_lib = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"{net.__class__.__name__}.tar")
    lib.export_library(path_lib)
    print(path_lib)
    exit()

# get tvm output
output_shape = input_shape
out_tvm = module.get_output(0).asnumpy()

script_module(pt_input)
acc = 0.0
for i in range(n):
    torch.cuda.synchronize()
    start = time.time()
    script_module(pt_input)
    torch.cuda.synchronize()
    acc +=  time.time() - start
print(f"Elapsed time for pytorch {net.__class__.__name__} using {dtype}: {acc*1000/n} ms")

# get pt output
out_pt = script_module(pt_input)
out_pt = out_pt.cpu().detach().numpy()

# check results
np.testing.assert_allclose(out_pt, out_tvm, rtol=1e-2)

