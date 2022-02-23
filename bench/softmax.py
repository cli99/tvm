import tvm
from tvm import relay
import torch
from torch import nn
from tvm import te
from tvm.contrib import graph_executor
import numpy as np
from tvm.contrib import utils
import os
from time import perf_counter
import time

dtype = torch.float16
N = 20480
input_shape = (1, N)
input = torch.randn(input_shape, dtype=dtype)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = nn.Softmax()
if dtype == torch.float16:
    net = net.half()
net.to(device)
net.eval()

input = input.to(device)
script_module = torch.jit.trace(net, input)

input_name = "input"
input_shapes = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(script_module, input_shapes)

# set show_meta_data=True if you want to show meta data
# print(mod.astext(show_meta_data=False))

target = tvm.target.cuda(arch="sm_80")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build_module.build(mod, target=target, params=params)

dev = tvm.cuda()
# create module
input = input.cpu()
module = graph_executor.GraphModule(lib["default"](dev))
# set input and parameters
module.set_input(input_name, input)

n = 100
module.run()

# save the graph, lib and params into separate files
if False:
    path_lib = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"{net.__class__.__name__}.tar")
    lib.export_library(path_lib)
    print(path_lib)
    exit()

acc = 0.0
for i in range(n):
    torch.cuda.synchronize()
    start = time.time()
    module.run()
    torch.cuda.synchronize()
    acc +=  time.time() - start
print(f"Elapsed time for tvm {net.__class__.__name__} using {dtype}: {acc*1000/n} ms")

# get tvm output
output_shape = input_shape
out_tvm = module.get_output(0).asnumpy()

input = input.to(device)
script_module(input)
acc = 0.0
for i in range(n):
    torch.cuda.synchronize()
    start = time.time()
    script_module(input)
    torch.cuda.synchronize()
    acc +=  time.time() - start
print(f"Elapsed time for pytorch {net.__class__.__name__} using {dtype}: {acc*1000/n} ms")

# get pt output
out_pt = script_module(input)
out_pt = out_pt.cpu().detach().numpy()

# check results
np.testing.assert_allclose(out_pt, out_tvm, rtol=1e-3)

