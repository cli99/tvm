import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.contrib import cublas

target = tvm.target.Target('cuda')

N = 2048
dtype = 'float32'
A = te.placeholder((N, N), name='data', dtype=dtype)
B = te.placeholder((N, N), name='kernel', dtype=dtype)
C = cublas.matmul(A, B, False, True, dtype=dtype)

sch = te.create_schedule(C.op)
args = [A, B, C]
func = tvm.build(sch, args, target)

# Check correctness
data_np = np.random.uniform(size=(N, N)).astype(np.float32)
weight_np = np.random.uniform(size=(N, N)).astype(np.float32)
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
print("Speed: %.3f TFLOPS" % (2 * (N**3) / time / 1e12))
