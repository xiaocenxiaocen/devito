import matplotlib
matplotlib.use('Agg')
from checkpointing.example import CheckpointedGradientExample
from seismic.acoustic.gradient_example import FullGradientExample
from memory_profiler import memory_usage
from time import time
import sys
import gc
import matplotlib.pyplot as plt
import numpy as np

def time_and_memory_profile(fun_call, num_tries=3):
    maxmem = 0
    timing = sys.maxsize
    for trial in range(num_tries):
        start_time = time()
        memory = memory_usage(fun_call)
        end_time = time()
        timing = min(timing, (end_time-start_time))
        maxmem = max(max(memory), maxmem)
        gc.collect()
    # Units: (MB, Sec)
    return maxmem, timing

def plot_results(results, reference):
    x, y = zip(*results)
    plt.plot(x, y)
    plt.title("Checkpointing - timing vs peak memory consumption")
    plt.xlabel("Peak memory consumption (MB)")
    plt.ylabel("Total runtime (s)")
    plt.ylim(ymin=0, ymax=max(y)*1.1)
    plt.xlim(xmin=0, xmax=max(x)*1.1)
    plt.hlines(reference[1], 0, max(x)*1.1, 'r', 'dashed')
    plt.vlines(reference[0], 0, max(y)*1.1, 'r', 'dashed')
    plt.savefig("results.png", bbox_inches='tight')
    

dimensions = (230, 230, 230)
spacing = (10, 10, 10)
maxmem = [1000, 2000, 4000, 8000, 16000, 32000, 64000]

ex_cp = CheckpointedGradientExample(dimensions, spacing=spacing)
ex_full = FullGradientExample(dimensions, spacing=spacing)

print("Calculate gradient from full")
grad_full = ex_full.do_gradient()

print("Calculate gradient from cp")
grad_cp = ex_cp.do_gradient(None)

assert(np.array_equal(grad_full, grad_cp))

print("Verify for full")
ex_full.do_verify(grad_full)

print("Verify for cp")
ex_cp.do_verify(grad_cp)

print("Full memory run now")

full_run = time_and_memory_profile((FullGradientExample.do_gradient, (ex_full, )))
print(full_run)


results = []
for mm in maxmem:
    results.append(time_and_memory_profile((CheckpointedGradientExample.do_gradient, (ex_cp, mm))))

print(results)

plot_results(results, full_run)

