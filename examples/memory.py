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
import multiprocessing as mp
from multiprocessing import Process, Queue

def profile(fun_call, q):
    start_time = time()
    memory, retval = memory_usage(fun_call, max_usage=True, retval=True)
    end_time = time()
    runtime = end_time - start_time
    q.put(((memory[0], runtime), retval))
    
def fork_and_profile(fun_call):
    q = mp.Queue()
    p = Process(target=profile, args=(fun_call, q))
    p.start()
    usage = q.get()
    p.join()
    return usage


def repeated_profile(fun_call, num_tries=3):
    best_usage = None
    retval = None
    for trial in range(num_tries):
        usage, retval = fork_and_profile(fun_call)
        if best_usage is None or usage[1] < best_usage[1]:
            best_usage = usage
    # Units: (MB, Sec)
    return best_usage, retval


def plot_results(results, reference):
    x, y = zip(*results)
    plt.plot(x, y)
    plt.title("Checkpointing - timing vs peak memory consumption")
    plt.xlabel("Peak memory consumption (MB)")
    plt.ylabel("Total runtime (s)")
    xmax = max(max(x), reference[0])*1.1
    ymax = max(max(y), reference[1])*1.1
    plt.ylim(ymin=0, ymax=ymax)
    plt.xlim(xmin=0, xmax=xmax)
    plt.hlines(reference[1], 0, xmax, 'r', 'dashed')
    plt.vlines(reference[0], 0, ymax, 'r', 'dashed')
    plt.savefig("results170.png", bbox_inches='tight')
    

dimensions = (170, 170, 170)
spacing = (10, 10, 10)
maxmem = [1000, 2000, 4000, 8000, 16000, 32000, 64000]

ex_cp = CheckpointedGradientExample(dimensions, spacing=spacing)

print("Calculate gradient from cp")
grad_cp = ex_cp.do_gradient(None)

print("Verify for cp")
ex_cp.do_verify(grad_cp)

results = []
for mm in maxmem:
    results.append(repeated_profile((CheckpointedGradientExample.do_gradient, (ex_cp, mm)))[0])
    
print(results)

ex_full = FullGradientExample(dimensions, spacing=spacing)

print("Calculate gradient from full")
print("Full memory run now")

full_run, grad_full = repeated_profile((FullGradientExample.do_gradient, (ex_full, )))
print(full_run)

assert(np.array_equal(grad_full, grad_cp))

plot_results(results, full_run)

