from checkpointing.example import run as cp_run
from seismic.acoustic.gradient_example import run as gradient_run
from memory_profiler import memory_usage
from time import time
import sys
import gc
import matplotlib.pyplot as plt

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

def plot_results(results):
    x, y = zip(*results)
    plt.plot(x, y)
    plt.title("Checkpointing - timing vs peak memory consumption")
    plt.xlabel("Peak memory consumption (MB)")
    plt.ylabel("Total runtime (s)")
    plt.savefig("foo.png", bbox_inches='tight')
    

dimensions = (20, 20, 20)

maxmem = [250, 500, 750, 1000, 1250, 1500]

results = []
# Gradient Run
# results.append(time_and_memory_profile((gradient_run, (dimensions, ))))

for mm in maxmem:
    results.append(time_and_memory_profile((cp_run, (dimensions, ), {'maxmem': mm})))

print(results)

plot_results(results)

print("Full memory run now")

print(time_and_memory_profile((gradient_run, (dimensions, ))))


