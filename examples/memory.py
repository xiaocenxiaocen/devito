import matplotlib
matplotlib.use('Agg')
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
    

dimensions = (450, 450, 450)
spacing = (75, 75, 75)
maxmem = [2500, 5000, 10000, 20000, 40000, 80000 ]

results = []
# Gradient Run
# results.append(time_and_memory_profile((gradient_run, (dimensions, ))))

for mm in maxmem:
    results.append(time_and_memory_profile((cp_run, (dimensions, ), {'maxmem': mm, 'spacing': spacing})))

print(results)

print("Full memory run now")

full_run = time_and_memory_profile((gradient_run, (dimensions, ), {'spacing': spacing}))
print(full_run)

plot_results(results, full_run)

