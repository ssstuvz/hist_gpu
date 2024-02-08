# Program creates 2 vectors size of N and implement slow kernel to them.
# General walkthrough:
# For each number of partitions n_subv from N_subv slow kernel is applied for each size of aarays N from N_arr

import ray
import cupy as cp
import time
import numpy as np
from scipy.stats import trim_mean
import os
import sys
import pickle
os.environ["RAY_DEDUP_LOGS"] = "0"

print("Import done, Hello")

# Launch the program using "python3 ray_gpu_slow_kernel.py N M gpuname"
# Where N is the number of GPUs, M is the number of CPU's cores 
# And 'gpuname' is the name for GPU (used only to name outputfile)

num_gpus = int(sys.argv[1])
num_cpus = int(sys.argv[2])
gpu = sys.argv[3]

repeat = 20 # Code repeats 20 times to gain statistics
cut_off = 0.1 # Value to trim_mean - cutting 10% percentile from both sides

N_arr = [10**i for i in range(8, 9)] # Size of arrays
N_subv = [2**i for i in range(0,6)] # Number of partitions to divide input arrays

slow_kernel = cp.ElementwiseKernel(
   'T d_x, T d_y',
   'T d_z',
   '''
   for(i = 0; i < 10000; ++i)
   {
    d_z = d_x - d_y;
   }
   ''',
   'squared_diff')

ray_time = np.zeros((len(N_subv), len(N_arr), 5)) #copy, calc, back, cpu in, cpu out

ray.init(num_cpus=num_cpus, num_gpus=num_gpus, include_dashboard=False, ignore_reinit_error=True)
print("Initiation done")
                        
for n_subv in range(len(N_subv)):
    
    print(f"starting for {N_subv[n_subv]}")
    
    num_subvectors = N_subv[n_subv]
    num_gpus_per_worker = num_gpus/num_subvectors
    
    # Declaration of Ray function is done for each n_subv to implement correct "fractional" GPU usage for Ray

    @ray.remote(num_gpus=num_gpus_per_worker)
    def add_vectors_on_gpu(vector_a, vector_b):

        # array to contain time estimation to gain statistics
        copy_time = [] # time to copy from Host to Device (H2D)
        calc_time = [] # H2D + add 2 vecs calc time (CALC)
        back_time = [] # H2D + CALC + time to copy back from Device to Host (D2H)
        cpu_time = [] # time measure with time.time() CPU method - only for comparison

        for i in range(repeat):

            # Setting start time points

            start_gpu_time = time.time()
            start_event = cp.cuda.Event()
            end_event_1 = cp.cuda.Event()
            end_event_2 = cp.cuda.Event()
            end_event_3 = cp.cuda.Event()

            # H2D procedure and timing

            start_event.record()
            gpu_vector_a = cp.asarray(vector_a)
            gpu_vector_b = cp.asarray(vector_b)
            end_event_1.record()
            end_event_1.synchronize()

            # Calculation procedure and H2D + CALC timing

            d_result = slow_kernel(gpu_vector_a, gpu_vector_b)
            end_event_2.record()
            end_event_2.synchronize()

            # D2H procedure and H2D + CALC + D2H timing
            res = cp.asnumpy(d_result)     
            end_event_3.record()
            end_event_3.synchronize()

            end_gpu_time = time.time()

            # Time data collection
            copy_time.append(cp.cuda.get_elapsed_time(start_event, end_event_1))
            calc_time.append(cp.cuda.get_elapsed_time(start_event, end_event_2))
            back_time.append(cp.cuda.get_elapsed_time(start_event, end_event_3))# Время в миллисекундах
            cpu_time.append((end_gpu_time - start_gpu_time)*1000)


        # After all repeats, calculate trim mean of each timing    
        append_time = trim_mean(cpu_time, cut_off)
        avg_copy = trim_mean(copy_time, cut_off)
        avg_calc = trim_mean(calc_time, cut_off)
        avg_back = trim_mean(back_time, cut_off)


        return res, avg_copy, avg_calc, avg_back, append_time
    
    # Decalration for Ray function without profinig overhead
    
    @ray.remote(num_gpus=num_gpus_per_worker)
    def add_vectors_on_gpu_no_profile(vector_a, vector_b):

        # H2D procedure and timing
        gpu_vector_a = cp.asarray(vector_a)
        gpu_vector_b = cp.asarray(vector_b)

        # Calculation procedure and H2D + CALC timing
        d_result = slow_kernel(gpu_vector_a, gpu_vector_b)

        # D2H procedure and H2D + CALC + D2H timing
        res = cp.asnumpy(d_result)     

        return res
                        
                        
    for N_idx in range(len(N_arr)):
        
        N = N_arr[N_idx]
        large_vector1 = np.linspace(0,1,N)
        large_vector2 = np.linspace(1,2,N)
        # Starting Ray instance

        start = time.time()
        
        sub_vectors = np.array_split(large_vector1, num_subvectors) # This is list of lists sizes of 
        sub_vectors2 = np.array_split(large_vector2, num_subvectors) # (num_subvectors, N/num_subvectors)

        results = [add_vectors_on_gpu_no_profile.remote(sub_v1, sub_v2) for sub_v1, sub_v2 in zip(sub_vectors, sub_vectors2)]
        final_results = ray.get(results)

        end = time.time()
        print(f"time elapsed from CPU for 1 loop for 10**{np.log10(N)}: {(end - start)*1000} ms")
        
        results = [add_vectors_on_gpu.remote(sub_v1, sub_v2) for sub_v1, sub_v2 in zip(sub_vectors, sub_vectors2)]
        final_results = ray.get(results)
        
        # Post production for profiling
        
        copy_time = np.array([res[1] for res in final_results])
        calc_time = np.array([res[2] for res in final_results])
        back_time = np.array([res[3] for res in final_results])
        append_time = np.array([res[4] for res in final_results])
        
        # this thing make right time in rigth place
        
        back_time = back_time - calc_time
        calc_time = calc_time - copy_time
        
        ray_time[n_subv, N_idx, 0] = np.mean(copy_time)
        ray_time[n_subv, N_idx, 1] = np.mean(calc_time)
        ray_time[n_subv, N_idx, 2] = np.mean(back_time)
        ray_time[n_subv, N_idx, 3] = np.mean(append_time)
        ray_time[n_subv, N_idx, 4] = (end - start)*1000/repeat
        
        
    print(f" For n_subv = {N_subv[n_subv]} is done!")
    
ray.shutdown()
print("RAY Work has been done!")


# File write option

info = f"""Work is done on {gpu}, num_gpus = {num_gpus} 
Here (copy + calc + back + cpu in + cpu out) DATA N_subv vs N"""

with open(f"ray_results/ray_slowkernel_results_{gpu}_{num_gpus}_test.pickle", 'wb') as file:
    pickle.dump(info, file)
    pickle.dump(ray_time, file)
    
print("_______ALL DATA STORAGE AND CALCULATION IS DONE_______")


