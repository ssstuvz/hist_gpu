# Program creates 2 vectors size of N and implement slow kernel to them.

import ray
import cupy as cp
import numpy as np
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

# PARAMETERS TO CHANGE 

N = 10**3 # Size of arrays
num_subvectors = 2 # Number of partitions to divide input arrays

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

# Cluster initialization

ray.init(num_cpus=num_cpus, num_gpus=num_gpus, include_dashboard=False, ignore_reinit_error=True)
print("Initiation done")

num_gpus_per_worker = num_gpus/num_subvectors

# Decalration for Ray function without profinig overhead

@ray.remote(num_gpus=num_gpus_per_worker)
def add_vectors_on_gpu_no_profile(vector_a, vector_b):

    # H2D procedure and timing
    gpu_vector_a = cp.asarray(vector_a)
    gpu_vector_b = cp.asarray(vector_b)

    # Calculation procedure
    d_result = slow_kernel(gpu_vector_a, gpu_vector_b)

    # D2H procedure
    res = cp.asnumpy(d_result)     

    return res
                   
# Creating 2 vectors
                   
large_vector1 = np.linspace(0,1,N)
large_vector2 = np.linspace(1,2,N)

# Splitting them to achieve parallelism

sub_vectors = np.array_split(large_vector1, num_subvectors) # This is list of lists sizes of 
sub_vectors2 = np.array_split(large_vector2, num_subvectors) # (num_subvectors, N/num_subvectors)

# Each subvector for both vectors is send into function in pairs. Ray scheduling

results = [add_vectors_on_gpu_no_profile.remote(sub_v1, sub_v2) for sub_v1, sub_v2 in zip(sub_vectors, sub_vectors2)]

# Getting results from Ray

final_results = ray.get(results)

slow_kernel_results = np.concatenate(final_results) # For our example result should be [-1, -1, -1, ..., -1]
#print(slow_kernel_results)

ray.shutdown()
print("RAY Work has been done!")

# File write option
# Dont forget to change path!

# info = f"""Work is done on {gpu}, num_gpus = {num_gpus} 
# Here (copy + calc + back + cpu in + cpu out) DATA N_subv vs N"""

# with open(f"ray_results/ray_slowkernel_results_{gpu}_{num_gpus}_test.pickle", 'wb') as file:
#     pickle.dump(info, file)
#     pickle.dump(slow_kernel_results, file)
    
# print("_______ALL DATA STORAGE AND CALCULATION IS DONE_______")


