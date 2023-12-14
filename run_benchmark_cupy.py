import os
import sys
import cupy as cp
import argparse

from cupyx import jit
from cupyx.profiler import benchmark

try:
	from astropy.table import Table
except:
	print ('No astropy package is found! Please install: pip install astropy')
	sys.exit()


# Option 1: jit translation of a pythonic code to CUDA/С, global atomic_add()
@jit.rawkernel()
def histogram_jit(data, bins_data, n_bins, min_x, max_x, data_size):

    gid    = jit.threadIdx.x + jit.blockIdx.x * jit.blockDim.x
    stride = jit.blockDim.x * jit.gridDim.x
    
    dbin   = (max_x-min_x)/n_bins

    while gid < data_size:
        number = data[gid]
        index  = cp.int32(number / dbin)
        jit.atomic_add(bins_data, index, 1)
        gid   += stride

# Option 2: Raw kernel, CUDA/C, global atomic_add
histogram_raw = cp.RawKernel(r'''
	                         extern "C" __global__
                             void histogram_raw( float* data, int* bins_data, int n_bins, float min_x, float max_x, int data_size )
                              {
                                int gid    = threadIdx.x + blockIdx.x * blockDim.x;
                                int stride = blockDim.x * gridDim.x;
                                
                                float bin_size = (max_x-min_x) / n_bins;

                                while (gid < data_size)
                                 {
                                   int index = (int)( data[gid] / bin_size );
                                  
                                   atomicAdd(&bins_data[index], 1);

                                   gid += stride;
                                 }
                              }
                             ''', 'histogram_raw')

# Option 3. ElementWise Kernel, CUDA/C, global memory atomic_add
histogram_elw = cp.ElementwiseKernel(
                                      'float32 data, int32 n_bins, float32 x_min, float32 x_max', # input
                                      'raw T bins_data',                                          # output

                                      '''
                                      float bin_size = (x_max - x_min) / n_bins;

                                      int index = (int)( data / bin_size);

                                      atomicAdd(&bins_data[index], 1);

                                      ''', 'histogram_elw'
                                    )

# Option 4. RawKernel, CUDA/C, shared memory atomic_add
histogram_raw_shd = cp.RawKernel(r'''
	                         extern "C" __global__
                             void histogram_raw_shd( float* data, int* bins_data, int n_bins, float min_x, float max_x, int data_size )
                              {
                                extern __shared__ unsigned int temp[];
                                
                                if (threadIdx.x < n_bins)
                                    {
                                      temp[threadIdx.x] = 0;
                                    }
                                __syncthreads();
                                
                                int gid    = threadIdx.x + blockIdx.x * blockDim.x;
                                int stride = blockDim.x * gridDim.x;         
                                
                                float bin_size = (max_x-min_x) / n_bins;

                                while (gid < data_size)
                                 {
                                   int index = (int)( data[gid] / bin_size );
                                  
                                   atomicAdd(&temp[index], 1);

                                   gid += stride;
                                 }
                                
                                __syncthreads();
                                if (threadIdx.x < n_bins)
                                {
                                  atomicAdd( &(bins_data[threadIdx.x]), temp[threadIdx.x] );
                                }
                             
                              }''','histogram_raw_shd'
                             )

# Now run benchmark tests. 
to_microsec = pow(10,6)

###################################################################################################################
# Test 1. Time_gpu VS N_data
# Check the execution times as a function of number of data

def test_time_vs_ndata( gpu='T4', verbose=False ):
	tab_out = Table()
	fname   = 'tab_Ndata_vs_time_GPU={}.txt'.format( gpu )
	fmts    = {'N_data' :  '%d',
	           't_jit_m':  '%.6f',
	           't_jit_s':  '%.6f',
	           't_raw_m':  '%.6f',
	           't_raw_s':  '%.6f',
	           't_shd_m':  '%.6f',
	           't_shd_s':  '%.6f',
	           't_elw_m':  '%.6f',
	           't_elw_s':  '%.6f',
	           't_npy_m':  '%.6f',
	           't_npy_s':  '%.6f',
	           't_cpy_m':  '%.6f',
	           't_cpy_s':  '%.6f'}

	n_bins    = 10
	n_threads = 1024
	max_x     = cp.float32(1.0)
	min_x     = cp.float32(0.0)
	smem      = n_bins * cp.dtype(cp.int32).itemsize

	exponent  = np.arange(1,7.1,1)    
	datas     = np.array([pow(10,x) for x in exponent],dtype=np.int32)

	t_jit_m, t_raw_m, t_elw_m, t_npy_m, t_cpy_m, t_shd_m = [],[],[],[],[],[]
	t_jit_s, t_raw_s, t_elw_s, t_npy_s, t_cpy_s, t_shd_s = [],[],[],[],[],[]

	for n_data in datas:
	  d_bins_data_jit = cp.zeros(n_bins, dtype=cp.int32)
	  d_bins_data_raw = cp.zeros(n_bins, dtype=cp.int32)
	  d_bins_data_elw = cp.zeros(n_bins, dtype=cp.int32)

	  cp.random.seed(42)
	  d_data = cp.random.rand(n_data).astype(cp.float32)
	  np.random.seed(42)
	  h_data = np.random.rand(n_data).astype(np.float32)

	  exe_gpu_jit = benchmark( histogram_jit, ((1,1,1), (n_threads,1,1), (d_data, d_bins_data_jit, n_bins, min_x, max_x, d_data.size )), n_repeat=5000, n_warmup=100 )
	  exe_gpu_raw = benchmark( histogram_raw, ((1,1,1), (n_threads,1,1), (d_data, d_bins_data_raw, n_bins, min_x, max_x, d_data.size )), n_repeat=5000, n_warmup=100 )
	  exe_gpu_shd = benchmark( histogram_raw_shd, ((1,1,1), (n_threads,1,1), (d_data, d_bins_data_raw, n_bins, min_x, max_x, d_data.size )), kwargs={'shared_mem':smem}, n_repeat=5000, n_warmup=100 )
	  exe_gpu_elw = benchmark( histogram_elw, args=((d_data, n_bins, min_x, max_x, d_bins_data_elw)), kwargs={'block_size':n_threads},   n_repeat=5000, n_warmup=100 )
	  exe_gpu_npy = benchmark( np.histogram,  args=((h_data, n_bins, (min_x, max_x))),   n_repeat=5000, n_warmup=100 )
	  exe_gpu_cpy = benchmark( cp.histogram,  args=((d_data, n_bins, (min_x, max_x))),   n_repeat=5000, n_warmup=100 )
	  
	  m_jit, s_jit = np.average(exe_gpu_jit.gpu_times), np.std(exe_gpu_jit.gpu_times)
	  m_raw, s_raw = np.average(exe_gpu_raw.gpu_times), np.std(exe_gpu_raw.gpu_times)
	  m_shd, s_shd = np.average(exe_gpu_shd.gpu_times), np.std(exe_gpu_shd.gpu_times)
	  m_elw, s_elw = np.average(exe_gpu_elw.gpu_times), np.std(exe_gpu_elw.gpu_times)
	  m_npy, s_npy = np.average(exe_gpu_npy.cpu_times), np.std(exe_gpu_npy.cpu_times)
	  m_cpy, s_cpy = np.average(exe_gpu_cpy.gpu_times), np.std(exe_gpu_npy.gpu_times)
	    
	  if verbose == True:
		  print ('N_data = {:d}'.format(n_data))
		  print ('Jit version:        t={:.6f}+/-{:.6f} us'.format(m_jit*to_microsec, s_jit*to_microsec))
		  print ('Raw version:        t={:.6f}+/-{:.6f} us'.format(m_raw*to_microsec, s_raw*to_microsec))
		  print ('ELW version:        t={:.6f}+/-{:.6f} us'.format(m_elw*to_microsec, s_elw*to_microsec))
		  print ('RAW shared version: t={:.6f}+/-{:.6f} us'.format(m_shd*to_microsec, s_shd*to_microsec))
		  print ('NPY version:        t={:.6f}+/-{:.6f} us'.format(m_npy*to_microsec, s_npy*to_microsec))
		  print ('CPY version:        t={:.6f}+/-{:.6f} us'.format(m_cpy*to_microsec, s_cpy*to_microsec))

	  d_bins_data_jit = cp.zeros(n_bins, dtype=cp.int32)
	  d_bins_data_raw = cp.zeros(n_bins, dtype=cp.int32)
	  d_bins_data_elw = cp.zeros(n_bins, dtype=cp.int32)
	  d_bins_data_shd = cp.zeros(n_bins, dtype=cp.int32)

	  t_jit_m.append(m_jit*to_microsec)
	  t_jit_s.append(s_jit*to_microsec)

	  t_raw_m.append(m_raw*to_microsec)
	  t_raw_s.append(s_raw*to_microsec)

	  t_shd_m.append(m_shd*to_microsec)
	  t_shd_s.append(s_shd*to_microsec)
	    
	  t_elw_m.append(m_elw*to_microsec)
	  t_elw_s.append(s_elw*to_microsec)
	    
	  t_npy_m.append(m_npy*to_microsec)
	  t_npy_s.append(s_npy*to_microsec)

	  t_cpy_m.append(m_cpy*to_microsec)
	  t_cpy_s.append(s_cpy*to_microsec)
	    
	tab_out['N_data']  = datas

	# JIT kernel GPU(!) mean and st.dev. times in [ms]
	tab_out['t_jit_m'] = t_jit_m
	tab_out['t_jit_s'] = t_jit_s
	# RAW kernel GPU(!) mean and st.dev. times in [ms]
	tab_out['t_raw_m'] = t_raw_m
	tab_out['t_raw_s'] = t_raw_s
	# RAW kernel GPU(!), shared memory, mean and st.dev times in [ms]
	tab_out['t_shd_m'] = t_shd_m
	tab_out['t_shd_s'] = t_shd_s
	# ElementWise kernel GPU(!) mean and st.dev. times in [ms]
	tab_out['t_elw_m'] = t_elw_m
	tab_out['t_elw_s'] = t_elw_s
	# NumPy CPU(!) mean and st.dev. times in [ms]
	tab_out['t_npy_m'] = t_npy_m
	tab_out['t_npy_s'] = t_npy_s
	# CuPy GPU(!) mean and st.dev. times in [ms]
	tab_out['t_cpy_m'] = t_cpy_m
	tab_out['t_cpy_s'] = t_cpy_s

	tab_out.write( fname, format='ascii.fixed_width', formats=fmts, bookend=False, delimiter=None, overwrite=True )

###################################################################################################################
# Test 2. time_gpu VS N_bins
# Check the execution times as a function of number of bins
def test_time_vs_nbins( gpu='T4', verbose=False )
	tab_out = Table()
	fname   = 'tab_Nbins_vs_time_GPU={}.txt'.format( gpu )
	fmts    = {'N_bins' :    '%d',
	           't_jit_m':    '%.6f',
	           't_jit_s':    '%.6f',
	           't_raw_m':    '%.6f',
	           't_raw_s':    '%.6f',
	           't_shd_m':    '%.6f',
	           't_shd_s':    '%.6f',
	           't_elw_m':    '%.6f',
	           't_elw_s':    '%.6f'}

	n_data    = pow(10,5)
	n_threads = 1024
	max_x  = cp.float32(1.0)
	min_x  = cp.float32(0.0)
	cp.random.seed(42)
	d_data = cp.random.rand(n_data).astype(cp.float32)

	n_bins_all = [2,4,8,10,16,32,64,128,256]
	t_jit_m, t_raw_m, t_elw_m, t_shd_m = [],[],[],[]
	t_jit_s, t_raw_s, t_elw_s, t_shd_s = [],[],[],[]

	for n_bins in n_bins_all:
	  exe_gpu_jit = benchmark( histogram_jit,     ((1,1,1), (n_threads,1,1), (d_data, d_bins_data_jit, n_bins, min_x, max_x, d_data.size )), n_repeat=5000, n_warmup=100 )
	  exe_gpu_raw = benchmark( histogram_raw,     ((1,1,1), (n_threads,1,1), (d_data, d_bins_data_raw, n_bins, min_x, max_x, d_data.size )), n_repeat=5000, n_warmup=100 )
	  exe_gpu_shd = benchmark( histogram_raw_shd, ((1,1,1), (n_threads,1,1), (d_data, d_bins_data_shd, n_bins, min_x, max_x, d_data.size )), kwargs={'shared_mem':n_bins * cp.dtype(cp.int32).itemsize}, n_repeat=5000, n_warmup=100 )
	  exe_gpu_elw = benchmark( histogram_elw, args=((d_data, n_bins, min_x, max_x, d_bins_data_elw)), kwargs={'block_size':n_threads},       n_repeat=5000, n_warmup=100 )
	  
	  m_jit, s_jit = np.average(exe_gpu_jit.gpu_times), np.std(exe_gpu_jit.gpu_times)
	  m_raw, s_raw = np.average(exe_gpu_raw.gpu_times), np.std(exe_gpu_raw.gpu_times)
	  m_elw, s_elw = np.average(exe_gpu_elw.gpu_times), np.std(exe_gpu_elw.gpu_times)
	  m_shd, s_shd = np.average(exe_gpu_shd.gpu_times), np.std(exe_gpu_shd.gpu_times)

	  if verbose == True:
		  print ('N_bins = {:d}'.format(n_bins))
		  print ('Jit version:        t={:.6f}+/-{:.6f} us'.format(m_jit*to_microsec, s_jit*to_microsec))
		  print ('Raw version:        t={:.6f}+/-{:.6f} us'.format(m_raw*to_microsec, s_raw*to_microsec))
		  print ('ELW version:        t={:.6f}+/-{:.6f} us'.format(m_elw*to_microsec, s_elw*to_microsec))
		  print ('RAW shared version: t={:.6f}+/-{:.6f} us'.format(m_shd*to_microsec, s_shd*to_microsec))

	  d_bins_data_jit = cp.zeros(n_bins, dtype=cp.int32)
	  d_bins_data_raw = cp.zeros(n_bins, dtype=cp.int32)
	  d_bins_data_elw = cp.zeros(n_bins, dtype=cp.int32)
	  d_bins_data_shd = cp.zeros(n_bins, dtype=cp.int32)
	 
	  t_jit_m.append(m_jit*to_microsec)
	  t_jit_s.append(s_jit*to_microsec)

	  t_raw_m.append(m_raw*to_microsec)
	  t_raw_s.append(s_raw*to_microsec)

	  t_elw_m.append(m_elw*to_microsec)
	  t_elw_s.append(s_elw*to_microsec)

	  t_shd_m.append(m_shd*to_microsec)
	  t_shd_s.append(s_shd*to_microsec)

	tab_out['N_bins']  = n_bins_all
	tab_out['t_jit_m'] = t_jit_m
	tab_out['t_jit_s'] = t_jit_s

	tab_out['t_raw_m'] = t_raw_m
	tab_out['t_raw_s'] = t_raw_s

	tab_out['t_elw_m'] = t_elw_m
	tab_out['t_elw_s'] = t_elw_s

	tab_out['t_shd_m'] = t_shd_m
	tab_out['t_shd_s'] = t_shd_s

	tab_out.write( fname, format='ascii.fixed_width', formats=fmts, bookend=False, delimiter=None, overwrite=True )

###################################################################################################################
# Test 2. time_gpu VS N_threads
# Check the execution times as a function of number of threads

def test_time_vs_nthreads( gpu='T4', verbose=False ):
	tab_out = Table()
	fname   = 'tab_Nthreads_vs_time_GPU={}.txt'.format( gpu )
	fmts    = {'N_threads':  '%d',
	           't_jit_m':    '%.6f',
	           't_jit_s':    '%.6f',
	           't_raw_m':    '%.6f',
	           't_raw_s':    '%.6f',
	           't_shd_m':    '%.6f',
	           't_shd_s':    '%.6f',
	           't_elw_m':    '%.6f',
	           't_elw_s':    '%.6f'}


	n_bins = 10
	n_data = pow(10,4)
	max_x  = cp.float32(1.0)
	min_x  = cp.float32(0.0)
	cp.random.seed(42)
	d_data = cp.random.rand(n_data).astype(cp.float32)

	smem   = n_bins * cp.dtype(cp.int32).itemsize

	n_threads_all = [16,32,64,128,256,512,1024]
	t_jit_m, t_raw_m, t_elw_m, t_shd_m = [],[],[],[]
	t_jit_s, t_raw_s, t_elw_s, t_shd_s = [],[],[],[]

	print ('Compare execution time of CuPy kernels for different number of threads')
	print ('N_data = {:d}'.format(n_data))

	for n_threads in n_threads_all:
	  exe_gpu_jit = benchmark( histogram_jit, ((1,1,1), (n_threads,1,1), (d_data, d_bins_data_jit, n_bins, min_x, max_x, d_data.size )),  n_repeat=5000, n_warmup=100 )
	  exe_gpu_raw = benchmark( histogram_raw, ((1,1,1), (n_threads,1,1), (d_data, d_bins_data_raw, n_bins, min_x, max_x, d_data.size  )), n_repeat=5000, n_warmup=100 )
	  exe_gpu_shd = benchmark( histogram_raw_shd, ((1,1,1), (n_threads,1,1), (d_data, d_bins_data_shd, n_bins, min_x, max_x, d_data.size )), kwargs={'shared_mem':smem}, n_repeat=5000, n_warmup=100 )
	  exe_gpu_elw = benchmark( histogram_elw, args=((d_data, n_bins, min_x, max_x, d_bins_data_elw)), kwargs={'block_size':n_threads},    n_repeat=5000, n_warmup=100 )
	  
	  m_jit, s_jit = np.average(exe_gpu_jit.gpu_times), np.std(exe_gpu_jit.gpu_times)
	  m_raw, s_raw = np.average(exe_gpu_raw.gpu_times), np.std(exe_gpu_raw.gpu_times)
	  m_elw, s_elw = np.average(exe_gpu_elw.gpu_times), np.std(exe_gpu_elw.gpu_times)
	  m_shd, s_shd = np.average(exe_gpu_shd.gpu_times), np.std(exe_gpu_shd.gpu_times)

	  if verbose == True:
		  print ('N_threads = {:d}'.format(n_threads))
		  print ('Jit version:        t={:.6f}+/-{:.6f} us'.format(m_jit*to_microsec, s_jit*to_microsec))
		  print ('Raw version:        t={:.6f}+/-{:.6f} us'.format(m_raw*to_microsec, s_raw*to_microsec))
		  print ('ELW version:        t={:.6f}+/-{:.6f} us'.format(m_elw*to_microsec, s_elw*to_microsec))
		  print ('RAW shared version: t={:.6f}+/-{:.6f} us'.format(m_shd*to_microsec, s_shd*to_microsec))

	  d_bins_data_jit = cp.zeros(n_bins, dtype=cp.int32)
	  d_bins_data_raw = cp.zeros(n_bins, dtype=cp.int32)
	  d_bins_data_elw = cp.zeros(n_bins, dtype=cp.int32)
	  d_bins_data_shd = cp.zeros(n_bins, dtype=cp.int32)
	 
	  t_jit_m.append(m_jit*to_microsec)
	  t_jit_s.append(s_jit*to_microsec)

	  t_raw_m.append(m_raw*to_microsec)
	  t_raw_s.append(s_raw*to_microsec)

	  t_elw_m.append(m_elw*to_microsec)
	  t_elw_s.append(s_elw*to_microsec)
	    
	  t_shd_m.append(m_shd*to_microsec)
	  t_shd_s.append(s_shd*to_microsec)
	    
	tab_out['N_threads']  = n_threads_all
	tab_out['t_jit_m'] = t_jit_m
	tab_out['t_jit_s'] = t_jit_s

	tab_out['t_raw_m'] = t_raw_m
	tab_out['t_raw_s'] = t_raw_s

	tab_out['t_elw_m'] = t_elw_m
	tab_out['t_elw_s'] = t_elw_s

	tab_out['t_shd_m'] = t_shd_m
	tab_out['t_shd_s'] = t_shd_s

	tab_out.write( fname, format='ascii.fixed_width', formats=fmts, bookend=False, delimiter=None, overwrite=True )
###################################################################################################################


def parse_args(options=None):
    # test for command-line arguments here
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', type=int, help="Specify the test to run. Check the run_benchmark_anton.py script for details (lines 260-450).")
    parser.add_argument('-g', '--gpu',  type=str, help="Specify the version of the GPU. The output files will have the corresponding suffix in their names.")
    args = parser.parse_args()
    return args


pargs = parse_args()
if pargs.test == 0:
	test_time_vs_ndata(    gpu='{}'.format(pargs.gpu), verbose=False )

if pargs.test == 1:
	test_time_vs_nbins(    gpu='{}'.format(pargs.gpu), verbose=False )

if pargs.test == 2:
	test_time_vs_nthreads( gpu='{}'.format(pargs.gpu), verbose=False )


