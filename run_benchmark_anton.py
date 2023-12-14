import os
import sys
import cupy as cp
import argparse
from cupyx.profiler import benchmark

try:
	from astropy.table import Table
except:
	print ('No astropy package is found! Please install: pip install astropy')
	sys.exit()

try:
	from py_boost.gpu.utils import *
except:
	print ('No py_boost package is found! Please install: pip install py_boost')



# This is the copy of Anton's original histogram kernel
histogram_kernel_idx_elw = cp.ElementwiseKernel(
    """
    uint64 i_, uint64 j_, uint64 k_,
    uint64 kk,

    raw uint64 jj,
    raw bool padded_bool_indexer,

    raw float32 target,
    raw T arr,
    raw int32 nodes,

    uint64 hlen,
    uint64 flen,
    uint64 length,
    uint64 feats,
    uint64 nout
    """,
    'raw float32 hist',

    """
    unsigned int feat_4t = arr[i_ * feats + j_];
    int d;
    int j;
    int val;
    int pos;
    float *x_ptr;
    float y = target[i_ * nout + k_];

    for (d = 0; d < 4; d++) {

        pos = (i_ + d) % 4;

        if (padded_bool_indexer[j_ * 4 + pos]) {

            val = (feat_4t >> (8 * pos)) % 256;
            j = jj[j_ * 4 + pos];
            x_ptr = &hist[0] +  kk * hlen + nodes[i_] * flen + j * length + val;
            atomicAdd(x_ptr, y);
        }
    }

    """,

    'histogram_kernel_idx')


# This is Sergey's updated version
histogram_kernel_idx_ser = cp.ElementwiseKernel(
    """
    uint64 i_, uint64 j_, uint64 k_,
    uint64 kk,

    raw uint64 jj,
    raw bool padded_bool_indexer,

    raw float32 target,
    raw T arr,
    raw int32 nodes,

    uint64 hlen,
    uint64 flen,
    uint64 length,
    uint64 feats,
    uint64 nout
    """,
    'raw float32 hist',

    """
    unsigned int feat_4t = arr[i_ * feats + j_];

    unsigned long long j;
    unsigned int val;
    unsigned char pos;
    float y = target[i_ * nout + k_];

    unsigned long long posj = j_ * 4;

    unsigned long long posk = kk * hlen + nodes[i_] * flen;

    for (unsigned char d = 0; d < 4; ++d) {

        pos = (i_ + d) % 4;

        if (padded_bool_indexer[posj + pos]) {

            val = (feat_4t >> (8 * pos)) % 256;
            j = jj[posj + pos];
            atomicAdd(&hist[posk + j * length  + val], y);
        }
    }

    """,

    'histogram_kernel_idx_ser')

def fill_histogram_tmp(res, arr, target, nodes, col_indexer, row_indexer, out_indexer, func='elw'):
    """Fill the histogram res

    Args:
        res: cp.ndarray, histogram of zeros, shape (n_out, n_nodes, n_features, n_bins)
        arr: cp.ndarray, features array, shape (n_data, n_features)
        target: cp.ndarray, values to accumulate, shape (n_data, n_out)
        nodes: cp.ndarray, tree node indices, shape (n_data, )
        col_indexer: cp.ndarray, indices of features to accumulate
        row_indexer: cp.ndarray, indices of rows to accumulate
        out_indexer: cp.ndarray, indices of outputs to accumulate
        func: numeric flag to choose the kernel that will be used in histogram calculations

    Returns:

    """
    # define data split for kernel launch
    nout, nnodes, nfeats, nbins = res.shape

    # padded array of 4 feature tuple
    arr_4t = arr.base.view(dtype=cp.uint32)
    pfeats = arr_4t.shape[1]

    # create 4 feats tuple indexer
    padded_bool_indexer = cp.zeros((arr.base.shape[1],), dtype=cp.bool_)
    padded_col_indexer = cp.zeros((arr.base.shape[1],), dtype=cp.uint64)
    tuple_indexer = cp.zeros((arr_4t.shape[1],), dtype=cp.bool_)

    feature_grouper_kernel(col_indexer, padded_bool_indexer, tuple_indexer, padded_col_indexer)
    tuple_indexer = cp.arange(arr_4t.shape[1], dtype=cp.uint64)[tuple_indexer]

    fb = nfeats * nbins
    nfb = nnodes * fb

    magic_constant = 2 ** 19  # optimal value for my V100

    # split features
    nsplits = math.ceil(nfb / magic_constant)
    # first split by feats
    feats_batch = math.ceil(pfeats / nsplits)
    # split by features
    if feats_batch == nfeats:
        out_batch = magic_constant // nfb
    else:
        out_batch = 1

    ri = row_indexer[:, cp.newaxis, cp.newaxis]
    ti = tuple_indexer[cp.newaxis, :, cp.newaxis]
    oi = out_indexer[cp.newaxis, cp.newaxis, :]

    oii = cp.arange(oi.shape[2], dtype=cp.uint64)[cp.newaxis, cp.newaxis, :]

    for j in range(0, pfeats, feats_batch):
        ti_ = ti[:, j: j + feats_batch]

        for k in range(0, nout, out_batch):
            oi_ = oi[..., k: k + out_batch]
            oii_ = oii[..., k: k + out_batch]

            if func == 'elw':
                # Use original Anton's solution
                histogram_kernel_idx_elw(ri, ti_, oi_,
                                     oii_,
                                     padded_col_indexer,
                                     padded_bool_indexer,
                                     target,
                                     arr_4t,
                                     nodes,
                                     nfb, fb, nbins, arr_4t.shape[1], nout,
                                     res)
            if func == 'ser':
            	# Use updated Sergey's version
                histogram_kernel_idx_ser(ri, ti_, oi_,
                                         oii_,
                                         padded_col_indexer,
                                         padded_bool_indexer,
                                         target,
                                         arr_4t,
                                         nodes,
                                         nfb, fb, nbins, arr_4t.shape[1], nout,
                                         res )

def sample_idx(n, sample):
    # THIST FUNCTION GENERATES IDS USED 
    # IN THE HISTOGRAM CALCULATIONS
    
    idx = cp.arange(n, dtype=cp.uint64)
    sl = cp.random.rand(n) < sample
    
    return cp.ascontiguousarray(idx[sl])

def generate_input( n_rows, n_cols, n_out, max_bin, nnodes, 
                    colsample=0.8, subsample=0.8, outsample=1.0, verbose=False, seed=42):
    # THIS FUNCTION GENERATES ALL INPUT
    # ARRAYS, REQUIRED BY THE HISTOGRAM 
    # FUNCTION IN PY-BOOST
    # Input: 
    # n_rows   - number of rows in the input array
    # n_cols   - number of cols in the input array
    # n_out    - number of ???? in the output array
    # max_bins - number of histogram bins (can't be >256 really)
    # nnodes   - ????
    
    np.random.seed(seed)
    features_cpu = np.random.randint(0, max_bin, size=(n_rows, n_cols)).astype(np.uint8)
    features_gpu = pad_and_move(features_cpu)
    cp.random.seed(seed)
    targets_gpu  = cp.random.rand(n_rows, n_out).astype(np.float32)
    cp.random.seed(seed)
    nodes_gpu    = cp.random.randint(0, nnodes, size=(n_rows, )).astype(np.int32)
    cp.random.seed(seed)
    
    if verbose == True:
        print('Initial CPU features shape: {}'.format(features_cpu.shape))
        print('Padded  GPU features shape: {}'.format(features_gpu.shape))
        print('Nodes   GPU vector   shape: {}'.format(nodes_gpu.shape   ))
        print('Targets GPU array    shape: {}'.format(targets_gpu.shape ))
    
    row_indexer = sample_idx(n_rows, subsample)
    col_indexer = sample_idx(n_cols, colsample)
    out_indexer = sample_idx(n_out, outsample)
    
    if verbose == True:
        print('Sampled rows shape:    {}'.format(row_indexer.shape))
        print('Sampled columns shape: {}'.format(col_indexer.shape))
        print('Sampled output shape:  {}'.format(out_indexer.shape))
    
    nout   = out_indexer.shape[0]
    nfeats = col_indexer.shape[0]
    
    # Anton's function takes the following input arguments + the empty array to 
    # store the resulting histogram bins (comes in the first position)
    # input: res, X, Y, nodes, col_indexer, row_indexer, out_indexer
    
    res    = cp.zeros((nout, nnodes, nfeats, max_bin), dtype=cp.float32)
    params = (res, features_gpu, targets_gpu, nodes_gpu, col_indexer, row_indexer, out_indexer)
    
    if verbose == True:
        print ('Sum of the resulting histogram must be {}'.format(nfeats * targets_gpu[row_indexer].sum()))
    return params


# Now run benchmark tests. 

###################################################################################################################
# Test 1. time_gpu VS N_data

# Check the performance of the algorithm for different values of n_rows (time is in microseconds)
def test_time_vs_nrows( gpu='T4', verbose=False ):
	tab_out = Table()
	fname   = 'tab_Anton_Nrows_vs_time_GPU={}.txt'.format( gpu )
	fmts    = {'N_rows'       :  '%.2f',
	           'tau_mean_elw' :  '%.6f',
	           'tau_stdev_elw':  '%.6f',
	           'tau_mean_ser' :  '%.6f',
	           'tau_stdev_ser':  '%.6f'}

	exponent    = np.arange(2,6.1,0.25)    
	n_rows      = np.array([pow(10,x) for x in exponent],dtype=np.int32)
	n_threads   = 1024
	to_microsec = pow(10,6)

	ns, means_elw, stdevs_elw, means_ser, stdevs_ser = [],[],[],[],[]
	for n in n_rows:
	    input_params = generate_input(n_rows=n,n_cols=99,n_out=10,max_bin=256,nnodes=32,verbose=False)
	    tau_elw = benchmark( fill_histogram_tmp, (*input_params, 'elw'), n_repeat=1000, n_warmup=10 )
	    tau_ser = benchmark( fill_histogram_tmp, (*input_params, 'ser'), n_repeat=1000, n_warmup=10 )
	    
	    mean_elw, stdev_elw = np.average(tau_elw.gpu_times), np.std(tau_elw.gpu_times)
	    mean_ser, stdev_ser = np.average(tau_ser.gpu_times), np.std(tau_ser.gpu_times)
	    
	    if verbose == True:
		    print ('N_rows = 10^{:.2f}'.format(np.log10(n)))
		    print ('time_elw = {:.6f}+/-{:.6f} microsec'.format(mean_elw*to_microsec, stdev_elw*to_microsec))
		    print ('time_ser = {:.6f}+/-{:.6f} microsec'.format(mean_ser*to_microsec, stdev_ser*to_microsec))

	    ns.append( np.log10(n) )
	    means_elw.append(  mean_elw  * to_microsec)
	    stdevs_elw.append( stdev_elw * to_microsec)

	    means_ser.append(  mean_ser  * to_microsec)
	    stdevs_ser.append( stdev_ser * to_microsec)
	    
	tab_out['N_rows']        = ns
	tab_out['tau_mean_elw']  = means_elw
	tab_out['tau_stdev_elw'] = stdevs_elw

	tab_out['tau_mean_ser']  = means_ser
	tab_out['tau_stdev_ser'] = stdevs_ser

	tab_out.write( fname, format='ascii.fixed_width', formats=fmts, bookend=False, delimiter=None, overwrite=True )

###################################################################################################################
# Test 2. time_gpu VS N_cols
# Check the performance of the algorithm for different values of N_cols (time is in microseconds)

def test_time_vs_ncols( gpu='T4', verbose=False ):
	tab_out = Table()
	fname   = 'tab_Anton_Ncols_vs_time_GPU={}.txt'.format( gpu )
	fmts    = {'N_cols'       :  '%d',
	           'tau_mean_elw' :  '%.6f',
	           'tau_stdev_elw':  '%.6f',
	           'tau_mean_ser' :  '%.6f',
	           'tau_stdev_ser':  '%.6f'}
	  
	n_cols      = np.arange(50, 1051, 100)
	n_threads   = 1024
	to_microsec = pow(10,6)

	ns, means_elw, stdevs_elw, means_ser, stdevs_ser = [],[],[],[],[]
	for n in n_cols:
	    input_params = generate_input(n_rows=pow(10,6),n_cols=n,n_out=10,max_bin=256,nnodes=32,verbose=False)
	    tau_elw = benchmark( fill_histogram_tmp, (*input_params, 'elw'), n_repeat=1000, n_warmup=10 )
	    tau_ser = benchmark( fill_histogram_tmp, (*input_params, 'ser'), n_repeat=1000, n_warmup=10 )
	    
	    mean_elw, stdev_elw = np.average(tau_elw.gpu_times), np.std(tau_elw.gpu_times)
	    mean_ser, stdev_ser = np.average(tau_ser.gpu_times), np.std(tau_ser.gpu_times)
	    
	    if verbose == True:
		    print ('N_cols = {}'.format(n))
		    print ('time_elw = {:.6f}+/-{:.6f} microsec'.format(mean_elw*to_microsec, stdev_elw*to_microsec))
		    print ('time_ser = {:.6f}+/-{:.6f} microsec'.format(mean_ser*to_microsec, stdev_ser*to_microsec))

	    ns.append( n )
	    means_elw.append(  mean_elw  * to_microsec)
	    stdevs_elw.append( stdev_elw * to_microsec)

	    means_ser.append(  mean_ser  * to_microsec)
	    stdevs_ser.append( stdev_ser * to_microsec)

	    
	tab_out['N_cols']        = ns
	tab_out['tau_mean_elw']  = means_elw
	tab_out['tau_stdev_elw'] = stdevs_elw

	tab_out['tau_mean_ser']  = means_ser
	tab_out['tau_stdev_ser'] = stdevs_ser

	tab_out.write( fname, format='ascii.fixed_width', formats=fmts, bookend=False, delimiter=None, overwrite=True )
###################################################################################################################
# Test 3. time_gpu VS N_bins
# Check the performance of the algorithm for different values of N_bins (time is in microseconds)

def test_time_vs_nbins( gpu='T4', verbose=False ):
	tab_out = Table()
	fname   = 'tab_Anton_Nmaxbins_vs_time_GPU={}.txt'.format( gpu )
	fmts    = {'N_mbins'      :  '%d',
	           'tau_mean_elw' :  '%.6f',
	           'tau_stdev_elw':  '%.6f',
	           'tau_mean_ser' :  '%.6f',
	           'tau_stdev_ser':  '%.6f'}

	n_bins      = np.arange(8,260,8) 
	n_threads   = 1024
	to_microsec = pow(10,6)

	ns, means_elw, stdevs_elw, means_ser, stdevs_ser = [],[],[],[],[]
	for n in n_bins:
	    input_params = generate_input(n_rows=pow(10,6),n_cols=99,n_out=10,max_bin=n,nnodes=32,verbose=False)
	    tau_elw = benchmark( fill_histogram_tmp, (*input_params, 'elw'), n_repeat=1000, n_warmup=10 )
	    tau_ser = benchmark( fill_histogram_tmp, (*input_params, 'ser'), n_repeat=1000, n_warmup=10 )
	    
	    mean_elw, stdev_elw = np.average(tau_elw.gpu_times), np.std(tau_elw.gpu_times)
	    mean_ser, stdev_ser = np.average(tau_ser.gpu_times), np.std(tau_ser.gpu_times)
	    
	    if verbose == True:
		    print ('N_bins = {}'.format(n))
		    print ('time_elw = {:.6f}+/-{:.6f} microsec'.format(mean_elw*to_microsec, stdev_elw*to_microsec))
		    print ('time_ser = {:.6f}+/-{:.6f} microsec'.format(mean_ser*to_microsec, stdev_ser*to_microsec))

	    ns.append( n )
	    means_elw.append(  mean_elw  * to_microsec)
	    stdevs_elw.append( stdev_elw * to_microsec)

	    means_ser.append(  mean_ser  * to_microsec)
	    stdevs_ser.append( stdev_ser * to_microsec)
	    
	tab_out['N_mbins']       = ns
	tab_out['tau_mean_elw']  = means_elw
	tab_out['tau_stdev_elw'] = stdevs_elw

	tab_out['tau_mean_ser']  = means_ser
	tab_out['tau_stdev_ser'] = stdevs_ser

	tab_out.write( fname, format='ascii.fixed_width', formats=fmts, bookend=False, delimiter=None, overwrite=True )
###################################################################################################################
# Test 4. time_gpu VS N_nodes
# Check the performance of the algorithm for different values of n_nodes (time is in microseconds)

def test_time_vs_nnodes( gpu='T4', verbose=False ):
	tab_out = Table()

	fname   = 'tab_Anton_Nnodes_vs_time_GPU={}.txt'.format( gpu )
	fmts    = {'N_nodes'  :  '%d',
	           'tau_mean_elw' :  '%.6f',
	           'tau_stdev_elw':  '%.6f',
	           'tau_mean_ser' :  '%.6f',
	           'tau_stdev_ser':  '%.6f'}
	  
	n_nodes     = np.arange(8,260,8) 
	n_threads   = 1024
	to_microsec = pow(10,6)

	ns, means_elw, stdevs_elw, means_ser, stdevs_ser = [],[],[],[],[]
	for n in n_nodes:
	    input_params = generate_input(n_rows=pow(10,6),n_cols=99,n_out=10,max_bin=256,nnodes=n,verbose=False)
	    tau_elw = benchmark( fill_histogram_tmp, (*input_params, 'elw'), n_repeat=1000, n_warmup=10 )
	    tau_ser = benchmark( fill_histogram_tmp, (*input_params, 'ser'), n_repeat=1000, n_warmup=10 )
	    
	    mean_elw, stdev_elw = np.average(tau_elw.gpu_times), np.std(tau_elw.gpu_times)
	    mean_ser, stdev_ser = np.average(tau_ser.gpu_times), np.std(tau_ser.gpu_times)
	    
	    if verbose == True:
		    print ('N_nodes = {}'.format(n))
		    print ('time_elw = {:.6f}+/-{:.6f} microsec'.format(mean_elw*to_microsec, stdev_elw*to_microsec))
		    print ('time_ser = {:.6f}+/-{:.6f} microsec'.format(mean_ser*to_microsec, stdev_ser*to_microsec))

	    ns.append( n )
	    means_elw.append(  mean_elw  * to_microsec)
	    stdevs_elw.append( stdev_elw * to_microsec)

	    means_ser.append(  mean_ser  * to_microsec)
	    stdevs_ser.append( stdev_ser * to_microsec)
	    
	tab_out['N_nodes']       = ns
	tab_out['tau_mean_elw']  = means_elw
	tab_out['tau_stdev_elw'] = stdevs_elw

	tab_out['tau_mean_ser']  = means_ser
	tab_out['tau_stdev_ser'] = stdevs_ser

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
	test_time_vs_nrows(  gpu='{}'.format(pargs.gpu), verbose=False )

if pargs.test == 1:
	test_time_vs_ncols(  gpu='{}'.format(pargs.gpu), verbose=False )

if pargs.test == 2:
	test_time_vs_nbins(  gpu='{}'.format(pargs.gpu), verbose=False )

if pargs.test == 3:
	test_time_vs_nnodes( gpu='{}'.format(pargs.gpu), verbose=False )








