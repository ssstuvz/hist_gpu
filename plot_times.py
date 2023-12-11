import numpy as np 
import matplotlib    as mpl 
import matplotlib.pyplot as pl
import matplotlib.transforms as transforms

from astropy.table import Table 
from cobra.plot    import setup
from   cobra.utils   import dataset_io

def plot_Ndata():
	fname = 'histogram_plots/n_data_vs_times.pdf'
	pl.style.use('{}/plot/cobra_default.mplstyle'.format(dataset_io.gpaths['cobra_root']))
	figure, grid, axlist = setup.func_get_plot_params( xsize=6, ysize=4, nrows=1, ncols=1 ) 

	fs = 14

	data  = Table.read('histogram_outputs/tab_Ndata_vs_time.txt', format='ascii')


	for j,ax in enumerate(axlist):
	    if j == 0:
	        ax.set_title( r'$ N_{\rm threads}=1024 $', loc='left' )
	        
	        #ax.set_xlim( [0, pow(10,6)])
	        #ax.xaxis.set_minor_locator(   setup.mplp['loc'](500)  )
	        #ax.xaxis.set_major_locator(   setup.mplp['loc']( pow(10,3) )  ) 
	        #ax.xaxis.set_major_formatter( setup.mplp['for']('%d') )
	        ax.set_xlabel( r'$ N_{\rm data} $', fontsize = fs)
	        ax.set_xscale('log')

	        # y-axis properties 
	        ax.set_ylim( [pow(10,1), pow(10,4) ] )
	        #ax.yaxis.set_minor_locator(   setup.mplp['loc'](50) )
	        #ax.yaxis.set_major_locator(   setup.mplp['loc'](250) )
	        #ax.yaxis.set_major_formatter( setup.mplp['for']('%.1f') )  
	        ax.set_ylabel( r'$ \tau_{\rm GPU}~\left[ {\rm ms} \right] $', fontsize = fs)
	        ax.set_yscale('log')

	        # Plot all

	        ax.errorbar( data['N_data'], data['t_jit_m'], yerr=data['t_jit_s'], color='black',      marker='o', ms=4, capsize=0, elinewidth=1, label=r'${\rm JIT\ kernel}$'  )
	        ax.errorbar( data['N_data'], data['t_raw_m'], yerr=data['t_raw_s'], color='OrangeRed',  marker='o', ms=4, capsize=0, elinewidth=1, label=r'${\rm RAW\ kernel}$'  )
	        ax.errorbar( data['N_data'], data['t_shd_m'], yerr=data['t_shd_s'], color='brown',      marker='o', ms=4, capsize=0, elinewidth=1, label=r'${\rm RAW\ kernel,\ shared\ memory}$', ls='--'  )	    
	        ax.errorbar( data['N_data'], data['t_elw_m'], yerr=data['t_elw_s'], color='DodgerBlue', marker='o', ms=4, capsize=0, elinewidth=1, label=r'${\rm ELW\ kernel}$'  )
	        ax.errorbar( data['N_data'], data['t_npy_m'], yerr=data['t_npy_s'], color='Magenta',    marker='o', ms=4, capsize=0, elinewidth=1, label=r'${\rm NumPy\ hist}$'  )
	        ax.errorbar( data['N_data'], data['t_cpy_m'], yerr=data['t_cpy_s'], color='green',      marker='o', ms=4, capsize=0, elinewidth=1, label=r'${\rm CuPy\  hist}$'  )
	        
	        ax.legend(loc=0, prop={'size':10}, ncol=1, frameon=False, numpoints=1)

	pl.savefig( fname )
	pl.clf()
	pl.close()

def plot_Nthreads():
	fname = 'histogram_plots/n_threads_vs_times.pdf'
	pl.style.use('{}/plot/cobra_default.mplstyle'.format(dataset_io.gpaths['cobra_root']))
	figure, grid, axlist = setup.func_get_plot_params( xsize=6, ysize=4, nrows=1, ncols=1 ) 

	fs = 14

	data1  = Table.read('histogram_outputs/tab_Nthreads_vs_time_ndata=10**4.txt', format='ascii')
	data2  = Table.read('histogram_outputs/tab_Nthreads_vs_time_ndata=10**5.txt', format='ascii')


	for j,ax in enumerate(axlist):
	    if j == 0:
	        #ax.set_title( r'$(-) N_{\rm data}=10^4~|~(--) N_{\rm data}=10^5$', loc='left' )
	        ax.set_title( r'$(-) N_{\rm data}=10^5~|~N_{\rm bins}=10$', loc='left' )
	        
	        ax.set_xlim( [0, 1024])
	        ax.xaxis.set_minor_locator(   setup.mplp['loc']( 64 )   )
	        ax.xaxis.set_major_locator(   setup.mplp['loc']( 256 )  ) 
	        ax.xaxis.set_major_formatter( setup.mplp['for']('%d')   )
	        ax.set_xlabel( r'$ N_{\rm threads} $', fontsize = fs)

	        # y-axis properties 
	        ax.set_ylim( [ pow(10,1), pow(10,3.5) ] )
	        #ax.yaxis.set_minor_locator(   setup.mplp['loc'](50) )
	        #ax.yaxis.set_major_locator(   setup.mplp['loc'](200) )
	        #ax.yaxis.set_major_formatter( setup.mplp['for']('%.1f') )  
	        ax.set_ylabel( r'$ \tau_{\rm GPU}~\left[ {\rm ms} \right] $', fontsize = fs)
	        ax.set_yscale('log')

	        # Plot all

	        #ax.errorbar( data1['N_threads'], data1['t_jit_m'], yerr=data1['t_jit_s'], color='black',      marker='o', ms=4, capsize=0, elinewidth=1, label=r'${\rm JIT\ kernel}$'  )
	        #ax.errorbar( data1['N_threads'], data1['t_raw_m'], yerr=data1['t_raw_s'], color='OrangeRed',  marker='o', ms=4, capsize=0, elinewidth=1, label=r'${\rm RAW\ kernel}$'  )
	        #ax.errorbar( data1['N_threads'], data1['t_elw_m'], yerr=data1['t_elw_s'], color='DodgerBlue', marker='o', ms=4, capsize=0, elinewidth=1, label=r'${\rm ELW\ kernel}$'  )

	        ax.errorbar( data2['N_threads'], data2['t_jit_m'], yerr=data2['t_jit_s'], color='black',      marker='o', ms=4, capsize=0, elinewidth=1, ls='-', label=r'${\rm JIT\ kernel}$' )
	        ax.errorbar( data2['N_threads'], data2['t_raw_m'], yerr=data2['t_raw_s'], color='OrangeRed',  marker='o', ms=4, capsize=0, elinewidth=1, ls='-', label=r'${\rm RAW\ kernel}$' )
	        ax.errorbar( data2['N_threads'], data2['t_elw_m'], yerr=data2['t_elw_s'], color='DodgerBlue', marker='o', ms=4, capsize=0, elinewidth=1, ls='-', label=r'${\rm ELW\ kernel}$' )
	        ax.errorbar( data2['N_threads'], data2['t_shd_m'], yerr=data2['t_shd_s'], color='brown',      marker='o', ms=4, capsize=0, elinewidth=1, ls='-', label=r'${\rm RAW\ shared\ kernel}$' )

	        ax.legend(loc=0, prop={'size':10}, ncol=1, frameon=False, numpoints=1)

	pl.savefig( fname )
	pl.clf()
	pl.close()
	return 


def plot_Nbins():
	fname = 'histogram_plots/n_bins_vs_times.pdf'
	pl.style.use('{}/plot/cobra_default.mplstyle'.format(dataset_io.gpaths['cobra_root']))
	figure, grid, axlist = setup.func_get_plot_params( xsize=6, ysize=4, nrows=1, ncols=1 ) 

	fs = 14

	data  = Table.read('histogram_outputs/tab_Nbins_vs_time.txt', format='ascii')

	for j,ax in enumerate(axlist):
	    if j == 0:
	        ax.set_title( r'$(-) N_{\rm data}=10^5~|~(--) N_{\rm threads}=1024$', loc='left' )
	        
	        ax.set_xlim( [-5, 260])
	        ax.xaxis.set_minor_locator(   setup.mplp['loc']( 8 )   )
	        ax.xaxis.set_major_locator(   setup.mplp['loc']( 64 )  ) 
	        ax.xaxis.set_major_formatter( setup.mplp['for']('%d')   )
	        ax.set_xlabel( r'$ N_{\rm bins} $', fontsize = fs)

	        # y-axis properties 
	        ax.set_ylim( [ pow(10,1), pow(10,3) ] )
	        #ax.yaxis.set_minor_locator(   setup.mplp['loc'](50) )
	        #ax.yaxis.set_major_locator(   setup.mplp['loc'](200) )
	        #ax.yaxis.set_major_formatter( setup.mplp['for']('%.1f') )  
	        ax.set_ylabel( r'$ \tau_{\rm GPU}~\left[ {\rm ms} \right] $', fontsize = fs)
	        ax.set_yscale('log')

	        # Plot all

	        ax.errorbar( data['N_bins'], data['t_jit_m'], yerr=data['t_jit_s'], color='black',      marker='o', ms=4, capsize=0, elinewidth=1, label=r'${\rm JIT\ kernel}$'  )
	        ax.errorbar( data['N_bins'], data['t_raw_m'], yerr=data['t_raw_s'], color='OrangeRed',  marker='o', ms=4, capsize=0, elinewidth=1, label=r'${\rm RAW\ kernel}$'  )
	        ax.errorbar( data['N_bins'], data['t_elw_m'], yerr=data['t_elw_s'], color='DodgerBlue', marker='o', ms=4, capsize=0, elinewidth=1, label=r'${\rm ELW\ kernel}$'  )
	        ax.errorbar( data['N_bins'], data['t_shd_m'], yerr=data['t_shd_s'], color='brown',      marker='o', ms=4, capsize=0, elinewidth=1, label=r'${\rm RAW\ shared\ kernel}$'  )

	        ax.legend(loc=0, prop={'size':10}, ncol=1, frameon=False, numpoints=1)

	pl.savefig( fname )
	pl.clf()
	pl.close()
	return 










