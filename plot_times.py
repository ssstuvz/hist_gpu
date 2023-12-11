import numpy as np 
import matplotlib    as mpl 
import matplotlib.pyplot as pl
import matplotlib.transforms as transforms

from astropy.table import Table 


def func_get_plot_params( xsize, ysize, nrows, ncols, ls=10, way='out', noframe=False, wspace=None, hspace=None, wr=None, pr=None ):
    #########################################################################   
    # THIS FUNCTION CREATES THE SKELETON OF THE FIGURE FOR A GIVEN SET OF   #
    # INPUT PARAMETERS                                                      #
    #                                                                       #
    # INPUT:                                                                #
    #         xsize, ysize   - x,y dimensions of the figure's canvas        #
    #         nrows, ncols   - number of rows and columns that define the   #
    #                          corresponding number of subplots in figure   #
    #         wspace, hspace - horizontal and vertical space between        #
    #                          the subplots                                 #
    #                                                                       #
    # OUTPUT:                                                               #
    #         figure, grid, axlist - figure props and the list of subplots  #
    ######################################################################### 

    pl.clf()
    pl.close()

    figure   = pl.figure( figsize=(xsize,ysize) )
    if wr == None:
        grid = mpl.gridspec.GridSpec(figure=figure, nrows=nrows, ncols=ncols )
    else:
        grid = mpl.gridspec.GridSpec(figure=figure, nrows=nrows, ncols=ncols, width_ratios=wr )

    if noframe == True:
        if nrows == 2:
            nf_list = [1]
        if nrows == 3:
            nf_list = [1,2,5]
        if nrows == 4:
            nf_list = [1,2,3,6,7,11]

    axlist = []
    n = 0
    for i in range(nrows):
        for j in range(ncols):
            frame = True
            if noframe == True:
	            if n in nf_list:
	                frame = False
            axlist.append( pl.subplot( grid[i,j], frameon=frame, projection=pr ) )
            n += 1

    grid.update( wspace=wspace, hspace=hspace )


    j = 0
    for ax in axlist:
        ax.tick_params( 'both', length=4, width=1, which='major', labelsize=ls, direction=way )
        ax.tick_params( 'both', length=2, width=1, which='minor', labelsize=ls, direction=way )
        if noframe == True:
            if j in nf_list:
                # x-axis
                ax.xaxis.set_major_formatter( mplp['for_0'] )
                ax.xaxis.set_minor_locator(   mplp['loc_0'] )
                ax.xaxis.set_major_locator(   mplp['loc_0'] )  
                # y-axis
                ax.yaxis.set_major_formatter( mplp['for_0'] )
                ax.yaxis.set_minor_locator(   mplp['loc_0'] )
                ax.yaxis.set_major_locator(   mplp['loc_0'] ) 
        j += 1 


    if noframe == False:
        return figure, grid, axlist
    else:
        return figure, grid, axlist, nf_list


pl.style.use('plot_default.mplstyle')
figure, grid, axlist = func_get_plot_params( xsize=6, ysize=4, nrows=1, ncols=1 ) 

mplp = {}
mplp['for_0']  = mpl.ticker.NullFormatter()
mplp['loc_0']  = mpl.ticker.NullLocator()

mplp['loc']    = mpl.ticker.MultipleLocator
mplp['for']    = mpl.ticker.FormatStrFormatter




def plot_Ndata():
	fname = 'histogram_plots/n_data_vs_times.pdf'

	fs = 14

	data  = Table.read('histogram_outputs/tab_Ndata_vs_time.txt', format='ascii')


	for j,ax in enumerate(axlist):
	    if j == 0:
	        ax.set_title( r'$ N_{\rm threads}=1024 $', loc='left' )
	        
	        #ax.set_xlim( [0, pow(10,6)])
	        #ax.xaxis.set_minor_locator(   mplp['loc'](500)  )
	        #ax.xaxis.set_major_locator(   mplp['loc']( pow(10,3) )  ) 
	        #ax.xaxis.set_major_formatter( mplp['for']('%d') )
	        ax.set_xlabel( r'$ N_{\rm data} $', fontsize = fs)
	        ax.set_xscale('log')

	        # y-axis properties 
	        ax.set_ylim( [pow(10,1), pow(10,4) ] )
	        #ax.yaxis.set_minor_locator(   mplp['loc'](50) )
	        #ax.yaxis.set_major_locator(   mplp['loc'](250) )
	        #ax.yaxis.set_major_formatter( mplp['for']('%.1f') )  
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

	fs = 14

	data1  = Table.read('histogram_outputs/tab_Nthreads_vs_time_ndata=10**4.txt', format='ascii')
	data2  = Table.read('histogram_outputs/tab_Nthreads_vs_time_ndata=10**5.txt', format='ascii')


	for j,ax in enumerate(axlist):
	    if j == 0:
	        #ax.set_title( r'$(-) N_{\rm data}=10^4~|~(--) N_{\rm data}=10^5$', loc='left' )
	        ax.set_title( r'$(-) N_{\rm data}=10^5~|~N_{\rm bins}=10$', loc='left' )
	        
	        ax.set_xlim( [0, 1024])
	        ax.xaxis.set_minor_locator(   mplp['loc']( 64 )   )
	        ax.xaxis.set_major_locator(   mplp['loc']( 256 )  ) 
	        ax.xaxis.set_major_formatter( mplp['for']('%d')   )
	        ax.set_xlabel( r'$ N_{\rm threads} $', fontsize = fs)

	        # y-axis properties 
	        ax.set_ylim( [ pow(10,1), pow(10,3.5) ] )
	        #ax.yaxis.set_minor_locator(   mplp['loc'](50) )
	        #ax.yaxis.set_major_locator(   mplp['loc'](200) )
	        #ax.yaxis.set_major_formatter( .mplp['for']('%.1f') )  
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

	fs = 14

	data  = Table.read('histogram_outputs/tab_Nbins_vs_time.txt', format='ascii')

	for j,ax in enumerate(axlist):
	    if j == 0:
	        ax.set_title( r'$(-) N_{\rm data}=10^5~|~(--) N_{\rm threads}=1024$', loc='left' )
	        
	        ax.set_xlim( [-5, 260])
	        ax.xaxis.set_minor_locator(   mplp['loc']( 8 )   )
	        ax.xaxis.set_major_locator(   mplp['loc']( 64 )  ) 
	        ax.xaxis.set_major_formatter( mplp['for']('%d')   )
	        ax.set_xlabel( r'$ N_{\rm bins} $', fontsize = fs)

	        # y-axis properties 
	        ax.set_ylim( [ pow(10,1), pow(10,3) ] )
	        #ax.yaxis.set_minor_locator(   mplp['loc'](50) )
	        #ax.yaxis.set_major_locator(   mplp['loc'](200) )
	        #ax.yaxis.set_major_formatter( mplp['for']('%.1f') )  
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










