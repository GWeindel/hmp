'''

'''
import scipy.stats as stats
import scipy.signal as ssignal
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from itertools import cycle
import warnings
default_colors =  ['cornflowerblue','indianred','orange','darkblue','darkgreen','gold']


def plot_topo_timecourse(channels, estimated, channel_position, init, time_step=1, ydim=None,
                figsize=None, dpi=100, magnify=1, times_to_display=None, cmap='Spectral_r',
                ylabels=[], xlabel = None, max_time = None, vmin=None, vmax=None, title=False, ax=None, 
                sensors=False, skip_channels_computation=False, contours=6, event_lines='tab:orange',
                colorbar=True, topo_size_scaling=False, as_time=False,
                        linecolors='black'):
    '''
    Plotting the event topologies at the average time of the onset of the next stage.
    
    Parameters
    ----------
    channels : ndarray | xr.Dataarray 
        a 2D or 3D matrix of channel activity with channels and event as dimension (+ eventually a varying dimension) OR
        the original EEG data in HMP format
    estimated : ndarray | hmp object
        a 1D or 2D matrix of times with event as dimension OR directly the results from a fitted hmp (either from fit_single or fit_single_conds)
    channel_position : ndarray
        Either a 2D array with dimension channel and [x,y] storing channel location in meters or an info object from
        the mne package containning digit. points for channel location
    init : float
        initialized HMP object
    time_step : float
        What unit to multiply all the times with, if you want to go on the second or millisecond scale you can provide 
        1/sf or 1000/sf where sf is the sampling frequency of the data
    ydim: str
        name for the extra dimensions (e.g. iteration)
    figsize : list | tuple | ndarray
        Length and heigth of the matplotlib plot
    dpi : float
        DPI of the  matplotlib plot
    magnify : float
        How much should the events be enlarged, useful to zoom on topologies, providing any other value than 1 will 
        however change the displayed size of the event
    times_to_display : ndarray
        Times to display (e.g. Reaction time or any other relevant time) in the time unit of the fitted data
    cmap : str
        Colormap of matplotlib
    xlabel : str
        label of x-axis, default = None, which give "Time (samples)" or "Time" in case time_step != 1
    ylabels : dict
        dictonary with {label_name : label_values}. E.g. {'Condition': ['Speed','Accuracy']}
    max_time : float
        limit of the x (time) axe
    vmin : float
        Colormap limits to use (see https://mne.tools/dev/generated/mne.viz.plot_topomap.html). If not explicitly
        set, uses min across all topos while keeping colormap symmetric.
    vmax : float
        Colormap limits to use (see https://mne.tools/dev/generated/mne.viz.plot_topomap.html). If not explicitly
        set, uses max across all topos while keeping colormap symmetric.
    title : str
        title of the plot
    ax : matplotlib.pyplot.ax
        Matplotlib object on which to draw the plot, can be useful if you want to control specific aspects of the plots
        outside of this function
    sensors : bool
        Whether to plot the sensors on the topologies
    skip_channel_contribution: bool
        if True assumes that the provided channel argument is already topologies for each channel
    contours : int / array_like
        The number of contour lines to draw (see https://mne.tools/dev/generated/mne.viz.plot_topomap.html)
    event_lines : bool / color
        Whether to plot lines and shading to indicate the moment of the event. If True uses tab:orange, if
        set as color, uses the color
    colorbar : bool
        Whether a colorbar is plotted.
    topo_size_scaling : bool
        Whether to scale the size of the topologies with the event size. If True, size of topologies depends
        on total plotted time interval, if False it is only dependent on magnify.
    as_time : bool
        if true, plot time (ms) instead of samples (time_step takes precedence). Ignored if times are provided as array.

    Returns
    -------
    ax : matplotlib.pyplot.ax
        if ax was specified otherwise returns the plot
    '''

    from mne.viz import plot_brain_colorbar, plot_topomap
    from mne import Info
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if as_time and time_step != 1:
        print('time_step takes precedence over as_time')
        as_time = False
    
    cond_plot = False

    #if condition estimates, prep conds
    if isinstance(estimated, xr.Dataset) and 'condition' in estimated.dims:
        cond_plot = True
        conds = estimated['cond'].values
        n_cond = estimated.parameters.shape[0]

        #make times_to_display in list with lines per condition
        if times_to_display is None:
            times_to_display = [np.mean(init.durations[conds==c]) for c in range(n_cond)]
            times_to_display.reverse()
        elif len(times_to_display) == 1: #only one set of times
            if isinstance(times_to_display, np.ndarray):
                times_to_display = [times_to_display] * n_cond
            elif isinstance(times_to_display, list):
                times_to_display = times_to_display * n_cond
        elif len(times_to_display) > 1 and len(times_to_display) != n_cond:
            print('times_to_display should either be a list of length n_cond or an ndarray which will be repeated across conditions')
            times_to_display = [np.mean(init.durations[conds==c]) for c in range(n_cond)].reverse() # default to RTs
            times_to_display.reverse()

        #set ylabels to conditions
        if ylabels == []:
            ylabels = estimated.clabels

    return_ax = True
    #if not times specified, plot average RT
    if times_to_display is None:
        times_to_display = init.mean_d
    
    #set xlabel depending on time_step
    if xlabel is None:
        if time_step == 1 and as_time == False:
            xlabel = 'Time (in samples)'
        elif as_time:
            xlabel = 'Time (ms)'
        else:
            xlabel = 'Time'

    #set color of event_lines 
    if event_lines == True:
        event_color='tab:orange'
    else:
        event_color=event_lines

    #if estimated is an fitted HMP instance, calculate topos and times 
    if isinstance(estimated, (xr.DataArray, xr.Dataset)) and 'event' in estimated:
        if ydim is None:
            if 'n_events' in estimated.dims and estimated.n_events.count() > 1: #and there are multiple different fits (eg backward estimation)
                ydim = 'n_events' #set ydim to 'n_events'
            elif 'condition' in estimated.dims:
                ydim = 'condition'
        if not skip_channels_computation:
            channels = init.compute_topologies(channels, estimated, init, ydim, peak=False).data #compute topologies
        times = init.compute_times(init, estimated, mean=True, extra_dim=ydim, as_time=as_time).data #compute corresponding times
    else:#assumes times/topologies already computed
        times = estimated 
    times = times * time_step

    if len(np.shape(channels)) == 2:
        channels = channels[np.newaxis]
    
    if cond_plot: #reverse order, to make correspond to condition maps
        channels = np.flipud(channels)
        times = np.flipud(times)

    n_iter = np.shape(channels)[0]

    if n_iter == 1:
        times = [times]
    times = np.array(times)

    if as_time:
        time_step = 1000/init.sfreq #time_step still needed below
    event_size = init.event_width_samples * time_step
    
    #based the size of the topologies on event_size and magnify or only on magnify
    if topo_size_scaling: #does topo width scale with time interval of plot?
        topo_size = event_size * magnify
    else:
        timescale = (max_time if max_time else (np.max(times_to_display) * time_step * 1.05)) + time_step
        topo_size = .08 * timescale * magnify #8% of time scale

    #fix vmin/vmax across topos, while keeping symmetric
    if vmax == None: #vmax = absolute max, unless no positive values
        vmax = np.nanmax(np.abs(channels[:])) 
        vmin = -vmax if np.nanmin(channels[:]) < 0 else 0
        if np.nanmax(channels[:]) < 0: vmax = 0

    #make axis
    if ax is None:
        if figsize == None:
            if n_iter == 1:
                figsize = (12, .7 * np.max([magnify,1.8])) #make sure they don't get too flat
            else:
                figsize = (12, n_iter*.7) #make sure they don't get too flat

        _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        return_ax = False

    axes = []

    #plot row by row
    rowheight = 1/n_iter 
    for iteration in np.arange(n_iter):
        times_iteration = times[iteration]
        missing_evts = np.where(np.isnan(times_iteration))[0]
        times_iteration = np.delete(times_iteration,missing_evts)
        channels_ = channels[iteration,:,:]
        channels_ = np.delete(channels_, missing_evts, axis=0)
        n_event = len(times_iteration)
        ylow = iteration * rowheight

        #plot topology per event
        for event in np.arange(n_event):

            #topology
            axes.append(ax.inset_axes([times_iteration[event] + event_size/2 - topo_size / 2, ylow+.1*rowheight,
                                topo_size, rowheight*.8], transform=ax.get_xaxis_transform())) 
            plot_topomap(channels_[event,:], channel_position, axes=axes[-1], show=False,
                         cmap=cmap, vlim=(vmin, vmax), sensors=sensors, contours=contours)

            #lines/fill of detected event
            if event_lines:
                #bottom of row + 5% if n_iter > 1
                ylow = iteration * rowheight if n_iter == 1 else iteration * rowheight + .05 * rowheight
                #top of row - 5% if n_iter > 1
                yhigh = (iteration + 1) * rowheight if n_iter == 1 else (iteration + 1) * rowheight - .05 * rowheight

                ax.vlines(times_iteration[event],ylow,yhigh, linestyles='dotted',color=event_color,alpha=.5,transform=ax.get_xaxis_transform())
                ax.vlines(times_iteration[event]+event_size,ylow, yhigh, linestyles='dotted',color=event_color,alpha=.5, transform=ax.get_xaxis_transform())
                ax.fill_between(np.array([times_iteration[event],times_iteration[event]+event_size]), ylow, yhigh, alpha=0.15,color=event_color, transform=ax.get_xaxis_transform(),edgecolor=None)

        #add lines per condition
        if cond_plot and times_to_display[iteration]:
            #bottom of row + 5% if n_iter > 1
            ylow = iteration * rowheight if n_iter == 1 else iteration * rowheight + .05 * rowheight
            #top of row - 5% if n_iter > 1
            yhigh = (iteration + 1) * rowheight if n_iter == 1 else (iteration + 1) * rowheight - .05 * rowheight
            ax.vlines(times_to_display[iteration] * time_step,ylow,yhigh, linestyles='--',transform=ax.get_xaxis_transform())

    #legend
    if colorbar:
        cheight = "100%" if n_iter == 1 else f"{200/n_iter}%" 
        axins = inset_axes(ax, width="0.5%", height=cheight, loc="lower left", bbox_to_anchor=(1.025, 0, 2, 1), bbox_transform=ax.transAxes, borderpad=0)
        if isinstance(channel_position, Info):
            lab = 'Voltage' if channel_position['chs'][0]['unit'] == 107 else channel_position['chs'][0]['unit']._name
        else:
            lab = 'Voltage'
        plot_brain_colorbar(axins, dict(kind='value', lims = [vmin,0,vmax]),colormap=cmap, label = lab, bgcolor='.5', transparent=None)

    #plot ylabels
    if isinstance(ylabels, dict):
        tick_labels = [str(x) for x in list(ylabels.values())[0]]
        if cond_plot: tick_labels.reverse() 
        ax.set_yticks(np.arange(len(list(ylabels.values())[0]))+.5,
                      tick_labels)
        ax.set_ylabel(str(list(ylabels.keys())[0]))
    else:
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
   
    #add vlines across all rows
    if not cond_plot:
        __display_times(ax, times_to_display, 0, time_step, max_time, times, linecolors, n_iter)

    if not return_ax:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, n_iter) #-1
        ax.set_xlabel(xlabel)
        if title:
            ax.set_title(title)
        if np.any(max_time) == None and np.any(times_to_display) == None:
            ax.set_xlim(0, np.nanmax(times)+np.nanmax(times)/10)
    if plt.get_backend()[0:2] == 'Qt' or plt.get_backend() == 'nbAgg': #fixes issue with yscaling
    #    plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85,bottom=.2)   #tight layout didn't work anymore
    if return_ax:
        ax.set_ylim(0, n_iter) #-1
        return ax
    else:
        plt.show()    

def save_model_topos(channels, estimated, channel_position, init, fname='topo', figsize=None, dpi=300, cmap='Spectral_r',
                vmin=None, vmax=None, sensors=False, contours=6, colorbar=True):
    '''
    Saving the event topologies to files, one per topology. Typically used for saving high quality topos.
    
    Parameters
    ----------
    channels : xr.Dataarray 
       the original EEG data in HMP format
    estimated : hmp object
        a fitted hmp (either from fit_single, fit_single_conds, or backward_estimation)
    channel_position : ndarray
        Either a 2D array with dimension channel and [x,y] storing channel location in meters or an info object from
        the mne package containning digit. points for channel location
    init : float
        initialized HMP object
    fname : str
        file name, other model specifications will be appended. Should not include extension.
    figsize : list | tuple | ndarray
        Length and heigth of the matplotlib plot
    dpi : float
        DPI of the  matplotlib plot
    cmap : str
        Colormap of matplotlib
    vmin : float
        Colormap limits to use (see https://mne.tools/dev/generated/mne.viz.plot_topomap.html). If not explicitly
        set, uses min across all topos while keeping colormap symmetric.
    vmax : float
        Colormap limits to use (see https://mne.tools/dev/generated/mne.viz.plot_topomap.html). If not explicitly
        set, uses max across all topos while keeping colormap symmetric.
    sensors : bool
        Whether to plot the sensors on the topologies
    contours : int / array_like
        The number of contour lines to draw (see https://mne.tools/dev/generated/mne.viz.plot_topomap.html)
    colorbar : bool
        Whether a colorbar is saved in a separate file (fname_colorbar)
      
    '''

    from mne.viz import plot_brain_colorbar, plot_topomap
    from mne.io.meas_info import Info

    plot_type = 'default'
    ydim = None

    #if condition estimates, prep conds
    if isinstance(estimated, xr.Dataset) and 'condition' in estimated.dims:
        plot_type = 'cond'
        conds = estimated['cond'].values
        n_cond = estimated.parameters.shape[0]
        cond_names = estimated.clabels
        ydim = 'condition'
    elif 'n_events' in estimated.dims and estimated.n_events.count() > 1:
        plot_type = 'backward'
        ydim = 'n_events'

    #calculate topos
    channels = init.compute_topologies(channels, estimated, init, ydim).data #compute topologies
    
    #fix vmin/vmax across topos, while keeping symmetric
    if vmax == None: #vmax = absolute max, unless no positive values
        vmax = np.nanmax(np.abs(channels[:])) 
        vmin = -vmax if np.nanmin(channels[:]) < 0 else 0
        if np.nanmax(channels[:]) < 0: vmax = 0

    #make axis
    if figsize == None:
        figsize = (2,2)

    #make sure it doesn't display plots
    mplint = mpl.is_interactive()
    if mplint:
        plt.ioff()

    #plot model by model
    if plot_type == 'default':

        #remove potential nans
        nans = np.where(np.isnan(np.mean(channels,axis=1)))[0]
        channels = np.delete(channels, nans, axis=0)
        n_event = channels.shape[0]

        for event in range(n_event):
            plot_name = fname + '_ev' + str(event+1) + '.png'
            fig, ax = plt.subplots(figsize=figsize)
            plot_topomap(channels[event,:], channel_position, show=False, cmap=cmap, vlim=(vmin, vmax), sensors=sensors, contours=contours, axes=ax)
            fig.savefig(plot_name,dpi=dpi,transparent=True)    

    elif plot_type == 'cond': #reverse order, to make correspond to condition maps
        channels = np.flipud(channels)
        cond_labels = [str(x[0]) for x in list(cond_names.values())[0]]
        
        #plot by condition
        for cond in range(n_cond):
            cond_name = cond_labels[cond]
            cond_channels = channels[cond]

            #remove potential nans
            nans = np.where(np.isnan(np.mean(cond_channels,axis=1)))[0]
            cond_channels = np.delete(cond_channels, nans, axis=0)
            n_event = cond_channels.shape[0]

            for event in range(n_event):
                plot_name = fname + '_cond' + cond_name + '_ev' + str(event+1) + '.png'
                fig, ax = plt.subplots(figsize=figsize)
                plot_topomap(cond_channels[event,:], channel_position, show=False, cmap=cmap, vlim=(vmin, vmax), sensors=sensors, contours=contours, axes=ax)
                fig.savefig(plot_name,dpi=dpi,transparent=True)    
    
    elif plot_type == 'backward':

        # plot by number of events
        for n_eve in range(channels.shape[0]):
            back_channels = channels[n_eve]

            #remove potential nans
            nans = np.where(np.isnan(np.mean(back_channels,axis=1)))[0]
            back_channels = np.delete(back_channels, nans, axis=0)
            n_event = back_channels.shape[0]

            for event in range(n_event):
                plot_name = fname + '_backward' + str(n_event) + '_ev' + str(event+1) + '.png'
                fig, ax = plt.subplots(figsize=figsize)
                plot_topomap(back_channels[event,:], channel_position, show=False, cmap=cmap, vlim=(vmin, vmax), sensors=sensors, contours=contours, axes=ax)
                fig.savefig(plot_name,dpi=dpi,transparent=True)    

    #legend
    if colorbar:
        if isinstance(channel_position, Info):
            lab = 'Voltage' if channel_position['chs'][0]['unit'] == 107 else channel_position['chs'][0]['unit']._name
        else:
            lab = 'Voltage'
        fig, ax = plt.subplots(figsize=(.5,2))
        plot_brain_colorbar(ax, dict(kind='value', lims = [vmin,0,vmax]),colormap=cmap, label = lab, bgcolor='.5', transparent=None)
        fig.savefig(fname + '_colorbar.png',dpi=dpi,transparent=True,bbox_inches='tight')    

    #switch plotting back on
    if mplint:
        plt.ion()


def plot_components_sensor(hmp_data, positions):
    """
      This function is used to visualize the topomap of the HMP principal components.
     
     Parameters
     ----------
     	 hmp_data: xr.Dataset
            Data returned from the function hmp.utils.transform_data()
     	 positions: mne.info | ndarray 
            List of x and y positions to plot channels on head model OR MNE info object
    """
    from mne.viz import plot_topomap
    fig, ax = plt.subplots(1,len(hmp_data.attrs['pca_weights'].component))
    for comp in hmp_data.attrs['pca_weights'].component:
        plot_topomap(hmp_data.attrs['pca_weights'].values[:,comp], positions, axes=ax[comp], show=False, cmap='Spectral_r')
    plt.show()


def plot_loocv(loocv_estimates, pvals=True, test='t-test', figsize=(16,5), indiv=True, ax=None, mean=False, additional_points=None, return_pvals=False):
    '''
    Plotting the LOOCV results
    
    Parameters
    ----------
    loocv_estimates : ndarray or xarra.DataArray
        results from a call to hmp.utils.loocv()
    pvals : bool
        Whether to display the pvalue with the associated test
    test : str
        which statistical test to compute for the difference in LOOCV-likelihood (one sample t-test or sign test)
    figsize : list | tuple | ndarray
        Length and heigth of the matplotlib plot
    indiv : bool
        Whether to plot individual lines
    ax : matplotlib.pyplot.ax
        Matplotlib object on which to draw the plot, can be useful if you want to control specific aspects of the plots
        outside of this function
    mean : bool
        Whether to plot the mean
    additional_points : 
        Additional likelihood points to be plotted. Should be provided as a list of tuples 
        containing the x coordinate and loocv estimates with a single event, e.g. [(5,estimates)].
    return_pvals : bool 
        Whether to return the pvalues

    Returns
    -------
    ax : matplotlib.pyplot.ax
        if ax was specified otherwise returns the plot
    '''
    ax=None
    if return_pvals:
        pvals = True
    if pvals:
        if test == 'sign':
            from statsmodels.stats.descriptivestats import sign_test 
        elif test == 't-test':
            from scipy.stats import ttest_1samp
        else:
            raise ValueError('Expected sign or t-test argument to test parameter')
    if ax is None:
        fig, ax = plt.subplots(1,2, figsize=figsize)
        return_ax = False
    else:
        return_ax = True
    loocv_estimates = loocv_estimates.dropna('n_event', how='all')
    
    #stats
    diffs, diff_bin, labels = [],[],[]
    pvalues = []
    for n_event in np.arange(2,loocv_estimates.n_event.max()+1):
        diffs.append(loocv_estimates.sel(n_event=n_event).data - loocv_estimates.sel(n_event=n_event-1).data) #differences
        diff_bin.append([1 for x in diffs[-1] if x > 0]) #nr of positive differences
        labels.append(str(n_event-1)+'->'+str(n_event))

        if pvals:
            if test == 'sign':
                diff_tmp = np.array(diffs)
                diff_tmp[np.isnan(diff_tmp)] = -np.inf 
                pvalues.append((sign_test(diff_tmp[-1])))
            elif test == 't-test':
                pvalues.append((ttest_1samp(diffs[-1], 0, alternative='greater')))

     
    #first plot
    if mean:
        alpha = .4#for the indiv plot
        marker_indiv = '.'
        means = np.nanmean(loocv_estimates.data,axis=1)[::-1]
        errs = (np.nanstd(loocv_estimates.data,axis=1)/np.sqrt(len(loocv_estimates.participant)))[::-1]
        ax[0].errorbar(x=np.arange(len(means))+1, y=means, \
                 yerr= errs, marker='o', color='k')
        #if pvals:
        #    for n_event in np.arange(2,loocv_estimates.n_event.max()+1):       
        #        mean = means[n_event-1] * 1.1
        #        ax[0].text(x=n_event-.25, y=mean, s=str(int(np.nansum(diff_bin[n_event-2])))+'/'+ str(len(diffs[-1])) + ': ' + str(np.around(pvalues[n_event-2][-1],3)), ha='right')
    else:
        alpha=1
        marker_indiv='o'
    if indiv:
        for loo in loocv_estimates.T:
            ax[0].plot(loocv_estimates.n_event,loo, alpha=alpha,marker=marker_indiv)
    ax[0].set_ylabel('LOOCV Loglikelihood')
    ax[0].set_xlabel('Number of events')
    ax[0].set_xticks(ticks=loocv_estimates.n_event)

    if additional_points: #only plot average for now
        if not isinstance(additional_points,list):
            additional_points = [additional_points]
        for ap in additional_points:
            xap = ap[0]
            meanap = np.mean(ap[1].values)
            err = np.nanstd(ap[1].values)/np.sqrt(len(ap[1].values))
        ax[0].errorbar(x=xap,y=meanap, yerr=err, marker='o')
      
    #second plot
    ax[1].plot(diffs,'.-', alpha=.6)
    ax[1].set_xticks(ticks=np.arange(0,loocv_estimates.n_event.max()-1), labels=labels)
    ax[1].hlines(0,0,len(np.arange(2,loocv_estimates.n_event.max())),color='lightgrey',ls='--')
    ax[1].set_ylabel('Change in likelihood')
    ax[1].set_xlabel('')
        
    if pvals:
        ymin = np.min(diffs[:])
        ymintext = ymin - (np.max(diffs[:]) - ymin) * .05
        ymin = ymin - (np.max(diffs[:]) - ymin) * .1
        ax[1].set_ylim(bottom=ymin)
        for n_event in np.arange(2,loocv_estimates.n_event.max()+1):       
            ax[1].text(x=n_event-2, y=ymintext, s=str(int(np.nansum(diff_bin[n_event-2])))+'/'+ str(len(diffs[-1])) + ': ' + str(np.around(pvalues[n_event-2][-1],3)), ha='center')
    
    if return_ax:
        if return_pvals:
            return [ax, pvalues]
        else:
            return ax
    else:
        plt.tight_layout()
        plt.show()
        if return_pvals:
            return pvalues
        

def plot_latencies_average(times, time_step=1, labels=[], colors=default_colors,
    figsize=None, errs='ci', max_time=None, times_to_display=None):
    print('Deprecated. Use plot_latencies() instead.')
    

def plot_distribution(times, colors=default_colors, xlims=False, figsize=(8, 3), survival=False):
    '''
    Plots the distribution of each stage latencies

    Parameters
    ----------
    times : ndarray
        2D or 3D numpy array, Either trials * events or conditions * trials * events
    colors : ndarray
        array of colors for the different stages
    xlims : tuple | list
        lower and upper limit of the x (time) axis
    figsize : list | tuple | ndarray
        Length and heigth of the matplotlib plot
    survival : bool
        Whether to draw density plots or survival density plots

    '''
    f, axs = plt.subplots(1,1, figsize=figsize, dpi=100)

    if len(np.shape(times)) == 2:
        times = np.asarray([times],dtype = 'object')
    cycol = cycle(colors)
    for iteration in times:
        for stage in iteration.T:
            if survival:
                axs.plot(1-stage.cumsum(axis=0), color=next(cycol) )
            else: 
                axs.plot(stage,color=next(cycol) )
    axs.set_ylabel('p(event)')
    axs.set_xlabel('Time (in samples)')
    if xlims:
        axs.set_xlim(xlims[0], xlims[1])
    return axs

def __display_times(ax, times_to_display, yoffset, time_step, max_time, times, linecolors, ymax=1):
    n_iter = len(times)
    times = np.asarray(times,dtype=object)
    if isinstance(times_to_display, (np.ndarray, np.generic)):
        ax.vlines(times_to_display*time_step, yoffset-1.1, yoffset+ymax+1.1, ls='--', colors=linecolors)
        ax.set_xlim(-1*time_step, np.max(times_to_display)*time_step * 1.05)
    if max_time:
        ax.set_xlim(-1*time_step, max_time)
    return ax

def plot_latencies_gamma(gammas, event_width=0, time_step=1, labels=[''], colors=default_colors, 
                         figsize=False, times_to_display=None, max_time=None, kind='bar', legend=False):
    '''
    Plots the average of stage latencies based on the estimated scale parameter of the gamma distributions

    Parameters
    ----------
    gammas : ndarray
        instance of hmp.hmp.parameters
    event_width : float
        Size of the event in time unit given sampling frequency, if drawing a fitted object using hmp you 
        can provide the event_width_sample of fitted hmp (e.g. init.event_width_sample)
    time_step : float
        What unit to multiply all the times with, if you want to go on the second or millisecond scale you can provide 
        1/sf or 1000/sf where sf is the sampling frequency of the data
    labels : tuples | list
        labels to draw on the y axis
    colors : ndarray
        array of colors for the different stages
    figsize : list | tuple | ndarray
        Length and heigth of the matplotlib plot
    times_to_display : ndarray
        Times to display (e.g. Reaction time or any other relevant time) in the time unit of the fitted data
    max_time : float
        limit of the x (time) axe
    kind : str
        Whether to draw a bar plot ('bar') or line connected point plot ('point')
    legend : bool
        Whether to draw legend handle or not
    '''
    j = 0
    
    if len(np.shape(gammas)) == 2:
        gammas = [gammas]#np.reshape([gammas], (1, np.shape(gammas)[0],np.shape(gammas)[1]))
    if not figsize:
        figsize = (8, 1*len(gammas)+2)
    f, axs = plt.subplots(1,1, figsize=figsize, dpi=100)
    for time in gammas:
        cycol = cycle(colors)
        try:
            time = time[np.isfinite(time)]
        except:
            print('todo')
        time = np.reshape(time, (np.shape(gammas)[1],np.shape(gammas)[2]))
        n_stages = int(len(time))
        colors = [next(cycol) for x in np.arange(n_stages)]
        colors.append(next(cycol))
        if kind=='bar':
            axs.bar(np.arange(n_stages)+1, time[:,0] *  time[:,1], color='w', edgecolor=colors)
        elif kind == 'point':
            axs.plot(np.arange(n_stages)+1, time[:,0] * time[:,1],'o-', label=labels[j], color=colors[j])
        j += 1
    #plt.xticks(np.arange(len(labels)),labels)
    #plt.xlim(0-1,j)
    #__display_times(axs, mean_d, np.arange(np.shape(gammas)[0]), time_step, max_time, gammas)
    axs.set_ylabel('Stages durations (Gamma)')
    axs.set_xlabel('Stage number')
    # Hide the right and top spines
    axs.spines.right.set_visible(False)
    axs.spines.top.set_visible(False)
    # Only show ticks on the left and bottom spines
    axs.yaxis.set_ticks_position('left')
    if legend:
        axs.legend()
    return axs

def plot_latencies(times, init=None, time_step=1, labels=[], colors=default_colors,
    figsize=False, errs=None, kind='bar', legend=False, max_time=None, as_time=True):
    '''
    Plots the average of stage latencies with choosen errors bars

    Parameters
    ----------
    times : ndarray | hmp results object
        2D or 3D numpy array, either trials * events or conditions * trials * events
        hmp results object either of fit_single or fit_single_conds
    init : hmp object
        Initialized hmp object, required if times is an hmp results object
    event_width : float
        Display size of the event in time unit given sampling frequency, if drawing a fitted object using hmp you 
        can provide the event_width_sample of fitted hmp (e.g. init.event_width_sample)
    time_step : float
        What unit to multiply all the times with, if you want to go on the second or millisecond scale you can provide 
        1/sf or 1000/sf where sf is the sampling frequency of the data
    labels : tuples | list
        labels to draw on the y axis
    colors : ndarray
        array of colors for the different stages
    figsize : list | tuple | ndarray
        Length and heigth of the matplotlib plot
    errs : str
        Whether to display no error bars (None), 95% confidence interval ('ci'), standard deviation ('std'),
        or standard error ('se')
    times_to_display : ndarray
        Times to display (e.g. Reaction time or any other relevant time) in the time unit of the fitted data
    max_time : float
        limit of the x (time) axe
    kind : str
        bar or point
    as_time : bool
        if true, plot time (ms) instead of samples (time_step takes precedence). Ignored if times are provided as array.
    '''

    from seaborn.algorithms import bootstrap #might be too much to ask for seaborn install?
       
    if as_time and time_step != 1:
        print('time_step takes precedence over as_time')
        as_time = False

    #if hmp estimates are provided, calculate time
    hmp_obj = False
    if isinstance(times, (xr.DataArray, xr.Dataset)):
        assert init is not None, 'If hmp results object provided, init is a required parameter.'
        hmp_obj = True
        ydim = None
        if 'n_events' in times.dims and times.n_events.count() > 1: #and there are multiple different fits (eg backward estimation)
            ydim = 'n_events' 
        elif 'condition' in times.dims:
            ydim = 'condition'
        avg_durations = init.compute_times(init, times, mean=True, duration=True,add_rt=True, extra_dim=ydim, as_time=as_time).data 
        avg_times = init.compute_times(init, times, mean=True, duration=False,add_rt=True, extra_dim=ydim, as_time=as_time).data 
        if errs is not None:
            errorbars = init.compute_times(init, times, duration=True,add_rt=True, extra_dim=ydim, as_time=as_time, errorbars=errs)
        if len(avg_times.shape) == 1:
            avg_durations = np.expand_dims(avg_durations, axis=0)
            avg_times = np.expand_dims(avg_times, axis=0)
            if errs is not None:
                errorbars = np.expand_dims(errorbars, axis=0)

        avg_durations = avg_durations * time_step
        avg_times = avg_times * time_step
        if errs is not None:
            errorbars = errorbars * time_step

        if ydim == 'condition': #reverse order, to make correspond to condition maps
            avg_durations = np.flipud(avg_durations)
            avg_times = np.flipud(avg_times)
            if errs is not None:
                errorbars = np.flipud(errorbars)

        n_model = avg_times.shape[0]
    else: #no hmp object
        if len(np.shape(times)) == 2:
            times = [times]
        times = [time * time_step for time in times]
        if isinstance(times[0], xr.DataArray):
            times = [time.values for time in times]
        n_model = len(times)
        avg_times=[]
        avg_durations=[]
        errorbars =[]
        
    if labels == []:
        if hmp_obj and ydim == 'condition':
            labels = [str(x) for x in reversed(list(times.clabels.values())[0])] 
        else:
            labels = np.arange(n_model)

    if not figsize:
        figsize = (8, 1*n_model+2)
    f, axs = plt.subplots(1,1, figsize=figsize, dpi=100)

    cycol = cycle(colors)
    cur_colors = [next(cycol) for x in np.arange(n_model)] #color per condition/model (line plot)
    
    for j in range(n_model): #per condition/model
        if hmp_obj: #all calcs have been done
            avg_times_model = np.hstack((0,avg_times[j,:]))
            avg_durations_model = avg_durations[j,:]
            n_stages = len(avg_durations_model)
            errorbars_model = errorbars[j,:,:] if errs is not None else None
        else: #call error bars etc
            time = times[j]
            duration = np.diff(time, axis=1, prepend=0) #time per trial, duration per trial
            avg_times_model = np.hstack((0, np.mean(time, axis=0)))
            avg_durations_model = np.mean(duration, axis=0)
            n_stages = len(avg_durations_model)
            errorbars_model = None if errs is None else np.zeros((2,n_stages))
            if errs == 'ci':
                for st in range(n_stages):
                    errorbars_model[:, st] = np.squeeze([np.nanpercentile(bootstrap(duration[:,st]), q=[2.5,97.5])])
                    errorbars_model[:, st] = np.abs(errorbars_model[:, st]-avg_durations_model[st])
            elif errs == 'std':
                errorbars_model = np.tile(np.std(duration, axis=0), (2,1))
            elif errs == 'se':
                print('Cannot calculate SE for non-hmp object')
            avg_durations.append(np.nanmax(avg_durations_model)) #for determining axes
            errorbars.append(np.nanmax(errorbars_model))
            avg_times.append(np.nanmax(avg_times_model))
            
        if kind == 'bar':
            cycol = cycle(colors) #get a color per stage
            cur_colors = [next(cycol) for x in np.arange(n_stages)]

            # #remove 0 stages and associated color
            if np.any(np.isnan(avg_durations_model)):
                missing_evts = np.where(np.isnan(avg_durations_model))[0]
                avg_durations_model = np.delete(avg_durations_model,missing_evts)
                avg_times_model = np.delete(avg_times_model, missing_evts + 1)
                if errs is not None:
                    errorbars_model = np.delete(errorbars_model, missing_evts, axis=1)
                cur_colors = np.delete(cur_colors, missing_evts)
                n_stages = n_stages - len(missing_evts)

            for st in reversed(range(n_stages)): #can't deal with colors in one call
                plt.barh(j, avg_durations_model[st],left=avg_times_model[st], color='w', edgecolor=cur_colors[st])
                if errs is not None:
                    plt.errorbar(np.repeat(avg_times_model[st+1],2), np.repeat(j,2), xerr=errorbars_model[:,st], color=cur_colors[st], fmt='none', capsize=10)

        elif kind == 'point':
            plt.errorbar(np.arange(n_stages)+1, avg_durations_model, 
                    yerr=errorbars_model, color=cur_colors[j], fmt='o-', capsize=10, label=labels[j])
        else:
            raise ValueError('Unknown \'kind\'')
    
    #general settings
    if kind == 'bar':
        plt.yticks(np.arange(len(labels)),labels)
        plt.ylim(0-1,j+1)
        
        if not max_time:
            max_time = (np.nanmax(avg_times) + np.nanmax(errorbars) if errs else np.nanmax(avg_times)) * 1.05               
        axs.set_xlim(0, max_time)
        if time_step == 1 and not as_time:
            plt.xlabel('Cumulative stage durations from stimulus onset (samples)')
        elif as_time:
            plt.xlabel('Cumulative stage durations from stimulus onset (ms)')
        else:
            plt.xlabel('Cumulative stage durations from stimulus onset')
    elif kind == 'point':                 
        plt.xlim(1-.5, n_stages+.5)

        max_y = (np.nanmax(avg_durations) + np.nanmax(errorbars) if errs else np.nanmax(avg_durations)) * 1.05
        axs.set_ylim(0, max_y)

        if time_step == 1 and as_time == False:
            plt.ylabel('Stage durations (samples)')
        elif as_time:
            plt.ylabel('Stage durations (ms)')
        else:
            plt.ylabel('Stage durations')
        plt.xlabel('Stage')
    
    plt.tight_layout()
    # Hide the right and top spines
    axs.spines.right.set_visible(False)
    axs.spines.top.set_visible(False)
    # Only show ticks on the left and bottom spines
    axs.yaxis.set_ticks_position('left')
    axs.xaxis.set_ticks_position('bottom')
    if legend:
        axs.legend()
    return axs


def plot_iterations(iterations, eeg_data, init, positions, dims=['magnitudes','parameters'], alpha=1, ax=None):
    """
    Currently DEPRECATED
    """
    from hmp.models import hmp
    if 'iteration' not in iterations.dims:
        try:
            iterations['iteration'] = [0]
        except:
            iterations = iterations.expand_dims({'iteration':[0]}, axis=1)
            iterations['iteration'] = [0]
    for iteration in iterations.iteration[:-1]:
        selected = init.fit_single(iterations.sel(iteration=iteration)[dims[0]].dropna(dim='event').event[-1].values+1,\
            magnitudes = iterations.sel(iteration=iteration)[dims[0]].dropna(dim='event'),\
            parameters = iterations.sel(iteration=iteration)[dims[1]].dropna(dim='stage'),\
            maximization=False, verbose=False)
        #Visualizing
        plot_topo_timecourse(eeg_data, selected, positions,  init, magnify=1, sensors=False,ax=ax)
    

def plot_bootstrap_results(bootstrapped, info, init, model_to_compare=None, epoch_data=None):
    """
     Plot bootstrapped time courses. This is a wrapper around plot_topo_timecourse to make it easier to use in other functions
     
     Parameters
     ----------
     	 bootstrapped: xr.Dataset 
            The resulting data from a call to hmp.resample.bootstrapping()
     	 info: 
            The MNE info/electrode positions
     	 init: 
            The initialized HMP model
     	 model_to_compare: 
            If None ( default ) the model to compare is taken as the maximal n_event model among the bootstrapped models 
     	 epoch_data: If None ( default
    """
    from hmp.resample import event_occurence
    maxboot_model, labels, counts, event_number, label_event_num = event_occurence(bootstrapped, model_to_compare)
    fig, axes = plt.subplot_mosaic([['a', 'a'], ['b', 'c'], ['b', 'c']],
                              layout='constrained')
    n_events = int(maxboot_model.event.max())
    if model_to_compare is None: 
        plot_topo_timecourse(maxboot_model.channels_activity.values, maxboot_model.event_times.values, info, init,ax=axes['a'],event_lines=False, colorbar=False, times_to_display=None)
        times = maxboot_model.event_times#init.compute_times(init, maxboot_model, mean=True)#computing predicted event times
    else:
        plot_topo_timecourse(epoch_data, model_to_compare, info, init,ax=axes['a'], event_lines=False, colorbar=False, times_to_display=None)
        times = init.compute_times(init, model_to_compare, mean=True)
        maxboot_model = model_to_compare
    counts_adjusted = np.zeros(n_events+1)
    counts_adjusted[:len(counts)] = counts
    for event in times.event.values:
        axes['a'].text(times.sel(event=event).values+init.event_width_samples/2.5, .7, event)
    # axes['a'].set_xlabel('Time (samples)')
    axes['b'].bar(maxboot_model.event,counts_adjusted)
    axes['b'].set_xlabel('Event number')
    axes['b'].set_xticks(maxboot_model.event)
    axes['b'].set_ylabel('Frequency')
    axes['b'].set_ylim(0,1)
    
    axes['c'].bar(label_event_num,event_number)
    axes['c'].set_xlabel('Number of events')
    
    axes['c'].set_ylabel('Frequency')
    axes['c'].set_ylim(0,1)
    axes['a'].set_xticks([])
    axes['a'].spines[['right', 'top', 'bottom']].set_visible(False)
    axes['b'].spines[['right', 'top']].set_visible(False)
    axes['c'].spines[['right', 'top']].set_visible(False)
    # plt.tight_layout()
    plt.show()
    
def plot_expected_distribution(distribution, mean, shape, location=0, xmax=300, xmin=0, num=100, label='distribution', color=None, ax=None, display_mean=False):
    '''
    Plot available distribution with a scale and shape parameter
    '''
    match distribution:
        case 'gamma':
            from scipy.stats import gamma as sp_dist
            from hmp.utils import gamma_scale_to_mean,gamma_mean_to_scale
            scale_to_mean, mean_to_scale = gamma_scale_to_mean,gamma_mean_to_scale
        case 'lognormal':
            from scipy.stats import lognorm as sp_dist
            from hmp.utils import logn_scale_to_mean,logn_mean_to_scale
            scale_to_mean, mean_to_scale = logn_scale_to_mean,logn_mean_to_scale
        case 'wald':
            from scipy.stats import invgauss as sp_dist
            from hmp.utils import wald_scale_to_mean,wald_mean_to_scale
            scale_to_mean, mean_to_scale = wald_scale_to_mean,wald_mean_to_scale
        case 'weibull':
            from scipy.stats import weibull_min as sp_dist
            from hmp.utils import weibull_scale_to_mean,weibull_mean_to_scale
            scale_to_mean, mean_to_scale = weibull_scale_to_mean,weibull_mean_to_scale
        case 'maxwell-boltzmann':
            from scipy.stats import chi as sp_dist
            from hmp.utils import maxb_scale_to_mean,maxb_mean_to_scale
            shape = 3
            scale_to_mean, mean_to_scale = maxb_scale_to_mean,maxb_mean_to_scale
        case 'log-logistic':
            from scipy.stats import fisk as sp_dist
            from hmp.utils import fisk_scale_to_mean,fisk_mean_to_scale
            scale_to_mean, mean_to_scale = fisk_scale_to_mean,fisk_mean_to_scale
        case 'rayleigh':
            from scipy.stats import chi as sp_dist
            from hmp.utils import ray_scale_to_mean,ray_mean_to_scale
            shape = 2
            scale_to_mean, mean_to_scale = ray_scale_to_mean,ray_mean_to_scale
        case 'half-normal':
            from scipy.stats import chi as sp_dist
            from hmp.utils import halfn_scale_to_mean,halfn_mean_to_scale
            shape = 1
            scale_to_mean, mean_to_scale = halfn_scale_to_mean,halfn_mean_to_scale
        case 'uniform':
            from scipy.stats import uniform as sp_dist
            from hmp.utils import uniform_scale_to_mean,uniform_mean_to_scale
            scale_to_mean, mean_to_scale = uniform_scale_to_mean,uniform_mean_to_scale
            location=None
        case _:
                raise ValueError(f'Unknown Distribution {distribution}')
    if ax is None:
        ax = plt
    x = np.linspace(xmin, xmax, num=num)
    if distribution != 'uniform': 
        y = sp_dist.cdf(x, shape, scale=mean_to_scale(mean, shape), loc=location)
    else:
        y = sp_dist.cdf(x, shape, scale=mean_to_scale(mean, shape))
    y = np.diff(y, prepend=0)#going to pmf
    ax.plot(x, y, label=label, color=color)
    if display_mean:
        if distribution != 'uniform': 
            ax.vlines(sp_dist.mean(shape, scale=mean_to_scale(mean, shape), loc=location), np.min(y), np.max(y), color=color)
        else: 
            ax.vlines(sp_dist.mean(shape, scale=mean_to_scale(mean, shape)), np.min(y), np.max(y), color=color)

def erp_data(epoched_data, times, channel,n_samples=None, pad=1):
    '''
    Create a data array compatible with the plot ERP function. Optionnally this function can resample the epochs to fit some provided times (e.g. onset of the events)

    Parameters
    ----------
     	epoched_data: xr.Dataset 
            Epoched physiological data with dims 'participant'X 'epochs' X 'channels'X 'samples'
        times: xr.Dataset
            Times between wich to extract or resample the data with dims 'trial_x_participant' X 'event'
        channel: str
            For which channel to extract the data
        n_samples: int
            How many samples to resample on if any
        pad: int
            padding added to the beginning and the end of the signal
    
    Returns
    -------
    data : nd.array
        array containing the extracted times for each epoch and stage with format epochs X events X samples.
    '''
    epoched_data = epoched_data.sel(channels=channel)
    if n_samples is None:
        data = np.zeros((len(times.trial_x_participant), len(times.event), len(epoched_data.samples)))*np.nan
    else:
        data = np.zeros((len(times.trial_x_participant), len(times.event), n_samples))*np.nan

    for i,trial in enumerate(times.trial_x_participant):
        for j,event in enumerate(times.event):
            if event == 0:
                sub_prevtime = 0
            else:
                sub_prevtime = times.sel(trial_x_participant=trial, event=event-1).data
            sub_times = times.sel(trial_x_participant=trial, event=event).data-1
            time_diff = sub_times-sub_prevtime
            if time_diff > 1:#rounds up to 0 otherwise
                subset = epoched_data.sel(trial_x_participant=trial, samples=slice(sub_prevtime,sub_times))
                if n_samples is None:
                    limit = np.sum(~subset.data.isnull()).values
                    data[i,j,:limit] = subset.data
                else:
                    padded_data = np.concatenate([np.repeat([subset.data[0]],pad), subset.data, np.repeat([subset.data[-1]], pad)])
                    #resample_poly seems to have better results than resample
                    data[i,j] = ssignal.resample_poly(padded_data,n_samples*10,len(padded_data.data)*(n_samples/10), padtype='median')
                    # data[i,j] = ssignal.resample(padded_data.data, n_samples)
    return data


def plot_erp(times, data, color='k',ax=None, minmax_lines=None, upsample=1, bootstrap=None, label=None):
    '''
    Plot the ERP based on the times extracted by HMP (or just stimulus and response and the data extracted from ```erp_data```.

    
    Parameters
    ----------
        times: xr.Dataset
            Times between wich to extract or resample the data with dims 
        data: nd.array
            numpy array from the erp_data functino
        color: str
            color for the lines
        ax: matplotlib.pyplot
            ax on which to draw
        minmax_lines: tuple
            Min and max arguments for the vertical lines on the plot
        upsample: float
            Upsampling factor for the times
        bootstrap: int
            how many bootstrap draw to perform
    '''
    if ax is None:
        ax = plt
    for event in times.event[1:]:
        time = times.sel(event=event-1).mean()
        time_current = int(times.sel(event=event).mean())
        if len(times.event)>2:
            x = np.linspace(time,time_current,num=np.shape(data)[-1])*upsample
            mean_signal =  np.nanmean(data[:,event,:],axis=0)
        else:
            x = np.arange(time,time_current)*upsample
            mean_signal =  np.nanmean(data[:,event,:],axis=0)[:time_current]
        ax.plot(x, mean_signal, color=color, label=label)
        if minmax_lines is not None:
            ax.vlines(times.mean('trial_x_participant')*upsample, minmax_lines[0],minmax_lines[1], color=color, ls=':', alpha=.25)
        if bootstrap is not None:
            test_boot = stats.bootstrap((data[:,event,:],), statistic=np.nanmean, n_resamples=bootstrap, axis=0, batch=10)
            if len(times.event)>2:
                ax.fill_between(x, test_boot.confidence_interval[0], test_boot.confidence_interval[1], alpha=0.5, color=color)
            else:
                ax.fill_between(x, test_boot.confidence_interval[0][:time_current], test_boot.confidence_interval[1][:time_current], alpha=0.5, color=color)
