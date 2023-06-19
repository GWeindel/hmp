'''

'''

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from itertools import cycle
default_colors =  ['cornflowerblue','indianred','orange','darkblue','darkgreen','gold']

def plot_topo_timecourse(channels, estimated, channel_position, init, time_step=1, ydim=None,
                figsize=None, dpi=100, magnify=1, times_to_display=None, cmap='Spectral_r',
                ylabels=[], max_time = None, vmin=None, vmax=None, title=False, ax=None, 
                sensors=False, skip_channels_computation=False):
    '''
    Plotting the event topologies at the average time of the end of the previous stage.
    
    Parameters
    ----------
    channels : ndarray | xr.Dataarray 
        a 2D or 3D matrix of channel activity with channels and event as dimension (+ eventually a varying dimension) OR
        the original EEG data in HMP format
    estimated : ndarray
        a 1D or 2D matrix of times with event as dimension OR directly the results from a fitted hmp 
    channel_position : ndarray
        Either a 2D array with dimension channel and [x,y] storing channel location in meters or an info object from
        the mne package containning digit. points for channel location
    init : float
        initialized HMP object
    time_step : float
        What unit to multiply all the times with, if you want to go on the second or millisecond scale you can provide 
        1/sf or 1000/sf where sf is the sampling frequency of the data
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
    ylabels : dict
        dictonary with {label_name : label_values}
    max_time : float
        limit of the x (time) axe
    vmin : float
        Colormap limits to use (see https://mne.tools/dev/generated/mne.viz.plot_topomap.html)
    vmax : float
        Colormap limits to use (see https://mne.tools/dev/generated/mne.viz.plot_topomap.html)
    title : str
        title of the plot
    ax : matplotlib.pyplot.ax
        Matplotlib object on which to draw the plot, can be useful if you want to control specific aspects of the plots
        outside of this function
    sensors : bool
        Whether to plot the sensors on the topologies
        
    Returns
    -------
    ax : matplotlib.pyplot.ax
        if ax was specified otherwise returns the plot
    '''
    
    from mne.viz import plot_topomap
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    return_ax = True
    if times_to_display is None:
        times_to_display = init.mean_d*time_step
    if isinstance(estimated, (xr.DataArray, xr.Dataset)) and 'event' in estimated:
        if ydim is None and 'n_events' in estimated.dims:
            if estimated.n_events.count() > 1:
                ydim = 'n_events'
        if ydim is not None and not skip_channels_computation:
            channels = init.compute_topologies(channels, estimated, init.event_width_samples, ydim).data
        elif not skip_channels_computation:
            channels = init.compute_topologies(channels, estimated, init.event_width_samples).data
        channels[channels == 0] = np.nan
        times = init.compute_times(init, estimated, mean=True).data
    else:#assumes times already computed
        times = estimated
    if len(np.shape(channels)) == 2:
        channels = channels[np.newaxis]
    n_iter = np.shape(channels)[0]
    if ax is None:
        if figsize == None:
            figsize = (12, 1*n_iter)
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        return_ax = False
    event_size = init.event_width_samples*time_step*magnify
    yoffset =.25*magnify
    axes = []
    if n_iter == 1:
        times = [times]
    times = np.array(times, dtype=object)
    for iteration in np.arange(n_iter):
        times_iteration = times[iteration]*time_step
        channels_ = channels[iteration,:,:]
        n_event = int(sum(np.isfinite(channels_[:,0])))
        channels_ = channels_[:n_event,:]
        for event in np.arange(n_event):
            axes.append(ax.inset_axes([times_iteration[event],iteration-yoffset,
                                (event_size),yoffset*2], transform=ax.transData))
            plot_topomap(channels_[event,:], channel_position, axes=axes[-1], show=False,
                         cmap=cmap, vlim=(vmin, vmax), sensors=sensors)
    if isinstance(ylabels, dict):
        ax.set_yticks(np.arange(len(list(ylabels.values())[0])),
                      [str(x) for x in list(ylabels.values())[0]])
        ax.set_ylabel(str(list(ylabels.keys())[0]))
    else:
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
    __display_times(ax, times_to_display, 0, time_step, max_time, times, n_iter)
    if not return_ax:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0-yoffset, n_iter-1+yoffset)
        if time_step == 1:
            ax.set_xlabel('Time (in samples)')
        else:
            ax.set_xlabel('Time')
        if title:
            ax.set_title(title)
        if np.any(max_time) == None and np.any(times_to_display) == None:
            ax.set_xlim(0, np.nanmax(times)+np.nanmax(times)/10)
    if return_ax:
        ax.set_ylim(0-yoffset, n_iter-1+yoffset)
        return ax
    else:
        plt.show()    


def plot_loocv(loocv_estimates, pvals=True, test='t-test', figsize=(16,5), indiv=True, ax=None, mean=False):
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

    Returns
    -------
    ax : matplotlib.pyplot.ax
        if ax was specified otherwise returns the plot
    '''
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
    if mean:
        alpha = .2#for the indiv plot
        means = np.nanmean(loocv_estimates.data,axis=1)
        ax[0].errorbar(x=np.arange(len(means))+1, y=means, \
                 yerr= np.nanstd(loocv_estimates.data,axis=1)/np.sqrt(len(loocv_estimates.participants))*1.96, marker='o', color='k')
    else:
        alpha=1
    if indiv:
        for loo in loocv_estimates.T:
            ax[0].plot(loocv_estimates.n_event,loo, alpha=alpha)
    ax[0].set_ylabel('LOOCV Loglikelihood')
    ax[0].set_xlabel('Number of events')
    ax[0].set_xticks(ticks=loocv_estimates.n_event)
    total_sub = len(loocv_estimates.participants)
    diffs, diff_bin, labels = [],[],[]
    for n_event in np.arange(2,loocv_estimates.n_event.max()+1):
        diffs.append(loocv_estimates.sel(n_event=n_event).data - loocv_estimates.sel(n_event=n_event-1).data)
        diff_bin.append([1 for x in diffs[-1] if x > 0])
        labels.append(str(n_event-1)+'->'+str(n_event))
        if pvals:
            pvalues = []
            if test == 'sign':
                diff_tmp = np.array(diffs)
                diff_tmp[np.isnan(diff_tmp)] = -np.inf 
                pvalues.append((sign_test(diff_tmp[-1])))
            elif test == 't-test':
                pvalues.append((ttest_1samp(diffs[-1], 0, alternative='greater')))
            mean = np.nanmean(loocv_estimates.sel(n_event=n_event).data)
            ax[1].text(x=n_event-2, y=0, s=str(int(np.nansum(diff_bin[-1])))+'/'+str(len(diffs[-1]))+':'+str(np.around(pvalues[-1][-1],3)))
    ax[1].plot(diffs,'.-', alpha=.3)
    ax[1].set_xticks(ticks=np.arange(0,loocv_estimates.n_event.max()-1), labels=labels)
    ax[1].hlines(0,0,len(np.arange(2,loocv_estimates.n_event.max())),color='k')
    ax[1].set_ylabel('Change in likelihood')
    ax[1].set_xlabel('')
    if return_ax:
        return ax
    else:
        plt.tight_layout()
        plt.show()

def plot_latencies_average(times, event_width, time_step=1, labels=[], colors=default_colors,
    figsize=None, errs='ci', max_time=None, times_to_display=None):
    '''
    REDUNDANT WITH plot_latencies() WILL BE DEPRECATED
    Plots the average of stage latencies with choosen errors bars

    Parameters
    ----------
    times : ndarray
        2D or 3D numpy array, Either trials * events or conditions * trials * events
    event_width : float
        Display size of the event in time unit given sampling frequency, if drawing a fitted object using hsmm_mvpy you 
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
        Whether to display 95% confidence interval ('ci') or standard deviation (std)
    times_to_display : ndarray
        Times to display (e.g. Reaction time or any other relevant time) in the time unit of the fitted data
    max_time : float
        limit of the x (time) axe
    '''
    from seaborn.algorithms import bootstrap #might be too much to ask for seaborn install?
    j = 0
    
    if len(np.shape(times)) == 2:
        times = [times]

    if figsize == None:
        figsize = (8, 1*len(times)+2)
    f, axs = plt.subplots(1,1, figsize=figsize,dpi=100)
    for time in times:
        time = time*time_step
        cycol = cycle(colors)
        n_stages = len(time[-1][np.isfinite(time[-1])])
        colors = [next(cycol) for x in np.arange(n_stages)]
        for stage in np.arange(n_stages-1,-1,-1):
            colors.append(next(cycol))
            plt.barh(j+.02*stage, np.mean(time[:,stage]), color='w', edgecolor=colors[stage])
            if errs == 'ci':
                errorbars = np.transpose([np.nanpercentile(bootstrap(time[:,stage]), q=[2.5,97.5])])
                errorbars = np.abs(errorbars-np.mean(time[:,stage].values))
            elif errs == 'std':
                errorbars = np.std(time[:,stage])
            else:
                print('Unknown errorbar type')
                errorbars = np.repeat(0,2)
            plt.errorbar(np.mean(time[:,stage]), j+.02*stage, xerr=errorbars, 
                     color=colors[stage], fmt='none', capsize=10)
        j += 1
    plt.yticks(np.arange(len(labels)),labels)
    plt.ylim(0-1,j)
    __display_times(axs, times_to_display, np.arange(np.shape(times)[0]), time_step, max_time, times)
    if time_step == 1:
        plt.xlabel('(Cumulative) Stages durations from stimulus onset (samples)')
    else:
        plt.xlabel('(Cumulative) Stages durations from stimulus onset')
    plt.tight_layout()
    # Hide the right and top spines
    axs.spines.right.set_visible(False)
    axs.spines.top.set_visible(False)
    # Only show ticks on the left and bottom spines
    axs.yaxis.set_ticks_position('left')
    axs.xaxis.set_ticks_position('bottom')
    return axs
    

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

def __display_times(ax, times_to_display, yoffset, time_step, max_time, times, ymax=1):
    n_iter = len(times)
    times = np.asarray(times,dtype=object)
    if isinstance(times_to_display, (np.ndarray, np.generic)):
        if isinstance(times_to_display, np.ndarray):
            ax.vlines(times_to_display*time_step, yoffset-1.1, yoffset+ymax+1.1, ls='--')
            ax.set_xlim(-1*time_step, max(times_to_display)*time_step+((max(times_to_display)*time_step)/15))
        else:
            ax.vlines(times_to_display*time_step, yoffset-1.1, yoffset+ymax+1.1, ls='--')
            ax.set_xlim(-1*time_step, times_to_display*time_step+(times_to_display*time_step)/15)
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
        Size of the event in time unit given sampling frequency, if drawing a fitted object using hsmm_mvpy you 
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

def plot_latencies(times, event_width, time_step=1, labels=[], colors=default_colors,
    figsize=False, errs='ci',  max_time=None, times_to_display=None, kind='bar', legend=False):
    '''
    Plots the average of stage latencies with choosen errors bars

    Parameters
    ----------
    times : ndarray
        2D or 3D numpy array, Either trials * events or conditions * trials * events
    event_width : float
        Display size of the event in time unit given sampling frequency, if drawing a fitted object using hsmm_mvpy you 
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
        Whether to display 95% confidence interval ('ci') or standard deviation (std)
    times_to_display : ndarray
        Times to display (e.g. Reaction time or any other relevant time) in the time unit of the fitted data
    max_time : float
        limit of the x (time) axe
    '''
    from seaborn.algorithms import bootstrap #might be too much to ask for seaborn install?
    j = 0
    
    if len(np.shape(times)) == 2:
        times = [times]

    if not figsize:
        figsize = (8, 1*len(times)+2)
    f, axs = plt.subplots(1,1, figsize=figsize, dpi=100)
    cycol = cycle(colors)
    colors = [next(cycol) for x in np.arange(len(times))]
    for time in times:
        time = time*time_step
        time = np.diff(time, axis=1, prepend=0)
        n_stages = len(time[-1])
        if errs == 'ci':
            errorbars = np.nanpercentile(bootstrap(time, axis=0), q=[2.5,97.5], axis=0)
            errorbars = np.abs(errorbars-np.mean(time, axis=0))
        elif errs == 'std':
            errorbars = np.std(time, axis=0)
        else:
            print('Unknown errorbar type')
            errorbars = np.repeat(0,2)
        if kind == 'bar':
            plt.bar(np.arange(n_stages)+1, np.mean(time, axis=0), color='w', edgecolor=colors[j])
            plt.errorbar(np.arange(n_stages)+1, np.mean(time, axis=0), yerr=errorbars, 
                     color=colors[j], fmt='none', capsize=10)
        elif kind == 'point':
            plt.errorbar(np.arange(n_stages)+1, np.mean(time, axis=0), 
                yerr=errorbars, color=colors[j], fmt='o-', capsize=10, label=labels[j])
        j += 1

    plt.xlim(1-.5, n_stages+.5)
    if time_step == 1:
        plt.ylabel('Stage durations (samples)')
    else:
        plt.ylabel('Stage durations')
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
    from hsmm_mvpy.models import hmp
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
            threshold=0, verbose=False)
        #Visualizing
        plot_topo_timecourse(eeg_data, selected, positions,  init, magnify=1, sensors=False,ax=ax)
    

def plot_bootstrap_results(bootstrapped, info, init, model_to_compare=None, epoch_data=None):
    from hsmm_mvpy.resample import percent_event_occurence
    maxboot_model, labels, counts, event_number, label_event_num = percent_event_occurence(bootstrapped, model_to_compare)
    fig, axes = plt.subplot_mosaic([['a', 'a'], ['b', 'c'], ['b', 'c']],
                              layout='constrained')
    if model_to_compare is None: 
        plot_topo_timecourse(maxboot_model.channels_activity.values, maxboot_model.event_times.values, info, init,ax=axes['a'])
        times = maxboot_model.event_times#init.compute_times(init, maxboot_model, mean=True)#computing predicted event times
    else:
        plot_topo_timecourse(epoch_data, model_to_compare, info, init,ax=axes['a'])
        times = init.compute_times(init, model_to_compare, mean=True)
        maxboot_model = model_to_compare
    axes['a'].set_xlabel('Time (samples)')
    axes['b'].bar(maxboot_model.event+1,counts)
    axes['b'].set_xlabel('Event number')
    axes['b'].set_xticks(maxboot_model.event+1)
    axes['b'].set_ylabel('Frequency')
    axes['b'].set_ylim(0,1)
    
    axes['c'].bar(label_event_num,event_number)
    axes['c'].set_xlabel('Number of events')
    
    axes['c'].set_ylabel('Frequency')
    axes['c'].set_ylim(0,1)
    axes['a'].spines[['right', 'top']].set_visible(False)
    axes['b'].spines[['right', 'top']].set_visible(False)
    axes['c'].spines[['right', 'top']].set_visible(False)
    # plt.tight_layout()
    plt.show()
    
                             