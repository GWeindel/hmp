'''

'''

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
default_colors =  ['cornflowerblue','indianred','orange','darkblue','darkgreen','gold']

def plot_topo_timecourse(electrodes, times, channel_position, time_step=1, bump_size=50,
                        figsize=None, dpi=100, magnify=1, times_to_display=None, cmap='Spectral_r',
                        ylabels=[], max_time = None, vmin=None, vmax=None, title=False, ax=False, sensors=True):
    from mne.viz import plot_topomap
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    if isinstance(ax, bool):
        if not figsize:
            figzise = (12, 2)
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    bump_size = bump_size*time_step*magnify
    yoffset =.25*magnify
    axes = []

    n_iter = np.shape(electrodes)[0]
    if n_iter == 1:
        times = [times]
    times = np.array(times, dtype=object)
    for iteration in np.arange(n_iter):
        times_iteration = times[iteration]*time_step
        electrodes_ = electrodes[iteration,:,:]
        n_bump = int(sum(np.isfinite(electrodes_[:,0])))
        electrodes_ = electrodes_[:n_bump,:]
        for bump in np.arange(n_bump):
            axes.append(ax.inset_axes([times_iteration[bump],iteration-yoffset,
                                       bump_size/2,yoffset*2], transform=ax.transData))
            plot_topomap(electrodes_[bump,:], channel_position, axes=axes[-1], show=False,
                         cmap=cmap, vmin=vmin, vmax=vmax, sensors=sensors)
    if isinstance(ylabels, dict):
        ax.set_yticks(np.arange(len(list(ylabels.values())[0])),
                      [str(x) for x in list(ylabels.values())[0]])
        ax.set_ylabel(str(list(ylabels.keys())[0]))
    else:
        ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0-yoffset, n_iter-1+yoffset)
    __display_times(ax, times_to_display, 0, time_step, max_time, times)
    if time_step == 1:
        ax.set_xlabel('Time (in samples)')
    else:
        ax.set_xlabel('Time')
    if title:
        ax.set_title(title)
    plt.show()    


def plot_LOOCV(loocv_estimates, pvals=True, test='t-test', figsize=(16,5), indiv=True, ax=False):
    if pvals:
        #if test == 'sign':
        #    from statsmodels.stats.descriptivestats import sign_test 
        if test == 't-test':
            from scipy.stats import ttest_1samp
        else:
            print('Expected  t-test argument')
    if isinstance(ax, bool):
        fig, ax = plt.subplots(1,2, figsize=figsize)
    ax[0].errorbar(x=np.arange(loocv_estimates.n_bump.max())+1,y=np.mean(loocv_estimates.data,axis=1),yerr=np.std(loocv_estimates.data,axis=1)/np.sqrt(len(loocv_estimates.participants))*1.96,marker='o')
    if indiv:
        for loo in loocv_estimates.T:
            ax[0].plot(np.arange(loocv_estimates.n_bump.max())+1,loo, alpha=.2)
    ax[0].set_ylabel('LOOCV Loglikelihood')
    ax[0].set_xlabel('Number of bumps')
    ax[0].set_xticks(ticks=np.arange(1,loocv_estimates.n_bump.max()+1))

    diffs, diff_bin, labels = [],[],[]
    for n_bump in np.arange(2,loocv_estimates.n_bump.max()+1):
        diffs.append(loocv_estimates.sel(n_bump=n_bump).data - loocv_estimates.sel(n_bump=n_bump-1).data)
        diff_bin.append([1 for x in diffs[-1] if x > 0])
        labels.append(str(n_bump-1)+'->'+str(n_bump))
        if pvals:
            pvalues = []
            if test == 'sign':
                pvalues.append((sign_test(diffs[-1])))
            elif test == 't-test':
                pvalues.append((ttest_1samp(diffs[-1], 0, alternative='greater')))
            mean = np.mean(loocv_estimates.sel(n_bump=n_bump).data)
            ax[0].text(x=n_bump-.5, y=mean+mean/10, s=str(np.sum(diff_bin[-1]))+'/'+str(len(diffs[-1]))+':'+str(np.around(pvalues[-1][-1],2)))
    ax[1].plot(diffs,'.-', alpha=.3)
    ax[1].set_xticks(ticks=np.arange(0,loocv_estimates.n_bump.max()-1), labels=labels)
    ax[1].hlines(0,0,len(np.arange(2,loocv_estimates.n_bump.max())),color='k')
    ax[1].set_ylabel('Change in likelihood')
    ax[1].set_xlabel('')
    plt.tight_layout()
    plt.show()

def plot_latencies_average(times, bump_width, time_step=1, labels=[], colors=default_colors,
    figsize=False, errs='ci', yoffset=0, max_time=None, times_to_display=None):
    '''


    Parameters
    ----------
    times : ndarray
        2D or 3D numpy array, Either trials * bumps or conditions * trials * bumps
        
        
    '''
    from seaborn.algorithms import bootstrap #might be too much to ask for seaborn install?
    j = 0
    
    if len(np.shape(times)) == 2:
        times = [times]

    if not figsize:
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
                errorbars = np.abs(errorbars-np.mean(time[:,stage]))
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
    f, axs = plt.subplots(1,1, figsize=figsize, dpi=100)
    '''


    Parameters
    ----------
    times : ndarray
        2D or 3D numpy array, Either trials * bumps or conditions * trials * bumps
        
        
    '''
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

def __display_times(ax, times_to_display, yoffset, time_step, max_time, times):
    n_iter = len(times)
    times = np.asarray(times,dtype=object)
    if isinstance(times_to_display, (np.ndarray, np.generic)):
        if isinstance(times_to_display, np.ndarray):
            ax.vlines(times_to_display*time_step, yoffset-.5, yoffset+1-.5, ls='--')
            ax.set_xlim(-1*time_step, max(times_to_display)*time_step+((max(times_to_display)*time_step)/15))
        else:
            ax.vlines(times_to_display*time_step, yoffset-.5, yoffset+1-.5, ls='--')
            ax.set_xlim(-1*time_step, times_to_display*time_step+(times_to_display*time_step)/15)
    if max_time:
        ax.set_xlim(-1*time_step, max_time)
    return ax

def plot_latencies_gamma(gammas, bump_width=0, time_step=1, labels=[''], colors=default_colors, 
                         figsize=False, yoffset=0, max_time=None, times_to_display=None, kind='bar', legend=False):
    '''


    Parameters
    ----------
    gammas : ndarray
        2D or 3D numpy array, Either  bumps * parameters or conditions * bumps * parameters with parameters being [shape * scale]
        
        
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
    #__display_times(axs, mean_rt, np.arange(np.shape(gammas)[0]), time_step, max_time, gammas)
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

def plot_latencies(times, bump_width, time_step=1, labels=[], colors=default_colors,
    figsize=False, errs='ci',  max_time=None, times_to_display=None, kind='bar', legend=False):
    '''


    Parameters
    ----------
    times : ndarray
        2D or 3D numpy array, Either trials * bumps or conditions * trials * bumps
        
        
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