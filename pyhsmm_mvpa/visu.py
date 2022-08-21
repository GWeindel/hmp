'''

'''

import matplotlib.pyplot as plt
import numpy as np

def plot_topo_timecourse(electrodes, times, channel_position, time_step=1, bump_size=50,
                        figsize=None, magnify=1, mean_rt=None, cmap='Spectral_r',
                        ylabels=[], max_time = None, vmin=None, vmax=None, title=False):
    from mne.viz import plot_topomap
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    if not figsize:
        figzise = (12, 2)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=200)
    bump_size = bump_size*time_step*magnify
    yoffset =.25*magnify
    axes = []

    n_iter = np.shape(electrodes)[0]
    if n_iter == 1:
        times = [times]
    times = np.array(times)
    for iteration in np.arange(n_iter):
        times_iteration = times[iteration]*time_step
        electrodes_ = electrodes[iteration,:,:]
        n_bump = int(sum(np.isfinite(electrodes_[:,0])))
        electrodes_ = electrodes_[:n_bump,:]
        for bump in np.arange(n_bump):
            axes.append(ax.inset_axes([times_iteration[bump],iteration-yoffset,
                                       bump_size,yoffset*2], transform=ax.transData))
            plot_topomap(electrodes_[bump,:], channel_position, axes=axes[-1], show=False,
                         cmap=cmap, vmin=vmin, vmax=vmax)
    if isinstance(ylabels, dict):
        ax.set_yticks(np.arange(len(list(ylabels.values())[0])),
                      [str(x) for x in list(ylabels.values())[0]])
        ax.set_ylabel(str(list(ylabels.keys())[0]))
    else:
        ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0-yoffset, n_iter-1+yoffset)
    if isinstance(mean_rt, (np.ndarray, np.generic)):
        if isinstance(mean_rt, np.ndarray):
            ax.vlines(mean_rt*time_step,0-yoffset, np.arange(len(mean_rt))+yoffset, ls='--')
            ax.set_xlim(0, max(mean_rt)*time_step+((max(mean_rt)*time_step)/15))
        else:
            ax.vlines(mean_rt*time_step,0-yoffset, n_iter-1+yoffset, ls='--')
            ax.set_xlim(0, mean_rt*time_step+(mean_rt*time_step)/15)
    if max_time:
        ax.set_xlim(0, max_time)
    if not max_time and not isinstance(mean_rt, (np.ndarray, np.generic)):
        ax.set_xlim(0, np.nanmax(times.flatten()))
    if time_step == 1:
        ax.set_xlabel('Time (in samples)')
    else:
        ax.set_xlabel('Time')
    if title:
        ax.set_title(title)
    plt.show()    


def plot_LOOCV(loocv_estimates, pvals=True, test='t-test', figsize=(16,5), indiv=True):
    if pvals:
        if test == 'sign':
            from statsmodels.stats.descriptivestats import sign_test 
        elif test == 't-test':
            from scipy.stats import ttest_1samp
        else:
            print('Expected sign or t-test arguments')
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

def plot_latencies(fits, labels, bump_width):
    from itertools import cycle
    f, axs = plt.subplots(1,1, figsize=(6, 4),sharey=True, sharex=False,dpi=100)
    j = 0
    if not isinstance(fits, list) and 'n_bumps' not in fits:
        fits = [fits]
    for fit in fits:
        cycol = cycle('bgrcm')
        n_stages = len(fit.parameters.isel(params=0).dropna('stage'))
        colors = [next(cycol) for x in np.arange(n_stages)]
        stages = fit.parameters.isel(params=0).dropna('stage') * \
                  fit.parameters.isel(params=1).dropna('stage') + \
                  np.concatenate([[0],np.repeat(bump_width,n_stages-1)])
        for stage in np.arange(len(stages)-1,0,-1):
            colors.append(next(cycol))
            plt.barh(j, stages[:stage].sum(), color=colors[stage-1], edgecolor='k')
        j += 1

    plt.yticks(np.arange(j),labels)
    plt.ylim(0-1,j)
    plt.xlabel('(Cumulative) Stages durations from stimulus onset (samples)')
    plt.tight_layout()
    # Hide the right and top spines
    axs.spines.right.set_visible(False)
    axs.spines.top.set_visible(False)

    # Only show ticks on the left and bottom spines
    axs.yaxis.set_ticks_position('left')
    axs.xaxis.set_ticks_position('bottom')
    plt.show()
    
def plot_latencies_gammas(fits, labels, bump_width):
    from itertools import cycle
    f, axs = plt.subplots(1,1, figsize=(6, 4),sharey=True, sharex=False,dpi=100)
    j = 0
    if not isinstance(fits, list) and 'n_bumps' not in fits:
        fits = [fits]
    if 'n_bumps' in fits:
        xrfits = fits.copy(deep=True)
        fits = []
        for n_bump in xrfits.n_bumps:
            fits.append(xrfits.sel(n_bumps=n_bump))
    for fit in fits:
        cycol = cycle('bgrcm')
        n_stages = len(fit.parameters.isel(params=0).dropna('stage'))
        colors = [next(cycol) for x in np.arange(n_stages)]
        stages = fit.parameters.isel(params=0).dropna('stage') * \
                  fit.parameters.isel(params=1).dropna('stage') + \
                  np.concatenate([[0],np.repeat(bump_width,n_stages-1)])
        for stage in np.arange(len(stages)-1,0,-1):
            colors.append(next(cycol))
            plt.barh(j, stages[:stage].sum(), color=colors[stage-1], edgecolor='k')
        j += 1

    plt.yticks(np.arange(j),labels)
    plt.ylim(0-1,j)
    plt.xlabel('(Cumulative) Stages durations from stimulus onset (samples)')
    plt.tight_layout()
    # Hide the right and top spines
    axs.spines.right.set_visible(False)
    axs.spines.top.set_visible(False)

    # Only show ticks on the left and bottom spines
    axs.yaxis.set_ticks_position('left')
    axs.xaxis.set_ticks_position('bottom')
    plt.show()
