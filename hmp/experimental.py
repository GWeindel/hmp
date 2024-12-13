'''
Clustering
resampling
'''

from mne.viz import plot_topomap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import multiprocessing as mp
import itertools

cols = np.array(['tab:blue',
'tab:orange',
'tab:green',
'tab:red',
'tab:purple',
'tab:brown',
'tab:pink',
'tab:gray',
'tab:olive',
'tab:cyan'])

def mahalanobis(x=None, data=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    """
    if x.shape[0] == 1:
        return np.array([0])
    else:
        x_minus_mu = x - np.mean(data, axis=0)
        VI = sp.linalg.inv(np.cov(data.T))
        left_term = np.dot(x_minus_mu, VI)
        mahal = np.dot(left_term, x_minus_mu.T)
        return np.sqrt(np.diag(mahal))

def cluster_events(init, lkhs, mags, channels, times, method='time_x_lkh_mags', max_clust=5, p_outlier=.01, info=None, nr_clust=None, calc_outliers=False, alpha=.05):
    #method = 'time_x_lkh' for clustering based on time and likelihood, or 'time_x_lkh_x_mags' for
    #         clustering based on time, likelihood and mags
    
    #zscore time and lkh
    times_scaled = (times - np.mean(times)) / np.std(times)
    lkhs_scaled = (lkhs - np.mean(lkhs)) / np.std(lkhs)

    #features
    feat = np.vstack((times_scaled,lkhs_scaled)).T
    if method == 'time_x_lkh_x_mags':
        feat = np.hstack((feat,mags))
    
    #calculate clusters up to max_clust
    kmeans_kwargs = {
        "init": "random",
        "n_init": 20,
        "max_iter": 400}

    kmeans_sols = []
    for k in range(1, max_clust+1):
        kmeans_sols.append(KMeans(n_clusters=k, **kmeans_kwargs))
        kmeans_sols[-1].fit(feat)
    
    #if nr_clust not given, determine number of clusters based on silhouette coefficient
    if not nr_clust:
        silhouette_coefficients = []

        #start at 2 clusters for silhouette coefficient
        if init.cpus > 1:
            with mp.Pool(processes=init.cpus) as pool:
                silhouette_coefficients = pool.starmap(silhouette_score, 
                    zip(itertools.repeat(feat), [x.labels_ for x in kmeans_sols[1:max_clust]]))
        else:
            for k in range(2, max_clust+1):
                silhouette_coefficients.append(silhouette_score(feat, kmeans_sols[k-1].labels_))

        #plot coefficients
        _, ax = plt.subplots(1,1,figsize=(5,5))
        plt.plot(range(2, max_clust+1), silhouette_coefficients)
        plt.xticks(range(2, max_clust+1))
        plt.xlabel("Number of Clusters")    
        plt.ylabel("Silhouette Coefficient")
        plt.title("Silhouette Coefficients")
        plt.show()
        plt.pause(.1)

        #best coefficient / select fit:
        try_n = np.argmax(silhouette_coefficients) + 2 #starts at 2
    else:
        try_n = nr_clust

    #explore solutions until user inputs 0, starting with proposed solution 
    while try_n > 0:

        #get solution
        kmeans = kmeans_sols[try_n-1]
    
        #determine outliers given fit
        if calc_outliers:
            maha_distances = np.zeros((kmeans.labels_.shape[0],2))
            for cl in range(kmeans.n_clusters):
                maha_distances[kmeans.labels_ == cl,0] = mahalanobis(feat[kmeans.labels_ == cl,:], feat[kmeans.labels_ == cl,:])
                maha_distances[kmeans.labels_ == cl,1] = 1 - sp.stats.chi2.cdf(maha_distances[kmeans.labels_ == cl,0], feat.shape[1]-1)                #add p-values

        #calc channels per cluster (needs to be done in two steps to get vmin/max)
        channels_cluster = np.zeros((kmeans.n_clusters, channels.shape[1]))
        for cl in range(kmeans.n_clusters):
            outliers = maha_distances[kmeans.labels_ == cl,1] < p_outlier if calc_outliers else np.repeat(False, np.sum(kmeans.labels_ == cl))
            channels_cluster[cl,:] = np.median(channels[kmeans.labels_ == cl,:][~outliers,:],axis=0)
            
        vmax = np.nanmax(np.abs(channels_cluster[:])) if np.nanmax(channels_cluster[:]) >= 0 else 0
        vmin = -np.nanmax(np.abs(channels_cluster[:])) if np.nanmin(channels_cluster[:]) < 0 else 0

        #plot results
        axes=[]
        yrange = (np.max(lkhs) - np.min(lkhs))
        topo_size = .6 * (np.max(times) - np.min(times)) #6% of time scale
        topo_size_y = .4 * yrange

        _, ax = plt.subplots(1,1,figsize=(20,3))
        time_step=1000/init.sfreq
        for cl in range(kmeans.n_clusters):
            outliers = maha_distances[kmeans.labels_ == cl,1] < p_outlier if calc_outliers else np.repeat(False, np.sum(kmeans.labels_ == cl))
            ax.plot(times[kmeans.labels_ == cl][outliers]*time_step, lkhs[kmeans.labels_ == cl][outliers], 'x', color=cols[cl],alpha=alpha)
            ax.plot(times[kmeans.labels_ == cl][~outliers]*time_step, lkhs[kmeans.labels_ == cl][~outliers], '.', color=cols[cl],alpha=alpha)

            #topo
            if info:
                axes.append(ax.inset_axes([(np.median(times[kmeans.labels_ == cl][~outliers]) - topo_size / 2)* time_step, np.max(lkhs[kmeans.labels_ == cl][~outliers]) + .1 * yrange, topo_size * time_step, topo_size_y], transform=ax.transData))
                plot_topomap(channels_cluster[cl,:], info, axes=axes[-1], show=False,
                                    cmap='Spectral_r', vlim=(vmin, vmax), sensors=False, contours=False)

        bottom,_ = ax.get_ylim()
        ax.set_xlim(0, init.mean_d*time_step)
        ax.set_ylim((bottom, np.max(lkhs) + yrange*.5))
        
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        ax.set_xlabel('Time (ms)', fontsize=18)
        ax.set_ylabel('Likelihood', fontsize=18)
        if plt.get_backend()[0:2] == 'Qt': #fixes issue with yscaling
            plt.tight_layout()
        plt.pause(.1)

        try_n = int(input('Do you agree with this solution [enter \'0\'], or would you like to explore a different number of clusters [enter the number of clusters]?'))

    #get cluster info
    best_n_clust = kmeans.n_clusters
    cl_times = np.zeros((best_n_clust,)) #cluster times (not necessarily sorted)
    cl_mags = np.zeros((best_n_clust, mags.shape[1])) #cluster mags
    for cl in range(best_n_clust):
        outliers = maha_distances[kmeans.labels_ == cl,1] < p_outlier if calc_outliers else np.repeat(False, np.sum(kmeans.labels_ == cl))
        cl_times[cl] = np.median(times[kmeans.labels_ == cl][~outliers])
        cl_mags[cl,:] = np.median(mags[kmeans.labels_ == cl,:][~outliers,:], axis=0)

    #calc mags and params based on nclust, get max likelihood in each
    mags = cl_mags[np.argsort(cl_times), :] #mags are easy, just sort by time
    cl_times = np.sort(cl_times) #sort
    cl_durations = np.hstack((cl_times, init.mean_d)) - np.hstack((0, cl_times)) #get stage durations
    pars = np.array([np.repeat(init.shape, best_n_clust + 1), init.mean_to_scale((cl_durations - np.hstack((0, np.repeat(init.location,best_n_clust)))), init.shape)]).T #calc params, take into account locations 
    null_stages = np.where(pars[:,1] < 0)
    pars[null_stages, 1] = 1/init.shape #avoids impossible parameters
    return mags, pars

import numpy as np
import xarray as xr
import multiprocessing as mp
import warnings
from warnings import warn, filterwarnings
from hmp.models import hmp as classhmp
from hmp.utils import transform_data, stack_data, save_eventprobs
from hmp.visu import plot_topo_timecourse

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm


def _gen_idx(data, dim, iterations=1):
    orig_index = data[dim].values
    indexes = np.array([np.random.choice(orig_index, size=len(orig_index), replace=True) for x in range(iterations)])
    sort_idx = orig_index.argsort()
    corresponding_indexes = sort_idx[np.searchsorted(orig_index,indexes[0],sorter = sort_idx)]
    return  indexes[0]#corresponding_indexes,

def _gen_dataset(data, dim, n_iterations):   
    try: 
        original_dim_order = list(data.dims.keys())
        original_dim_order_data = list(data.data.dims)
    except: #Data Array
        original_dim_order = list(data.dims)
        original_dim_order_data = list(data.dims)
    if isinstance(dim, list):
        if len(dim) == 2:
            data = data.stack({str(dim[0])+'_x_'+str(dim[1]):dim}).dropna(str(dim[0])+'_x_'+str(dim[1]), how='all')
            dim = str(dim[0])+'_x_'+str(dim[1])
        elif len(dim) > 2:
            raise ValueError('Cannot stack more than two dimensions')
        else: dim = dim[0]
    named_index = []
    for iteration in range(n_iterations):
        named_index.append(_gen_idx(data, dim))
    return named_index, data, dim, original_dim_order_data

def _bootstrapped_run(fit, data, dim, indexes, order, init, n_iter, use_starting_points, rerun_pca, pca_weights, summarize, verbose, cpus, trace, path):
    sfreq = init.sfreq
    print(true)
    resampled_data = data.loc[{dim:list(indexes)}].unstack().transpose(*order)
    if '_x_' in dim:
        dim = dim.split('_x_')
    if isinstance(dim, list):
        resampled_data = resampled_data.assign_coords({dim[0]: np.arange(len(resampled_data[dim[0]]))})#Removes indices to avoid duplicates
        resampled_data = resampled_data.assign_coords({dim[1]: np.arange(len(resampled_data[dim[1]]))})
    else:
        resampled_data = resampled_data.assign_coords({dim: np.arange(len(resampled_data[dim]))})
    if 'channels' in resampled_data.dims:
        if rerun_pca:
            hmp_data_boot = transform_data(resampled_data, n_comp=init.n_dims)
        else:
            hmp_data_boot = transform_data(resampled_data, pca_weights=pca_weights)
    else:
        hmp_data_boot = stack_data(resampled_data)
    if use_starting_points:
        init_boot = classhmp(hmp_data_boot, sfreq=sfreq, event_width=init.event_width, cpus=1,
                        shape=init.shape, estimate_magnitudes=init.estimate_magnitudes, 
                        estimate_parameters=init.estimate_parameters, template=init.template,
                        location=init.location, distribution=init.distribution)
    else:
        init_boot = classhmp(hmp_data_boot, sfreq=sfreq, event_width=init.event_width, cpus=1,
                        shape=init.shape, template=init.template,
                        location=init.location, distribution=init.distribution)
    print(true)
    estimates_boot = init_boot.fit_single(fit.magnitudes.sizes['event'], verbose=verbose, parameters=fit.parameters,
                                         magnitudes=fit.magnitudes)
    if trace:
        save_eventprobs(estimates_boot.eventprobs, path+str(n_iter)+'.nc')
    if summarize and 'channels' in resampled_data.dims:
        times = init_boot.compute_times(init_boot, estimates_boot, mean=True)
        channels = init_boot.compute_topologies(resampled_data, estimates_boot, 
                                                init_boot, mean=True)
        boot_results = xr.combine_by_coords([estimates_boot.magnitudes.to_dataset(name='magnitudes'),
                                         estimates_boot.parameters.to_dataset(name='parameters'), 
                                         times.to_dataset(name='event_times'),
                                         channels.to_dataset(name='channels_activity')])
    else:
        boot_results = xr.combine_by_coords([estimates_boot.magnitudes.to_dataset(name='magnitudes'),
                                         estimates_boot.parameters.to_dataset(name='parameters'), 
                                         estimates_boot.eventprobs.to_dataset(name='eventprobs')])
    return boot_results

def bootstrapping(fit, data, dim, n_iterations, init, use_starting_points=True,
                  rerun_pca=False, pca_weights=None, summarize=True,
                  verbose=False, cpus=1, trace=False, path='./'):
    '''
    Performs bootstrapping on ```data``` by fitting the same model as ```fit```

    parameters
    ----------
         fit: hmp fitted model
            fitted object, should contain the estimated parameters and magnitudes
         data: xarray.Dataset 
            epoched raw data
         dim: str | list
            on which dimension to perform the bootstrap (e.g. epochs, participant or particiants and epochs (e.g. ['epochs','participant'])
        n_iterations: int
            How many bootstrap to perform
        init: class hmp()
            initialized hmp object
        use_starting_points: bool
            Whether to use the starting points from the fit (True) or not (False), can be used to check robustness to starting point specification
        rerun_pca: bool
            if True re-performs the PCA on the resampled data (not advised as magnitudes would be meaningless). if False pca_weights need to be passed in pca_weights parameter
        pca_weights: ndarray
            Matrix from the pca performed on the initial hmp_data (e.g. hmp_data.pca_weights)
        summarize: bool
            Whether to keep only channel activity and stage times (True) or wether to store the whole fitted model (False)
        verbose: bool
            Display additional informations on the fits
        trace: bool 
            If True save the event probabilities in a file with path+number_of_iteration.nc
        path: bool 
            where to save the event probabilities of the bootstrapped models
     
     Returns:
     ----------
        boot: xarray.Dataset
            The concatenation of all bootstrapped models into an xarray
    '''
    import itertools
    if 'channels' in data.dims:
        data_type = 'raw'
        if not rerun_pca:
            assert pca_weights is not None, 'If PCA is not re-computed, PC weights from the HMP initial data should be provided through the pca_weights argument'
    else:
        data_type = 'transformed'
    if verbose:
        print(f'Bootstrapping {data_type} on {n_iterations} iterations')
    data_views, data, dim, order = _gen_dataset(data, dim, n_iterations)

    inputs = zip(itertools.repeat(fit), itertools.repeat(data), itertools.repeat(dim), data_views, itertools.repeat(order), 
                itertools.repeat(init),  np.arange(n_iterations),
                itertools.repeat(use_starting_points),
                itertools.repeat(rerun_pca), itertools.repeat(pca_weights),
                itertools.repeat(summarize), itertools.repeat(verbose),
                itertools.repeat(cpus), itertools.repeat(trace), itertools.repeat(path))
    with mp.Pool(processes=cpus) as pool:
            boot = list(tqdm(pool.imap(_boot_star, inputs), total=n_iterations))
        
    boot = xr.concat(boot, dim='iteration')
    # if plots:#Doesn't work with multiprocessing
    #     plot_topo_timecourse(boot.channels, boot.event_times, positions, init_boot, title='iteration = '+str(n_iter), skip_electodes_computation=True)
    return boot

def event_occurence(iterations,  model_to_compare=None, frequency=True, return_mags=None):
    from scipy.spatial import distance_matrix
    if model_to_compare is None:
        max_n_bump_model_index = np.where(np.sum(np.isnan(iterations.magnitudes.values[:,:,0]), axis=(1)) == 0)[0][0]
        max_mags = iterations.sel(iteration=max_n_bump_model_index).magnitudes.squeeze()
        model_to_compare = iterations.sel(iteration=max_n_bump_model_index)
    else:
        max_mags = model_to_compare.magnitudes
        max_n_bump_model_index = 99999
    all_diffs = np.zeros(len(max_mags.event))
    aggregated_mags = np.zeros((len(iterations.iteration), len(max_mags), len(iterations.component)))*np.nan
    aggregated_pars = np.zeros((len(iterations.iteration), len(max_mags)+1))*np.nan
    
    all_n = np.zeros(len(iterations.iteration))
    for iteration in iterations.iteration:
        n_events_iter = int(np.sum(np.isfinite(iterations.sel(iteration=iteration).magnitudes.values[:,0])))
        if iteration != max_n_bump_model_index:
            diffs = distance_matrix(iterations.sel(iteration=iteration).magnitudes.squeeze(),
                        max_mags)[:n_events_iter]
            index_event = diffs.argmin(axis=1)
            i = 0
            for event in index_event:
                if i > 0 and event != index_event[i-1] or i < 1:#Only keeps first encounter
                    all_diffs[event] += 1
                    aggregated_mags[iteration, event] = iterations.sel(iteration=iteration, event=i).magnitudes
                i += 1
                
        all_n[iteration] = n_events_iter
    labels, counts = np.arange(len(max_mags.event)), all_diffs
    n_event_labels, n_event_count = np.unique(all_n, return_counts=True)

    if frequency:
        labels, counts = labels, counts/iterations.iteration.max().values
        n_event_count = n_event_count/iterations.iteration.max().values
    if return_mags is not None:
        return aggregated_mags[:, return_mags, :]
    else:    
        return model_to_compare, labels, counts, n_event_count, n_event_labels

def _boot_star(args): #for tqdm usage
    return _bootstrapped_run(*args)

def select_events(iterations, model_to_compare, selected_events=None, criterion=.65):
    if selected_events is not None:
        boot_mags = event_occurence(iterations, model_to_compare, return_mags=selected_events)
    else:
        _, labels, counts, _, _ = event_occurence(iterations, model_to_compare, frequency=True)
        selected_events = labels[counts>criterion]
        boot_mags = event_occurence(iterations, model_to_compare, frequency=False, return_mags=selected_events)
    mean_mags = np.nanmean(boot_mags, axis=0)
    return mean_mags

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
            If None ( default ) the model to compare is taken as the model with the maximum likelihood among the bootstrapped models 
     	 epoch_data: If model to compare is not among the bootstrapped models, provide the associated epoch_data
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
    

    def propose_fit_params(self, n_events, by_sample, step, j, mags, pars, locations, end):

        if by_sample and n_events > 1: #go through the whole range sample-by-sample, j is sample since start
                
                scale_j = self.mean_to_scale(step*j, self.shape)

                #New parameter proposition
                pars_prop = pars[:n_events].copy() #pars so far
                n_event_j = np.argwhere(scale_j > np.cumsum(pars_prop[:,1])) + 2 #counting from 1
                n_event_j = np.max(n_event_j) if len(n_event_j) > 0 else 1
                n_event_j = np.min([n_event_j, n_events]) #do not insert even after last stage

                #insert j at right spot, subtract prev scales
                pars_prop = np.insert(pars_prop, n_event_j-1, [self.shape, scale_j - np.sum(pars_prop[:n_event_j-1,1])],axis=0)
                #subtract inserted scale from next event
                pars_prop[n_event_j, 1] =  pars_prop[n_event_j, 1] - pars_prop[n_event_j-1, 1]
                last_stage = self.mean_to_scale(end, self.shape) - np.sum(pars_prop[:-1,1])
                pars_prop[n_events,1] = last_stage

                #New location proposition
                locations_props = locations[:n_events].copy()
                locations_props[-1] = 0
                locations_props = np.insert(locations_props, n_event_j, 0)
                if self.location_corr_threshold is None:
                    locations_props[1:-1] = self.location

                mags_props = np.zeros((1,n_events, self.n_dims)) #always 0?
                mags_props[:,:n_events-1,:] = np.tile(mags[:n_events-1,:], (len(mags_props), 1, 1))
                #shift new event to correct position
                mags_props = np.insert(mags_props[:,:-1,:],n_event_j-1,mags_props[:,-1,:],axis=1)

        else: 
            #New parameter proposition
            pars_prop = pars[:n_events+1].copy()
            pars_prop[n_events-1,1] = self.mean_to_scale(step*j, self.shape)
            last_stage = self.mean_to_scale(end, self.shape) - np.sum(pars_prop[:-1,1])
            pars_prop[n_events,1] = last_stage
            
            #New location proposition
            locations_props = locations[:n_events+1].copy()
            locations_props[-1] = 0
            if self.location_corr_threshold is None:
                locations_props[1:-1] = self.location
            mags_props = np.zeros((1,n_events, self.n_dims)) #always 0?
            mags_props[:,:n_events-1,:] = np.tile(mags[:n_events-1,:], (len(mags_props), 1, 1))

        #in edge cases scale can get negative, make sure that doesn't happen:
        pars_prop[:,1] = np.maximum(pars_prop[:,1],self.mean_to_scale(1, self.shape)) 
       
        return mags_props, pars_prop, locations_props

