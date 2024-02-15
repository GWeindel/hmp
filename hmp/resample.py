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
                        location=init.location, distribution=init.distribution, em_method=init.em_method)
    else:
        init_boot = classhmp(hmp_data_boot, sfreq=sfreq, event_width=init.event_width, cpus=1,
                        shape=init.shape, template=init.template,
                        location=init.location, distribution=init.distribution, em_method=init.em_method)
    estimates_boot = init_boot.fit_single(fit.magnitudes.shape[0], verbose=verbose, parameters=fit.parameters,
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

# def _bootstrapping_fit(data, dim, n_iterations, init, positions, tolerance=1e-4,
#                   rerun_pca=False, pca_weights=None, decimate=None, summarize=True,
#                   verbose=False, plots=True, cpus=1, trace=False, path='./'):
#     import itertools
#     if 'channels' in data.dims:
#         data_type = 'raw'
#         if not rerun_pca:
#             assert pca_weights is not None, 'If PCA is not re-computed, PC weights from the HMP initial data should be provided through the pca_weights argument'
#     else:
#         data_type = 'transformed'
#     if verbose:
#         print(f'Bootstrapping {data_type} on {n_iterations} iterations')
#     data_views, data, dim, order = _gen_dataset(data, dim, n_iterations)
    
#     inputs = zip(itertools.repeat(data), itertools.repeat(dim), data_views, itertools.repeat(order), 
#                 itertools.repeat(init), 
#                 itertools.repeat(positions), itertools.repeat(init.sfreq), np.arange(n_iterations),
#                 itertools.repeat(tolerance), itertools.repeat(rerun_pca), itertools.repeat(pca_weights),
#                 itertools.repeat(decimate), itertools.repeat(summarize), itertools.repeat(verbose),
#                 itertools.repeat(plots),itertools.repeat(cpus), itertools.repeat(trace), itertools.repeat(path))
#     with mp.Pool(processes=cpus) as pool:
#             boot = list(tqdm(pool.imap(_boot_star, inputs), total=n_iterations))
        
#     boot = xr.concat(boot, dim='iteration')
#     # if plots:
#     #     plot_topo_timecourse(boot.channels, boot.event_times, positions, init_boot, title='iteration = '+str(n_iter), skip_electodes_computation=True)
#     return boot

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