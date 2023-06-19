import numpy as np
import xarray as xr
import multiprocessing as mp
import warnings
from warnings import warn, filterwarnings
from hsmm_mvpy.models import hmp
from hsmm_mvpy.utils import transform_data, stack_data, save_eventprobs
from hsmm_mvpy.visu import plot_topo_timecourse

def _gen_idx(data, dim, iterations=1):
    orig_index = data[dim].values
    indexes = np.array([np.random.choice(orig_index, size=len(orig_index), replace=True) for x in range(iterations)])
    sort_idx = orig_index.argsort()
    corresponding_indexes = sort_idx[np.searchsorted(orig_index,indexes[0],sorter = sort_idx)]
    return  indexes[0]#corresponding_indexes,

def _gen_dataset(data, dim, n_iterations):   
    original_dim_order = list(data.dims.keys())
    original_dim_order_data = list(data.data.dims)
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

def _bootstrapped_run(data, dim, indexes, order, init, positions, sfreq, n_iter, threshold, summarize=True, verbose=True, plots=True, cpus=1,
                      trace=False, path='./'):
    resampled_data = data.loc[{dim:list(indexes)}].unstack().transpose(*order)
    if 'channels' in resampled_data.dims:
        hmp_data_boot = transform_data(resampled_data, n_comp=init.n_dims)
    else:
        hmp_data_boot = resampled_data
    init_boot = hmp(hmp_data_boot, sfreq=sfreq, event_width=init.event_width, cpus=init.cpus)
    estimates_boot = init_boot.fit(verbose=verbose, threshold=threshold)
    if trace:
        save_eventprobs(estimates_boot.eventprobs, path+str(n_iter)+'.nc')
    if summarize:
        times = init_boot.compute_times(init_boot, estimates_boot, mean=True)
        channels = init_boot.compute_topologies(resampled_data, estimates_boot, 
                                                init_boot.event_width_samples, mean=True)
        boot_results = xr.combine_by_coords([estimates_boot.magnitudes.to_dataset(name='magnitudes'),
                                         estimates_boot.parameters.to_dataset(name='parameters'), 
                                         times.to_dataset(name='event_times'),
                                         channels.to_dataset(name='channels_activity')])
    else:
        boot_results = xr.combine_by_coords([estimates_boot.magnitudes.to_dataset(name='magnitudes'),
                                         estimates_boot.parameters.to_dataset(name='parameters'), 
                                         estimates_boot.eventprobs.to_dataset(name='eventprobs')])
    return boot_results

def bootstrapping(data, dim, n_iterations, init, positions, sfreq, threshold, summarize=True, verbose=True,
                  plots=True, cpus=1, trace=False, path='./'):
    import multiprocessing
    import itertools
    if 'channels' in data.dims:
        data_type = 'raw'
    else:
        data_type = 'transformed'
    if verbose:
        print(f'Bootstrapping {data_type} on {n_iterations} iterations')
    data_views, data, dim, order = _gen_dataset(data, dim, n_iterations)
    with multiprocessing.Pool(processes=cpus) as pool:
        boot = pool.starmap(_bootstrapped_run, 
            zip(itertools.repeat(data), itertools.repeat(dim), data_views, itertools.repeat(order), 
                itertools.repeat(init), 
                itertools.repeat(positions), itertools.repeat(init.sfreq), np.arange(n_iterations),
                itertools.repeat(threshold), itertools.repeat(summarize), itertools.repeat(verbose),
                itertools.repeat(plots),itertools.repeat(cpus), itertools.repeat(trace), itertools.repeat(path)))
    boot = xr.concat(boot, dim='iteration')
    # if plots:
    #     plot_topo_timecourse(boot.channels, boot.event_times, positions, init_boot, title='iteration = '+str(n_iter), skip_electodes_computation=True)
    return boot

def percent_event_occurence(iterations,  model_to_compare=None, count=False):
    from scipy.spatial import distance_matrix
    if model_to_compare is None:
        max_n_bump_model_index = np.where(np.sum(np.isnan(iterations.magnitudes.values[:,:,0]), axis=(1)) == 0)[0][0]
        max_mags = iterations.sel(iteration=max_n_bump_model_index).magnitudes.squeeze()
        model_to_compare = iterations.sel(iteration=max_n_bump_model_index)
    else:
        max_mags = model_to_compare.magnitudes
    all_diffs = []
    all_n = np.zeros(len(iterations.iteration))
    for iteration in iterations.iteration:
        n_events_iter = int(np.sum(np.isfinite(iterations.sel(iteration=iteration).magnitudes.values[:,0])))
        # if iteration != max_n_bump_model_index:
        diffs = distance_matrix(iterations.sel(iteration=iteration).magnitudes.squeeze(),
                    max_mags)
        index_event = diffs.argmin(axis=1)
        index_event[n_events_iter:] = 9999
        all_diffs.append(index_event)
        all_n[iteration] = n_events_iter
    labels, counts = np.unique(all_diffs, return_counts=True)
    n_event_labels, n_event_count = np.unique(all_n, return_counts=True)
    if count:
        labels, counts = labels[:-1], counts[:-1]
    else:
        labels, counts = labels[:-1], counts[:-1]/iterations.iteration.max().values
        n_event_count = n_event_count/iterations.iteration.max().values
    return model_to_compare, labels, counts, n_event_count, n_event_labels

