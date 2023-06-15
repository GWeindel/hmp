import numpy as np
import xarray as xr
import multiprocessing as mp
import warnings
from warnings import warn, filterwarnings
from hsmm_mvpy.models import hmp
from hsmm_mvpy.utils import transform_data, stack_data
from hsmm_mvpy.visu import plot_topo_timecourse

def _gen_idx(data, dim, iterations=1):
    orig_index = data[dim].values
    indexes = np.array([np.random.choice(orig_index, size=len(orig_index), replace=True) for x in range(iterations)])
    sort_idx = orig_index.argsort()
    corresponding_indexes = sort_idx[np.searchsorted(orig_index,indexes[0],sorter = sort_idx)]
    return indexes[0], corresponding_indexes

def _gen_dataset(data, dim):   
    original_dim_order = list(data.dims.keys())
    if isinstance(dim, list):
        if len(dim) == 2:
            data = data.stack({str(dim[0])+'_x_'+str(dim[1]):dim})
            dim = str(dim[0])+'_x_'+str(dim[1])
        elif len(dim) > 2:
            raise ValueError('Cannot stack more than two dimensions')
        else: dim = dim[0]
    named_index, data_index = _gen_idx(data, dim)
    data = data.loc[{dim:named_index}]
    if '_x_' in dim: 
        data = data.unstack(dim)
    data = data.transpose(*original_dim_order) 
    return data

def _bootstrapped_run(resampled_data, init, positions, sfreq, threshold=1, verbose=True, plots=True, cpus=1, retransform=True):
    # if 'electrodes' in resampled_data.dims:
    print('PCA')
    hmp_data_boot = transform_data(resampled_data, n_comp=init.n_dims)
    print(True)
    # else:
    #     hmp_data_boot = resampled_data
    init_boot = hmp(hmp_data_boot, sfreq=sfreq, event_width=init.event_width, cpus=init.cpus)
    estimates_boot = init_boot.fit(verbose=verbose, threshold=threshold)
    return estimates_boot
#         mags_boot_mat.append(estimates_boot.magnitudes)
#         pars_boot_mat.append(estimates_boot.parameters)
#         if plots:
#             plot_topo_timecourse(bootstapped.squeeze('iteration'), estimates_boot, positions, init_boot)

#     all_pars_aligned = np.tile(np.nan, (iterations, np.max([len(x) for x in pars_boot_mat]), 2))
#     all_mags_aligned = np.tile(np.nan, (iterations, np.max([len(x) for x in mags_boot_mat]), init_boot.n_dims))
#     for iteration, _i in enumerate(zip(pars_boot_mat, mags_boot_mat)):
#         all_pars_aligned[iteration, :len(_i[0]), :] = _i[0]
#         all_mags_aligned[iteration, :len(_i[1]), :] = _i[1]

#     booted = xr.Dataset({'parameters': (('iteration', 'stage','parameter'), 
#                                  all_pars_aligned),
#                         'magnitudes': (('iteration', 'event','component'), 
#                                  all_mags_aligned)})
#     return booted

def percent_event_occurence(iterations, count=False):
    from scipy.spatial import distance_matrix
    max_n_bump_model_index = np.where(np.sum(np.isnan(iterations.magnitudes.values[:,:,0]), axis=(1)) == 0)[0][0]
    all_diffs = []
    all_n = np.zeros(len(iterations.iteration))
    for iteration in iterations.iteration:
        n_events_iter = int(np.sum(np.isfinite(iterations.sel(iteration=iteration).magnitudes.values[:,0])))
        if iteration != max_n_bump_model_index:
            diffs = distance_matrix(iterations.sel(iteration=iteration).magnitudes.squeeze(),
                        iterations.sel(iteration=max_n_bump_model_index).magnitudes.squeeze())
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
    return iterations.sel(iteration=max_n_bump_model_index), labels, counts, n_event_count, n_event_labels

