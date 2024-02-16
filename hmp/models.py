'''

'''

import numpy as np
import xarray as xr
import multiprocessing as mp
import itertools
from pandas import MultiIndex
from warnings import warn, filterwarnings, resetwarnings
from scipy.stats import gamma as sp_gamma
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from hmp import utils
from itertools import cycle, product
from scipy.stats import sem
import gc

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

default_colors =  ['cornflowerblue','indianred','orange','darkblue','darkgreen','gold', 'brown']

class hmp:
    
    def __init__(self, data, epoch_data=None, sfreq=None, cpus=1, event_width=50, shape=2, estimate_magnitudes=True, estimate_parameters=True, template=None, location=None, distribution='gamma', em_method="mean"):
        '''
        This function intializes an HMP model by providing the data, the expected probability distribution for the by-trial variation in stage onset, and the expected duration of the transition event.

        parameters
        ----------
        data : xr.Dataset
            xr.Dataset obtained dthrough the hmp.utils.transform_data() function
        epoch_data: xr.Dataset
            Xarray dataset with the EEG/MEG data
        sfreq : float
            (optional) Sampling frequency of the signal if not provided, inferred from the epoch_data
        cpus: int
            How many cpus to use for the functions`using multiprocessing`
        event_width : float
            width of events in milliseconds, originally 50 ms
        shape: float
            shape of the probability distributions of the by-trial stage onset (one shape for all stages)
        estimated_magnitudes: bool
            To estimate (True) or not the magnitudes of the transition events
        estimate_parameters: bool
            To estimate (True) or not the parameters parameter of the stages
        template: ndarray
            Expected shape for the transition event used in the cross-correlation, should be a vector of values capturing the expected shape over the sampling frequency of the data. If None, the template is created as a half-sine shape with a frequency derived from the event_width argument
        location : float
            Minimum stage duration in samples. 
        distribution: str
            Probability distribution for the by-trial onset of stages can be one of 'gamma','lognormal','wald', or 'weibull'
        em_method: str
            can be either mean or max, max method isn't yet supported, only use the 'mean'method (default)
        '''
        match distribution:
            case 'gamma':
                from scipy.stats import gamma as sp_dist
                from hmp.utils import gamma_scale_to_mean, gamma_mean_to_scale
                self.scale_to_mean, self.mean_to_scale = gamma_scale_to_mean, gamma_mean_to_scale
            case 'lognormal':
                from scipy.stats import lognorm as sp_dist
                from hmp.utils import logn_scale_to_mean,logn_mean_to_scale
                self.scale_to_mean, self.mean_to_scale = logn_scale_to_mean, logn_mean_to_scale
            case 'wald':
                from scipy.stats import invgauss as sp_dist
                from hmp.utils import wald_scale_to_mean,wald_mean_to_scale
                self.scale_to_mean, self.mean_to_scale = wald_scale_to_mean, wald_mean_to_scale
            case 'weibull':
                from scipy.stats import weibull_min as sp_dist
                from hmp.utils import weibull_scale_to_mean,weibull_mean_to_scale
                self.scale_to_mean, self.mean_to_scale = weibull_scale_to_mean, weibull_mean_to_scale
            case 'log-logistic':
                from scipy.stats import fisk as sp_dist
                from hmp.utils import fisk_scale_to_mean,fisk_mean_to_scale
                self.scale_to_mean, self.mean_to_scale = fisk_scale_to_mean,fisk_mean_to_scale
            case 'maxwell-boltzmann':
                from scipy.stats import chi as sp_dist
                from hmp.utils import maxb_scale_to_mean,maxb_mean_to_scale
                shape = 3
                self.scale_to_mean, self.mean_to_scale = maxb_scale_to_mean,maxb_mean_to_scale
            case 'rayleigh':
                from scipy.stats import chi as sp_dist
                from hmp.utils import ray_scale_to_mean,ray_mean_to_scale
                shape = 2
                self.scale_to_mean, self.mean_to_scale = ray_scale_to_mean,ray_mean_to_scale
            case 'half-normal':
                from scipy.stats import chi as sp_dist
                from hmp.utils import halfn_scale_to_mean,halfn_mean_to_scale
                shape = 1
                self.scale_to_mean, self.mean_to_scale = halfn_scale_to_mean,halfn_mean_to_scale
            case 'uniform':
                from scipy.stats import uniform as sp_dist
                from hmp.utils import uniform_scale_to_mean,uniform_mean_to_scale
                self.scale_to_mean, self.mean_to_scale = uniform_scale_to_mean,uniform_mean_to_scale
            case _:
                raise ValueError(f'Unknown Distribution {distribution}')
        self.distribution = distribution
        self.pdf = sp_dist.pdf
        if sfreq is None:
            sfreq = data.sfreq
        self.sfreq = sfreq
        self.steps = 1000/self.sfreq
        self.shape = float(shape)
        self.event_width = event_width
        self.event_width_samples = int(np.round(self.event_width / self.steps))
        if location is None:
            self.location = int(self.event_width / self.steps//2)+1
        else:
            self.location = int(np.rint(location))
        durations = data.unstack().sel(component=0).rename({'epochs':'trials'})\
            .stack(trial_x_participant=['participant','trials']).dropna(dim="trial_x_participant",\
            how="all").groupby('trial_x_participant').count(dim="samples").cumsum().squeeze()
        if durations.trial_x_participant.count() > 1:
            dur_dropped_na = durations.dropna("trial_x_participant")
            starts = np.roll(dur_dropped_na.data, 1)
            starts[0] = 0
            ends = dur_dropped_na.data-1
        else: 
            dur_dropped_na = durations
            starts = np.array([0])
            ends = np.array([dur_dropped_na.data-1])
        self.starts = starts
        self.ends = ends
        self.durations =  self.ends-self.starts+1
        if durations.trial_x_participant.count() > 1:
            self.named_durations =  durations.dropna("trial_x_participant") - durations.dropna("trial_x_participant").shift(trial_x_participant=1, fill_value=0)
            self.coords = durations.reset_index('trial_x_participant').coords
        else: 
            self.named_durations = durations
            self.coords = durations.coords
        self.mean_d = self.durations.mean()
        self.n_trials = durations.trial_x_participant.count().values
            
        self.cpus = cpus
        self.n_samples, self.n_dims = np.shape(data.data.T)
        if template is None:
            self.template = self._event_shape()
        else: self.template = template
        self.events = self.cross_correlation(data.data.T)#adds event morphology
        self.max_d = self.durations.max()
        self.em_method = em_method
        if self.em_method == "mean":
            self.data_matrix = np.zeros((self.max_d, self.n_trials, self.n_dims), dtype=np.float64)
            for trial in range(self.n_trials):
                self.data_matrix[:self.durations[trial],trial,:] = \
                    self.events[self.starts[trial]:self.ends[trial]+1,:]
                #Reorganize samples crosscorrelated with template on trial basis
        self.estimate_magnitudes = estimate_magnitudes
        self.estimate_parameters = estimate_parameters
        if self.max_d > 500:#FFT conv from scipy faster in this case
            from scipy.signal import fftconvolve
            self.convolution = fftconvolve
        else:
            self.convolution = np.convolve
        self.trial_coords = data.unstack().sel(component=0,samples=0).rename({'epochs':'trials'}).\
            stack(trial_x_participant=['participant','trials']).dropna(dim="trial_x_participant",how="all").coords
    
    def _event_shape(self):
        '''
        Computes the template of a half-sine (event) with given frequency f and sampling frequency
        '''
        event_idx = np.arange(self.event_width_samples)*self.steps + self.steps / 2
        event_frequency = 1000/(self.event_width*2)#gives event frequency given that events are defined as half-sines
        template = np.sin(2*np.pi*event_idx/1000*event_frequency)#event morph based on a half sine with given event width and sampling frequency
        template = template/np.sum(template**2)#Weight normalized
        return template
            
    def cross_correlation(self,data):
        '''
        This function puts on each sample the correlation of that sample and the next 
        x samples (depends on sampling frequency and event size) with a half sine on time domain.
        
        parameters
        ----------
        data : ndarray
            2D ndarray with n_samples * components

        Returns
        -------
        events : ndarray
            a 2D ndarray with samples * PC components where cell values have
            been correlated with event morphology
        '''
        from scipy.signal import correlate
        events = np.zeros(data.shape)
        for trial in range(self.n_trials):#avoids confusion of gains between trials
            for dim in np.arange(self.n_dims):
                events[self.starts[trial]:self.ends[trial]+1,dim] = correlate(data[self.starts[trial]:self.ends[trial]+1, dim], self.template, mode='same', method='direct')
        return events

    def fit_single(self, n_events=None, magnitudes=None, parameters=None, parameters_to_fix=None, 
                   magnitudes_to_fix=None, tolerance=1e-4, max_iteration=1e3, maximization=True, min_iteration = 1,
                   starting_points=1, method='random', return_max=True, verbose=True, cpus=None):
        '''
        Fit HMP for a single n_events model
        
        parameters
        ----------
        n_events : int
            how many events are estimated
        magnitudes : ndarray
            2D ndarray n_events * components (or 3D iteration * n_events * n_components), initial conditions for events magnitudes. If magnitudes are estimated, the list provided is used as starting point,
            if magnitudes are fixed, magnitudes estimated will be the same as the one provided. When providing a list, magnitudes need to be in the same order
            _n_th magnitudes parameter is  used for the _n_th event
        parameters : list
            list of initial conditions for Gamma distribution parameters parameter (2D stage * parameter or 3D iteration * n_events * n_components). If parameters are estimated, the list provided is used as starting point,
            if parameters are fixed, parameters estimated will be the same as the one provided. When providing a list, stage need to be in the same order
            _n_th gamma parameter is  used for the _n_th stage
        parameters_to_fix : bool
            To fix (True) or to estimate (False, default) the parameters of the gammas
        magnitudes_to_fix: bool
            To fix (True) or to estimate (False, default) the magnitudes of the channel contribution to the events
        tolerance: float
            Tolerance applied to the expectation maximization in the EM() function
        max_iteration: int
            Maximum number of iteration for the expectation maximization in the EM() function
        maximization: bool
            If True (Default) perform the maximization phase in EM() otherwhise skip
        min_iteration: int
            Minimum number of iteration for the expectation maximization in the EM() function
        starting_points: int
            How many starting points to use for the EM() function
        method: str
            What starting points generation method to use, 'random'or 'grid' (grid is not yet fully supported)
        return_max: bool
            In the case of multiple starting points, dictates whether to only return the max likelihood model (True, default) or all of the models (False)
        verbose: bool
            True displays output useful for debugging, recommended for first use
        cpus: int
            number of cores to use in the multiprocessing functions
        '''
        assert n_events is not None, 'The fit_single() function needs to be provided with a number of expected transition events'
        assert self.location*(n_events-2) < min(self.durations), f'{n_events} events do not fit given the minimum duration of {min(self.durations)} and a location of {self.location}'
        if verbose:
            if parameters is None:
                print(f'Estimating {n_events} events model with {starting_points} starting point(s)')
            else:
                print(f'Estimating {n_events} events model')
        if cpus is None:
            cpus = self.cpus
        if n_events is None and parameters is not None:
            n_events = len(parameters)-1
        if self.estimate_magnitudes == False:#Don't need to manually fix mags if not estimated
            magnitudes_to_fix = np.arange(n_events)
        if self.estimate_parameters == False:#Don't need to manually fix pars if not estimated
            parameters_to_fix = np.arange(n_events+1)            
        #Formatting parameters
        if isinstance(parameters, (xr.DataArray,xr.Dataset)):
            parameters = parameters.dropna(dim='stage').values
        if isinstance(magnitudes, (xr.DataArray,xr.Dataset)):
            magnitudes = magnitudes.dropna(dim='event').values  
        if isinstance(magnitudes, np.ndarray):
            magnitudes = magnitudes.copy()
        if isinstance(parameters, np.ndarray):
            parameters = parameters.copy()          
        if parameters_to_fix is None: parameters_to_fix=[]
        if magnitudes_to_fix is None: magnitudes_to_fix=[]
        
        if parameters is not None:
            if len(np.shape(parameters)) == 3:
                starting_points = np.shape(parameters)[0]
        if magnitudes is not None:
            if len(np.shape(magnitudes)) == 3:
                starting_points = np.shape(magnitudes)[0]
        
        if starting_points > 0:#Initialize with equally spaced option
            if parameters is None:
                parameters = np.tile([self.shape, self.mean_to_scale(np.mean(self.durations)/(n_events+1),self.shape)], (n_events+1,1))
            initial_p = parameters
            
            if magnitudes is None:
                magnitudes = np.zeros((n_events,self.n_dims), dtype=np.float64)
            initial_m = magnitudes
        if starting_points > 1 or len(np.shape(magnitudes)) == 3 or len(np.shape(parameters)) == 3:
            filterwarnings('ignore', 'Convergence failed, estimation hitted the maximum ', )#will be the case but for a subset only hence ignore
            if len(np.shape(initial_m)) == 2:
                parameters = [initial_p]
                magnitudes = np.zeros((starting_points, n_events, self.n_dims))
                magnitudes[0] = initial_m
                if method == 'random':
                    for _ in np.arange(starting_points):
                        proposal_p = self.gen_random_stages(n_events)
                        proposal_p[parameters_to_fix] = initial_p[parameters_to_fix]
                        parameters.append(proposal_p)
                    magnitudes[1:] = self.gen_mags(n_events, starting_points-1, method='random', verbose=False)
                    magnitudes[:,magnitudes_to_fix,:] = np.tile(initial_m[magnitudes_to_fix], (len(magnitudes), 1, 1))
                elif method == 'grid':
                    parameters = self._grid_search(n_events+1, iter_limit=starting_points, method='grid')
                    magnitudes = np.zeros((len(parameters), n_events, self.n_dims), dtype=np.float64)#Grid search is not yet done for mags
                else:
                    raise ValueError('Unknown starting point method requested, use "random" or "grid"')
            elif len(np.shape(initial_m)) == 3:
                magnitudes = initial_m
                if len(np.shape(initial_p)) == 3:
                    parameters = initial_p
                else:
                    parameters = np.tile(parameters, (len(magnitudes),1,1))
            if cpus>1: 
                inputs = zip(magnitudes, parameters, itertools.repeat(maximization),
                        itertools.repeat(magnitudes_to_fix),itertools.repeat(parameters_to_fix), itertools.repeat(max_iteration), itertools.repeat(tolerance), itertools.repeat(min_iteration))
                with mp.Pool(processes=cpus) as pool:
                    if starting_points > 1:
                        estimates = list(tqdm(pool.imap(self._EM_star, inputs), total=len(magnitudes)))
                    else:
                        estimates = pool.starmap(self.EM, inputs)
 
            else:#avoids problems if called in an already parallel function
                estimates = []
                for pars, mags in zip(parameters, magnitudes):
                    estimates.append(self.EM(mags, pars, maximization,\
                    magnitudes_to_fix, parameters_to_fix, max_iteration, tolerance, min_iteration))
                resetwarnings()
            lkhs_sp = [x[0] for x in estimates]
            mags_sp = [x[1] for x in estimates]
            pars_sp = [x[2] for x in estimates]
            eventprobs_sp = [x[3] for x in estimates]
            traces_sp = [x[4] for x in estimates]
            non_converged = 0
            for iteration in range(len(estimates)):
                #Filters out non-converged models
                if np.diff(estimates[iteration][4][-2:]) < 0:
                    lkhs_sp[iteration] = -np.inf
                    non_converged += 1
                    
            if verbose and non_converged > 0:
                warn(f'{non_converged}/{starting_points} starting points ended up not converging', RuntimeWarning)
            if return_max:
                max_lkhs = np.argmax(lkhs_sp)
                lkh = lkhs_sp[max_lkhs]
                mags = mags_sp[max_lkhs]
                pars = pars_sp[max_lkhs]
                eventprobs = eventprobs_sp[max_lkhs]
                traces = traces_sp[max_lkhs]
            else:
                lkh = lkhs_sp
                mags = mags_sp
                pars = pars_sp
                eventprobs = eventprobs_sp
                traces = np.zeros((len(lkh),  max([len(x) for x in traces_sp])))*np.nan
                for i, _i in enumerate(traces_sp):
                    traces[i, :len(_i)] = _i
            
        elif starting_points==1:#informed starting point
            lkh, mags, pars, eventprobs, traces = self.EM(initial_m, initial_p,\
                                        maximization, magnitudes_to_fix, parameters_to_fix, \
                                         max_iteration, tolerance, min_iteration)
            
        else:#uninitialized    
            if np.any(parameters)== None:
                parameters = np.tile([self.shape, self.mean_to_scale(np.mean(self.durations)/(n_events+1),self.shape)], (n_events+1,1))
            initial_p = parameters
            if np.any(magnitudes)== None:
                magnitudes = np.zeros((n_events, self.n_dims), dtype=np.float64)
            lkh, mags, pars, eventprobs, traces = self.EM(magnitudes, parameters, maximization, magnitudes_to_fix, parameters_to_fix,\
                                        max_iteration, tolerance, min_iteration)
        if len(np.shape(eventprobs)) == 3:
            n_event_xr = n_event_xreventprobs = len(mags)
            n_stage = n_event_xr+1
        elif len(np.shape(eventprobs)) == 2:#0 event case
            eventprobs = np.transpose(eventprobs[np.newaxis], axes=(1,2,0))
            mags = np.transpose(mags[np.newaxis], axes=(1,0))
            n_event_xr = 0
            n_event_xreventprobs = 1
            n_stage = 1
        if len(np.shape(eventprobs)) == 3:
            xrlikelihoods = xr.DataArray(lkh , name="likelihoods")
            xrtraces = xr.DataArray(traces, dims=("em_iteration"), name="traces", coords={'em_iteration':range(len(traces))})
            xrparams = xr.DataArray(pars, dims=("stage",'parameter'), name="parameters", 
                            coords = [range(n_stage), ['shape','scale']])
            xrmags = xr.DataArray(mags, dims=("event","component"), name="magnitudes",
                        coords={'event':range(n_event_xr),
                                "component":range(self.n_dims)})
            part, trial = self.coords['participant'].values, self.coords['trials'].values
            trial_x_part = xr.Coordinates.from_pandas_multiindex(MultiIndex.from_arrays([part,trial],
                                    names=('participant','trials')),'trial_x_participant')
            xreventprobs = xr.Dataset({'eventprobs': (('event', 'trial_x_participant','samples'), 
                                             eventprobs.T)},
                             {'event':range(n_event_xreventprobs),
                              'samples':range(np.shape(eventprobs)[0])})
            xreventprobs = xreventprobs.assign_coords(trial_x_part)
            
            xreventprobs = xreventprobs.transpose('trial_x_participant','samples','event')
        else: 
            n_event_xr = len(mags[0])
            if verbose and traces[0, -1] - traces[0, -2] < 0:
                warn(f'Last iteration of the estimation procedure lead to a decrease in log-likelihood, convergence issue. The resulting fit likely contains a duplicated event', RuntimeWarning)
            xrlikelihoods = xr.DataArray(lkh , dims=("iteration"), name="likelihoods", coords={'iteration':range(len(lkh))})
            xrtraces = xr.DataArray(traces, dims=("iteration","em_iteration"), name="traces", coords={'iteration':range(len(lkh)), 'em_iteration':range(len(traces[0]))})
            xrparams = xr.DataArray(pars, dims=("iteration","stage",'parameter'), name="parameters", 
                            coords = {'iteration': range(len(lkh)), 'parameter':['shape','scale']})
            xrmags = xr.DataArray(mags, dims=("iteration","event","component"), name="magnitudes",
                        coords={'iteration':range(len(lkh)), 'event':range(n_event_xr),
                                "component":range(self.n_dims)})
            part, trial = self.coords['participant'].values, self.coords['trials'].values
            trial_x_part = xr.Coordinates.from_pandas_multiindex(MultiIndex.from_arrays([part,trial],
                                    names=('participant','trials')),'trial_x_participant')
            xreventprobs = xr.Dataset({'eventprobs': (('iteration','event', \
                                    'trial_x_participant','samples'), [x.T for x in eventprobs])},
                             {'iteration':range(len(lkh)),
                              'event':np.arange(n_event_xr),
                              'samples':np.arange(np.shape(eventprobs)[1])})
            xreventprobs = xreventprobs.assign_coords(trial_x_part)
            xreventprobs = xreventprobs.transpose('iteration','trial_x_participant','samples','event')
        estimated = xr.merge((xrlikelihoods, xrparams, xrmags, xreventprobs, xrtraces))

        if verbose:
            print(f"parameters estimated for {n_events} events model")
        return estimated
    
    def fit_single_conds(self, magnitudes=None, parameters=None, parameters_to_fix=None, 
                   magnitudes_to_fix=None, tolerance=1e-4, max_iteration=1e3, maximization=True, min_iteration = 1,
                   starting_points=1, method='random', return_max=True, verbose=True, cpus=None,
                   mags_map=None, pars_map=None, conds=None):
        '''
        Fit HMP for a single n_events model
        
        parameters
        ----------
        n_events : int
            how many events are estimated
        magnitudes : ndarray
            2D or 3D ndarray n_events * components (3D: n_cond * n_events * n_components), initial conditions for events magnitudes. If magnitudes are estimated, the list provided is used as starting point, if magnitudes are fixed, magnitudes estimated will be the same as the one provided.
        parameters : list
            2D or 3D nd_array stage * parameters (3D: n_cond * stage * parameters) of initial conditions for Gamma distribution parameters parameter. If parameters are estimated, the list provided is used as starting point,
            if parameters are fixed, parameters estimated will be the same as the one provided.
        parameters_to_fix : bool
            To fix (True) or to estimate (False, default) the parameters of the gammas
        magnitudes_to_fix: bool
            To fix (True) or to estimate (False, default) the magnitudes of the channel contribution to the events
        tolerance: float
            Tolerance applied to the expectation maximization in the EM() function
        max_iteration: int
            Maximum number of iteration for the expectation maximization in the EM() function
        maximization: bool
            If True (Default) perform the maximization phase in EM() otherwhise skip
        min_iteration: int
            Minimum number of iteration for the expectation maximization in the EM() function
        starting_points: int
            How many starting points to use for the EM() function
        method: str
            What starting points generation method to use, 'random'or 'grid' (grid is not yet fully supported)
        return_max: bool
            In the case of multiple starting points, dictates whether to only return the max likelihood model (True, default) or all of the models (False)
        verbose: bool
            True displays output useful for debugging, recommended for first use
        cpus: int
            number of cores to use in the multiprocessing functions
        mags_map: 2D nd_array n_cond * n_events indicating which magnitudes are shared between conditions.
        pars_map: 2D nd_array n_cond * n_stages indicating which parameters are shared between conditions.
        conds: dict | list
            if one condition, use a dict with the name in the metadata and a list of the levels in the same
            order as the rows of the map(s). E.g., {'cue': ['SP', 'AC']}
            if multiple conditions need to be crossed, use a list of dictionaries per condition. E.g.,
            [{'cue': ['SP', 'AC',]}, {'resp': ['left', 'right']}]. These are crossed by repeating
            the first condition as many times as there are levels in the second condition. E.g., SP-left, 
            SP-right, AC-left, AC-right.
        '''
        #Conditions
        assert mags_map is not None or pars_map is not None, 'fit_single_conds() requires a magnitude and/or parameter map'
        assert conds is not None, 'fit_single_conds() requires conditions argument'
        assert isinstance(conds, dict) or isinstance(conds[0], dict), 'conditions have to be specified as a dictionary, or list of dictionaries'
        if isinstance(conds, dict): conds = [conds]
        conds_dict = conds

        #collect condition names, levels, and trial coding
        cond_names = []
        cond_levels = []
        cond_trials = []
        for cond in conds:
            assert len(cond) == 1, 'Each condition dictionary can only contain one condition (e.g. {\'cue\': [\'SP\', \'AC\']})'
            cond_names.append(list(cond.keys())[0])
            cond_levels.append(cond[cond_names[-1]])
            cond_trials.append(self.trial_coords[cond_names[-1]].data.copy())
            if verbose:
                print('Condition \"' + cond_names[-1] + '\" analyzed, with levels:', cond_levels[-1])

        cond_levels = list(product(*cond_levels))
        cond_levels = np.array(cond_levels, dtype=object)
        n_conds = len(cond_levels)

        #build condition array with digit indicating the combined levels
        cond_trials = np.vstack(cond_trials).T
        conds = np.zeros((cond_trials.shape[0])) * np.nan
        if verbose:
            print('\nCoded as follows: ')
        for i, level in enumerate(cond_levels):
            assert len(np.where((cond_trials == level).all(axis=1))[0]) > 0, f'condition level {level} does not occur in the data'
            conds[np.where((cond_trials == level).all(axis=1))] = i
            if verbose:
                print(str(i) + ': ' + str(level))
        conds=np.int8(conds)

        clabels = {'Condition ' + str(cond_names): cond_levels}

        #check maps
        n_conds_mags = 0 if mags_map is None else mags_map.shape[0]
        n_conds_pars = 0 if pars_map is None else pars_map.shape[0]
        if n_conds_mags > 0 and n_conds_pars > 0: #either both maps should have the same number of conds, or 0
            assert n_conds_mags == n_conds_pars, 'magnitude and parameters maps have to indicate the same number of conditions'
            #make sure nr of events correspond per row
            for c in range(n_conds):
                assert sum(mags_map[c,:] >= 0) + 1 == sum(pars_map[c,:] >= 0), 'nr of events in magnitudes map and parameters map do not correspond on row ' + str(c)
        else: #if 0, copy n_conds as zero map
            if n_conds_mags == 0:
                assert not (pars_map < 0).any(), 'If negative parameters are provided, magnitude map is required.'
                mags_map = np.zeros((n_conds, pars_map.shape[1]-1), dtype=int)
            else:
                pars_map = np.zeros((n_conds, mags_map.shape[1] + 1), dtype=int)
                if (mags_map < 0).any():
                    for c in range(n_conds):
                        pars_map[c, np.where(mags_map[c,:] < 0)[0]] = -1
                        pars_map[c, np.where(mags_map[c,:] < 0)[0]+1] = 1

        #print maps to check level/row mathcing
        if verbose:
            print('\nMagnitudes map:')
            for cnt in range(n_conds):
                print(str(cnt) + ': ', mags_map[cnt,:])

            print('\nParameters map:')
            for cnt in range(n_conds):
                print(str(cnt) + ': ', pars_map[cnt,:])

            #give explanation if negative parameters:
            if (pars_map < 0).any():
                print('\n-----')
                print('Negative parameters. Note that this stage is left out, while the parameters')
                print('of the other stages are compared column by column. In this parameter map example:')
                print(np.array([[0, 0, 0, 0],
                                [0, -1, 0, 0]]))
                print('the parameters of stage 1 are shared, as well as the parameters of stage 3 of')
                print('condition 1 with stage 2 (column 3) of condition 2 and the last stage of both')
                print('conditions.')
                print('Given that event 2 is probably missing in condition 2, it would typically')
                print('make more sense to let both stages around event 2 in condition 1 vary as')
                print('compared to condition 2:')
                print(np.array([[0, 0, 0, 0],
                                [0, -1, 1, 0]]))
                print('-----')

                
        #at this point, all should indicate the same number of conditions
        assert n_conds == mags_map.shape[0] == pars_map.shape[0], 'number of unique conditions should correspond to number of rows in map(s)'

        n_events = mags_map.shape[1]
        assert self.location*(n_events) < min(self.durations), f'{n_events} events do not fit given the minimum duration of {min(self.durations)} and a location of {self.location}'
        assert conds.shape[0] == self.durations.shape[0], 'Conds parameter should contain the condition per epoch.'
        if verbose:
            if parameters is None:
                print(f'\nEstimating {n_events} events model with {starting_points} starting point(s)')
            else:
                print(f'\nEstimating {n_events} events model')
        if cpus is None:
            cpus = self.cpus
     
        if self.estimate_magnitudes == False:#Don't need to manually fix mags if not estimated
            magnitudes_to_fix = np.arange(n_events)
        if self.estimate_parameters == False:#Don't need to manually fix pars if not estimated
            parameters_to_fix = np.arange(n_events+1)            
        
        #Formatting parameters
        if isinstance(parameters, (xr.DataArray,xr.Dataset)):
            parameters = parameters.dropna(dim='stage').values
        if isinstance(magnitudes, (xr.DataArray,xr.Dataset)):
            magnitudes = magnitudes.dropna(dim='event').values  
        if isinstance(magnitudes, np.ndarray):
            magnitudes = magnitudes.copy()
        if isinstance(parameters, np.ndarray):
            parameters = parameters.copy()          
        if parameters_to_fix is None: parameters_to_fix=[]
        if magnitudes_to_fix is None: magnitudes_to_fix=[]
        
        if parameters is not None:
            if len(np.shape(parameters)) == 2: #broadcast parameters across conditions
                parameters = np.tile(parameters, (n_conds, 1, 1))
            assert parameters.shape[1] == n_events + 1, f'Provided parameters ({ parameters.shape[1]} should match number of stages {n_events + 1} in parameters map'

            #set params missing stages to nan to make it obvious in the results
            if (pars_map < 0).any():
                for c in range(n_conds):
                    parameters[c, np.where(pars_map[c,:]<0)[0],:] = np.nan
            
        if magnitudes is not None:
            if len(np.shape(magnitudes)) == 2: #broadcast magnitudes across conditions
                magnitudes = np.tile(magnitudes, (n_conds, 1, 1))
            assert magnitudes.shape[1] == n_events, 'Provided magnitudes should match number of events in magnitudes map'

            #set mags missing events to nan to make it obvious in the results
            if (mags_map < 0).any():
                for c in range(n_conds):
                    magnitudes[c, np.where(mags_map[c,:]<0)[0],:] = np.nan

        if starting_points > 0:#Initialize with equally spaced option
            if parameters is None:
                parameters = np.zeros((n_conds,n_events + 1, 2)) * np.nan #by default nan for missing stages
                for c in range(n_conds):
                    pars_cond = np.where(pars_map[c,:]>=0)[0]
                    n_stage_cond = len(pars_cond)
                    parameters[c,pars_cond,:] = np.tile([self.shape, self.mean_to_scale(np.mean(self.durations[conds==c])/(n_stage_cond),self.shape)], (n_stage_cond,1))
            initial_p = parameters
            if magnitudes is None:
                magnitudes = np.zeros((n_events,self.n_dims), dtype=np.float64)
                magnitudes = np.tile(magnitudes, (n_conds, 1, 1)) #broadcast across conditions
                if (mags_map < 0).any(): #set missing mags to nan
                    for c in range(n_conds):
                        magnitudes[c, np.where(mags_map[c,:]<0)[0],:] = np.nan
            initial_m = magnitudes

        if starting_points > 1: #use multiple starting points
            filterwarnings('ignore', 'Convergence failed, estimation hitted the maximum ', )#will be the case but for a subset only hence ignore
            parameters = [initial_p]
            magnitudes = np.tile(initial_m, (starting_points+1, 1, 1, 1))
            if method == 'random':
                for _ in np.arange(starting_points):
                    proposal_p = np.zeros((n_conds,n_events + 1, 2)) * np.nan #by default nan for missing stages
                    for c in range(n_conds):
                        pars_cond = np.where(pars_map[c,:]>=0)[0]
                        n_stage_cond = len(pars_cond)
                        proposal_p[c,pars_cond,:] = self.gen_random_stages(n_stage_cond-1)
                        proposal_p[i][parameters_to_fix] = initial_p[0][parameters_to_fix]
                    parameters.append(proposal_p)
                
                proposals_m = self.gen_mags(n_events, starting_points, method='random', verbose=False)
                for i in range(starting_points):
                    proposal_m = proposals_m[i]
                    proposal_m = np.tile(proposal_m, (n_conds, 1,1))
                    for j in range(n_conds):
                        if (mags_map < 0).any(): #set missing mags to nan
                            proposal_m[c, np.where(mags_map[c,:]<0)[0],:] = np.nan
                        proposal_m[j][magnitudes_to_fix,:] = initial_m[0][magnitudes_to_fix,:]
                    magnitudes[i+1,:,:,:] = proposal_m
            else:
                raise ValueError('Unknown starting point method requested, use "random" ')
            
            if cpus > 1: 
                inputs = zip(magnitudes, parameters, itertools.repeat(maximization),
                        itertools.repeat(magnitudes_to_fix),itertools.repeat(parameters_to_fix), itertools.repeat(max_iteration), itertools.repeat(tolerance), itertools.repeat(min_iteration),
                        itertools.repeat(mags_map), itertools.repeat(pars_map), itertools.repeat(conds),itertools.repeat(1))
                with mp.Pool(processes=cpus) as pool:
                    if starting_points > 1:
                        estimates = list(tqdm(pool.imap(self._EM_star, inputs), total=len(magnitudes)))
                    else:
                        estimates = pool.starmap(self.EM, inputs)
 
            else:#avoids problems if called in an already parallel function
                estimates = []
                for pars, mags in zip(parameters, magnitudes):
                    estimates.append(self.EM(mags, maximization,\
                    magnitudes_to_fix, parameters_to_fix, max_iteration, tolerance, min_iteration,
                     pars, mags_map, pars_map, conds, 1))
                resetwarnings()
            
            lkhs_sp = [x[0] for x in estimates]
            mags_sp = [x[1] for x in estimates]
            pars_sp = [x[2] for x in estimates]
            eventprobs_sp = [x[3] for x in estimates]
            traces_sp = [x[4] for x in estimates]
            if return_max:
                max_lkhs = np.argmax(lkhs_sp)
                lkh = lkhs_sp[max_lkhs]
                mags = mags_sp[max_lkhs]
                pars = pars_sp[max_lkhs]
                eventprobs = eventprobs_sp[max_lkhs]
                traces = traces_sp[max_lkhs]
            else:
                lkh = lkhs_sp
                mags = mags_sp
                pars = pars_sp
                eventprobs = eventprobs_sp
                traces = np.zeros((len(lkh),  max([len(x) for x in traces_sp])))*np.nan
                for i, _i in enumerate(traces_sp):
                    traces[i, :len(_i)] = _i
            
        elif starting_points == 1:#informed starting point
            lkh, mags, pars, eventprobs, traces = self.EM(initial_m, initial_p, \
                                         maximization, magnitudes_to_fix, parameters_to_fix, \
                                         max_iteration, tolerance, min_iteration, 
                                         mags_map, pars_map, conds, 1)

        else:#uninitialized    
            if np.any(parameters)== None:
                parameters = np.tile([self.shape, self.mean_to_scale(np.mean(self.durations)/(n_events+1),self.shape)], (n_events+1,1))
                parameters = np.tile(parameters, (n_conds, 1, 1))  #broadcast across conditions                             
            
            if np.any(magnitudes)== None:
                magnitudes = np.zeros((n_events,self.n_dims), dtype=np.float64)
                magnitudes = np.tile(magnitudes, (n_conds, 1, 1)) #broadcast across conditions

            lkh, mags, pars, eventprobs, traces = self.EM(magnitudes, parameters, maximization, magnitudes_to_fix, parameters_to_fix, max_iteration, tolerance, min_iteration, mags_map, pars_map, conds, 1)
        
        #make output object
        xrlikelihoods = xr.DataArray(lkh , name="likelihoods")
        xrtraces = xr.DataArray(traces, dims=("em_iteration"), name="traces", coords={'em_iteration':range(len(traces))})
        
        xrparams = xr.DataArray(pars, dims=("condition","stage",'parameter'), name="parameters", 
                                 coords ={'condition':range(n_conds), 'stage':range(n_events+1), 'parameter':['shape','scale']})
        xrmags = xr.DataArray(mags, dims=("condition", "event", "component"), name="magnitudes",
                     coords={'condition':range(n_conds), 'event':range(n_events), "component":range(self.n_dims)})
        part, trial = self.coords['participant'].values, self.coords['trials'].values
        trial_x_part = xr.Coordinates.from_pandas_multiindex(MultiIndex.from_arrays([part,trial],
                                    names=('participant','trials')),'trial_x_participant')
        
        xreventprobs = xr.Dataset({'eventprobs': (('event', 'trial_x_participant','samples'),
                                             eventprobs.T)},
                                {'event': ('event', range(n_events)),
                                 'samples': ('samples', range(np.shape(eventprobs)[0])),
                                 'cond_x_participant': ('trial_x_participant', MultiIndex.from_arrays([part,conds],
                                 names=('participant','cond'))),
                                 'cond':('trial_x_participant',conds)})
        xreventprobs = xreventprobs.assign_coords(trial_x_part)
        xreventprobs = xreventprobs.transpose('trial_x_participant','samples','event')

        estimated = xr.merge((xrlikelihoods, xrparams, xrmags, xreventprobs, xrtraces))
        estimated.attrs['mags_map'] = mags_map
        estimated.attrs['pars_map'] = pars_map
        estimated.attrs['clabels'] = clabels
        estimated.attrs['conds_dict'] = conds_dict
        
        if verbose:
            print(f"parameters estimated for {n_events} events model")
        return estimated


    def _EM_star(self, args): #for tqdm usage
        return self.EM(*args)
    
    def EM(self, magnitudes, parameters, maximization=True, magnitudes_to_fix=None, parameters_to_fix=None, max_iteration=1e3, tolerance=1e-4, min_iteration=1, mags_map=None, pars_map=None,conds=None, cpus=1):  
        '''
        Expectation maximization function underlying fit

        parameters
        ----------
        n_events : int
            how many events are estimated
        magnitudes : ndarray
            2D ndarray n_events * components (or 3D iteration * n_events * n_components), initial conditions for events magnitudes. If magnitudes are estimated, the list provided is used as starting point,
            if magnitudes are fixed, magnitudes estimated will be the same as the one provided. When providing a list, magnitudes need to be in the same order
            _n_th magnitudes parameter is  used for the _n_th event
        parameters : list
            list of initial conditions for Gamma distribution parameters parameter (2D stage * parameter or 3D iteration * n_events * n_components). If parameters are estimated, the list provided is used as starting point,
            if parameters are fixed, parameters estimated will be the same as the one provided. When providing a list, stage need to be in the same order
            _n_th gamma parameter is  used for the _n_th stage
        maximization: bool
            If True (Default) perform the maximization phase in EM() otherwhise skip
        magnitudes_to_fix: bool
            To fix (True) or to estimate (False, default) the magnitudes of the channel contribution to the events
        parameters_to_fix : bool
            To fix (True) or to estimate (False, default) the parameters of the gammas
        max_iteration: int
            Maximum number of iteration for the expectation maximization
        tolerance: float
            Tolerance applied to the expectation maximization
        min_iteration: int
            Minimum number of iteration for the expectation maximization in the EM() function

        Returns
        -------
        lkh_prev : float
            Summed log probabilities
        magnitudes_prev : ndarray
            Magnitudes of the channel contribution to each event
        parameters_prev: ndarray
            parameterss for the gammas of each stage
        eventprobs_prev: ndarray
            Probabilities with shape max_samples*n_trials*n_events
        traces: ndarray
            Values of the log-likelihood for each EM iteration
        ''' 

        if mags_map is not None or pars_map is not None or conds is not None: #condition version
            assert mags_map is not None and pars_map is not None and conds is not None, 'Both magnitude and parameter maps need to be provided when doing EM based on conditions, as well as conditions.'
            assert mags_map.shape[0] == pars_map.shape[0], 'Both maps need to indicate the same number of conditions.'
            n_cond = mags_map.shape[0]
        else:
            n_cond = None
        
        if not isinstance(maximization, bool):#Backward compatibility with previous versions
            warn('Deprecated use of the threshold function, use maximization and tolerance arguments. Setting tolerance at 1 for compatibility')
            maximization = {1:True, 0:False}[maximization]
            if maximization:#Backward compatibility, equivalent to previous threshold = 1
                tolerance = 1
        
        null_stages = np.where(parameters[...,-1].flatten()<0)[0]
        wrong_shape = np.where(parameters[...,-2]!=self.shape)[0]
        if n_cond is None and len(null_stages)>0:
            raise ValueError(f'Wrong scale parameter input, provided scale parameter(s) {null_stages} should be positive but have value {parameters[...,-1].flatten()[null_stages]}')
        if n_cond is None and len(wrong_shape)>0:
            raise ValueError(f'Wrong shape parameter input, provided parameter(s) {wrong_shape} shape is {parameters[...,-2][wrong_shape]} but expected {self.shape}')
        
        initial_parameters = np.copy(parameters)
        initial_magnitudes = np.copy(magnitudes)
        n_events = magnitudes.shape[magnitudes.ndim-2]
        
        if n_cond is not None:
            lkh, eventprobs = self.estim_probs_conds(magnitudes, parameters, mags_map, pars_map, conds, cpus=cpus)
        else:
            lkh, eventprobs = self.estim_probs(magnitudes, parameters, n_events)

        traces = [lkh]
        i = 0
        if not maximization or n_events==0:
            lkh_prev = lkh
        else:
            lkh_prev = lkh
            while i < max_iteration :#Expectation-Maximization algorithm
                if i >= min_iteration and (np.isneginf(lkh) or tolerance > (lkh-lkh_prev)/np.abs(lkh_prev)):
                    break
                    #As long as new run gives better likelihood, go on  
                lkh_prev = lkh.copy()
                
                if n_cond is not None: #condition dependent
                    for c in range(n_cond): #get params/mags

                        mags_map_cond = np.where(mags_map[c,:]>=0)[0]
                        pars_map_cond = np.where(pars_map[c,:]>=0)[0]
                        epochs_cond = np.where(conds == c)[0]
                        
                        #get mags/pars by condition
                        magnitudes[c,mags_map_cond,:], parameters[c,pars_map_cond,:] = self.get_magnitudes_parameters_expectation(eventprobs[np.ix_(range(self.max_d),epochs_cond, mags_map_cond)], subset_epochs=epochs_cond)

                        magnitudes[c,magnitudes_to_fix,:] = initial_magnitudes[c,magnitudes_to_fix,:].copy()
                        parameters[c,parameters_to_fix,:] = initial_parameters[c,parameters_to_fix,:].copy()
                    
                    #set mags to mean if requested in map
                    for m in range(n_events):
                        for m_set in np.unique(mags_map[:,m]):
                            if m_set >= 0:
                                magnitudes[mags_map[:,m] == m_set,m,:] = np.mean(magnitudes[mags_map[:,m] == m_set,m,:],axis=0)

                    #set params to mean if requested in map
                    for p in range(n_events+1):
                        for p_set in np.unique(pars_map[:,p]):
                            if p_set >= 0:
                                parameters[pars_map[:,p] == p_set,p,:] = np.mean(parameters[pars_map[:,p] == p_set,p,:],axis=0)

                else: #general
                    magnitudes, parameters = self.get_magnitudes_parameters_expectation(eventprobs)
                    magnitudes[magnitudes_to_fix,:] = initial_magnitudes[magnitudes_to_fix,:].copy()
                    parameters[parameters_to_fix, :] = initial_parameters[parameters_to_fix,:].copy()

                if n_cond is not None:
                    lkh, eventprobs = self.estim_probs_conds(magnitudes, parameters, mags_map, pars_map, conds, cpus=cpus)
                else:
                    lkh, eventprobs = self.estim_probs(magnitudes, parameters, n_events)
                traces.append(lkh)
                i += 1

        if i == max_iteration:
            warn(f'Convergence failed, estimation hitted the maximum number of iteration ({int(max_iteration)})', RuntimeWarning)
        return lkh, magnitudes, parameters, eventprobs, np.array(traces)

    def get_magnitudes_parameters_expectation(self,eventprobs,subset_epochs=None):
        
        n_events = eventprobs.shape[2]
        n_trials = eventprobs.shape[1]
        if subset_epochs is None: #all trials
            subset_epochs = range(n_trials)

        magnitudes = np.zeros((n_events, self.n_dims))

        #Magnitudes from Expectation
        if self.em_method == "max": 
            event_times = np.zeros((n_events+1, n_trials))
            event_times[-1,:] = self.durations[subset_epochs]
        for event in range(n_events):
            if self.em_method == "max":
                #Take time point at maximum p() for each trial
                #Average channel activity at those points
                event_values = np.zeros((n_trials, self.n_dims))
                for trial in range(n_trials):
                    event_times[event,trial]  = np.argmax(eventprobs[:, trial, event])
                    event_values[trial] = self.events[self.starts[subset_epochs][trial] + int(event_times[event,trial])]
                magnitudes[event] = np.mean(event_values, axis=0)
            elif self.em_method == "mean":
                for comp in range(self.n_dims):
                    magnitudes[event,comp] = np.mean(np.sum( \
                        eventprobs[:,:,event]*self.data_matrix[:,subset_epochs,comp], axis=0))
                # scale cross-correlation with likelihood of the transition
                # sum by-trial these scaled activation for each transition events
                # average across trials
        
        #Gamma parameters from Expectation
        if self.em_method == "max":
            parameters = self.scale_parameters(eventprobs=None, n_events=n_events, averagepos=np.mean(event_times,axis=1))
        elif self.em_method == "mean":
            #calc averagepos here as mean_d can be condition dependent, whereas scale_parameters() assumes it's general
            event_times_mean = np.concatenate([np.arange(self.max_d) @ eventprobs.mean(axis=1), [np.mean(self.durations[subset_epochs])-1]])
            parameters = self.scale_parameters(eventprobs=None, n_events=n_events, averagepos=event_times_mean)                            

        return [magnitudes, parameters]

    def estim_probs(self, magnitudes, parameters, n_events=None, subset_epochs=None, lkh_only=False):
        '''
        parameters
        ----------
        magnitudes : ndarray
            2D ndarray n_events * components (or 3D iteration * n_events * n_components), initial conditions for events magnitudes. If magnitudes are estimated, the list provided is used as starting point,
            if magnitudes are fixed, magnitudes estimated will be the same as the one provided. When providing a list, magnitudes need to be in the same order
            _n_th magnitudes parameter is  used for the _n_th event
        parameters : list
            list of initial conditions for Gamma distribution parameters parameter (2D stage * parameter or 3D iteration * n_events * n_components). If parameters are estimated, the list provided is used as starting point,
            if parameters are fixed, parameters estimated will be the same as the one provided. When providing a list, stage need to be in the same order
            _n_th gamma parameter is  used for the _n_th stage
        n_events : int
            how many events are estimated
        lkh_only: bool
            Returning eventprobs (True) or not (False)
        
        Returns
        -------
        likelihood : float
            Summed log probabilities
        eventprobs : ndarray
            Probabilities with shape max_samples*n_trials*n_events
        '''
        if n_events is None:
            n_events = magnitudes.shape[0]
        n_stages = n_events+1

        if subset_epochs is not None:
            if len(subset_epochs) == self.n_trials: #boolean indices
                subset_epochs = np.where(subset_epochs)[0]
            n_trials = len(subset_epochs)
            durations = self.durations[subset_epochs]
            starts = self.starts[subset_epochs]
            ends = self.ends[subset_epochs]
        else:
            n_trials = self.n_trials
            durations = self.durations
            starts = self.starts
            ends = self.ends

        gains = np.zeros((self.n_samples, n_events), dtype=np.float64)
        for i in range(self.n_dims):
            # computes the gains, i.e. congruence between the pattern shape
            # and the data given the magnitudes of the sensors
            gains = gains + self.events[:,i][np.newaxis].T * magnitudes[:,i]-magnitudes[:,i]**2/2
        gains = np.exp(gains)
        probs = np.zeros([self.max_d, n_trials,n_events], dtype=np.float64) # prob per trial
        probs_b = np.zeros([self.max_d, n_trials,n_events], dtype=np.float64)# Sample and state reversed
        for trial in np.arange(n_trials):
            # Following assigns gain per trial to variable probs 
            probs[:durations[trial],trial,:] = \
                gains[starts[trial]:ends[trial]+1,:] 
            # Same but samples and events are reversed, this allows to compute
            # fwd and bwd in the same way in the following steps
            probs_b[:durations[trial],trial,:] = \
                gains[starts[trial]:ends[trial]+1,:][::-1,::-1]

        pmf = np.zeros([self.max_d, n_stages], dtype=np.float64) # Gamma pmf for each stage scale
        for stage in range(n_stages):
            pmf[:,stage] = self.distribution_pmf(parameters[stage,0], parameters[stage,1])
        pmf[:,1:-1] = np.concatenate([np.tile([0],(self.location,n_events-1)),pmf[self.location:,1:-1]])#all stages except first and last stages have a location (mainly to avoid overlap in cross-correlated signal)
        pmf_b = pmf[:,::-1] # Stage reversed gamma pmf, same order as prob_b

        if n_events > 0:
            forward = np.zeros((self.max_d, n_trials, n_events), dtype=np.float64)
            backward = np.zeros((self.max_d, n_trials, n_events), dtype=np.float64)
            # Computing forward and backward helper variable
            #  when stage = 0:
            forward[:,:,0] = np.tile(pmf[:,0][np.newaxis].T,\
                (1,n_trials))*probs[:,:,0] #first stage transition is p(B) * p(d)
            backward[:,:,0] = np.tile(pmf_b[:,0][np.newaxis].T,\
                        (1,n_trials)) #Reversed gamma (i.e. last stage) without probs as last event ends at time T

            for event in np.arange(1,n_events):#Following stage transitions integrate previous transitions
                add_b = backward[:,:,event-1]*probs_b[:,:,event-1]#Next stage in back
                for trial in np.arange(n_trials):
                    # convolution between gamma * gains at previous event and event
                    forward[:,trial,event] = self.convolution(forward[:,trial,event-1], pmf[:,event])[:self.max_d]
                    # same but backwards
                    backward[:,trial,event] = self.convolution(add_b[:,trial], pmf_b[:, event])[:self.max_d]
                forward[:,:,event] = forward[:,:,event]*probs[:,:,event]
            #re-arranging backward to the expected variable
            backward = backward[:,:,::-1]#undoes stage inversion
            for trial in np.arange(n_trials):#Undoes sample inversion
                backward[:durations[trial],trial,:] = \
                    backward[:durations[trial],trial,:][::-1]
            
            eventprobs = forward * backward
            eventprobs = np.clip(eventprobs, 0, None) #floating point precision error
            
            #eventprobs can be so low as to be 0, avoid dividing by 0
            #this only happens when magnitudes are 0 and gammas are randomly determined
            if (eventprobs.sum(axis=0) == 0).any() or (eventprobs[:,:,0].sum(axis=0) == 0).any(): 

                #set likelihood
                eventsums = eventprobs[:,:,0].sum(axis=0)
                eventsums[eventsums != 0] = np.log(eventsums[eventsums != 0])
                eventsums[eventsums == 0] = -np.inf
                likelihood = np.sum(eventsums)

                #set eventprobs, check if any are 0   
                eventsums = eventprobs.sum(axis=0)
                if (eventsums == 0).any():
                    for i in range(eventprobs.shape[0]):
                        eventprobs[i,:,:][eventsums == 0] = 0
                        eventprobs[i,:,:][eventsums != 0] = eventprobs[i,:,:][eventsums != 0] / eventsums[eventsums != 0]
                else:
                    eventprobs = eventprobs / eventprobs.sum(axis=0)

            else:

                likelihood = np.sum(np.log(eventprobs[:,:,0].sum(axis=0)))#sum over max_samples to avoid 0s in log
                eventprobs = eventprobs / eventprobs.sum(axis=0)
            #conversion to probabilities, divide each trial and state by the sum of the likelihood of the n points in a trial
            
        else:
            forward = np.zeros((self.max_d, n_trials), dtype=np.float64)
            backward = np.zeros((self.max_d, n_trials), dtype=np.float64)
            forward[:,:] = np.tile(pmf[:,0][np.newaxis].T,\
                (1,n_trials))
            backward[:,:] = np.tile(pmf_b[:,0][np.newaxis].T,\
                        (1,n_trials))
            for trial in np.arange(n_trials):#Undoes sample inversion
                backward[:durations[trial],trial] = \
                    backward[:durations[trial],trial][::-1]
            eventprobs = forward * backward
            likelihood = np.sum(np.log(eventprobs[:,:].sum(axis=0)))#sum over max_samples to avoid 0s in log
            eventprobs = eventprobs / eventprobs.sum(axis=0)

        if lkh_only:
            return likelihood
        else:
            return [likelihood, eventprobs]

    def estim_probs_conds(self, magnitudes, parameters, mags_map, pars_map, conds, lkh_only=False, cpus=1):
        '''
        parameters
        ----------
        magnitudes : ndarray
            2D ndarray n_events * components (or 3D iteration * n_events * n_components), initial conditions for events magnitudes. If magnitudes are estimated, the list provided is used as starting point,
            if magnitudes are fixed, magnitudes estimated will be the same as the one provided. When providing a list, magnitudes need to be in the same order
            _n_th magnitudes parameter is  used for the _n_th event
        parameters : list
            list of initial conditions for Gamma distribution parameters parameter (2D stage * parameter or 3D iteration * n_events * n_components). If parameters are estimated, the list provided is used as starting point,
            if parameters are fixed, parameters estimated will be the same as the one provided. When providing a list, stage need to be in the same order
            _n_th gamma parameter is  used for the _n_th stage
        n_events : int
            how many events are estimated
        lkh_only: bool
            Returning eventprobs (True) or not (False)
        
        Returns
        -------
        likelihood : float
            Summed log probabilities
        eventprobs : ndarray
            Probabilities with shape max_samples*n_trials*n_events
        '''

        n_conds = mags_map.shape[0]
        likes_events_cond = []

        if cpus > 1:
            with mp.Pool(processes=cpus) as pool:
                likes_events_cond = pool.starmap(self.estim_probs, 
                    zip([magnitudes[c, mags_map[c,:]>=0, :] for c in range(n_conds)], [parameters[c, pars_map[c,:]>=0, :] for c in range(n_conds)], itertools.repeat(None),
                        [conds == c for c in range(n_conds)], itertools.repeat(False)))
        else:
            for c in range(n_conds):
                magnitudes_cond = magnitudes[c, mags_map[c,:]>=0, :] #select existing magnitudes
                parameters_cond = parameters[c, pars_map[c,:]>=0, :] #select existing params
                likes_events_cond.append(self.estim_probs(magnitudes_cond, parameters_cond, subset_epochs = (conds == c)))

        likelihood = np.sum([x[0] for x in likes_events_cond])
        eventprobs = np.zeros((self.max_d, len(conds), mags_map.shape[1]))
        for c in range(n_conds):
            eventprobs[np.ix_(range(self.max_d), conds == c, mags_map[c,:]>=0)] = likes_events_cond[c][1]

        if lkh_only:
            return likelihood
        else:
            return [likelihood, eventprobs]

    def distribution_pmf(self, shape, scale):
        '''
        Returns PMF for a provided scipy disttribution with shape and scale, on a range from 0 to max_length 
        
        parameters
        ----------
        shape : float
            shape parameter
        scale : float
            scale parameter     
        Returns
        -------
        p : ndarray
            probabilty mass function for the distribution with given scale
        '''
        if scale == 0:
            warn('Convergence failed: one stage has been found to be null')
        p = self.pdf(np.arange(self.max_d), shape, scale=scale)
        p = p/np.sum(p)
        p[np.isnan(p)] = 0 #remove potential nans
        return p
    
    def scale_parameters(self, eventprobs=None, n_events=None, averagepos=None):
        '''
        Used for the re-estimation in the EM procdure. The likeliest location of 
        the event is computed from eventprobs. The scale parameter are then taken as the average 
        distance between the events
        
        parameters
        ----------
        eventprobs : ndarray
            [samples(max_d)*n_trials*n_events] = [max_d*trials*nTransition events]
        durations : ndarray
            1D array of trial length
        mags : ndarray
            2D ndarray components * nTransition events, initial conditions for events magnitudes
        shape : float
            shape parameter for the gamma, defaults to 2  
        
        Returns
        -------
        params : ndarray
            shape and scale for the gamma distributions
        '''
        params = np.zeros((n_events+1,2), dtype=np.float64)
        params[:,0] = self.shape
        params[:,1] = np.diff(averagepos, prepend=0)
        params[:,1] = [self.mean_to_scale(x[1],x[0]) for x in params]
        return params


    def backward_estimation(self,max_events=None, min_events=0, max_fit=None, max_starting_points=1, method="random", tolerance=1e-4, maximization=True, max_iteration=1e3):
        '''
        First read or estimate max_event solution then estimate max_event - 1 solution by 
        iteratively removing one of the event and pick the one with the highest 
        likelihood
        
        parameters
        ----------
        max_events : int
            Maximum number of events to be estimated, by default the output of hmp.models.hmp.compute_max_events()
        min_events : int
            The minimum number of events to be estimated
        max_fit : xarray
            To avoid re-estimating the model with maximum number of events it can be provided 
            with this arguments, defaults to None
        max_starting_points: int
            how many random starting points iteration to try for the model estimating the maximal number of events
        method: str
            What starting points generation method to use, 'random'or 'grid' (grid is not yet fully supported)
        tolerance: float
            Tolerance applied to the expectation maximization in the EM() function
        maximization: bool
            If True (Default) perform the maximization phase in EM() otherwhise skip
        max_iteration: int
            Maximum number of iteration for the expectation maximization in the EM() function
        '''
        if max_events is None and max_fit is None:
            max_events = self.compute_max_events()
        if not max_fit:
            if max_starting_points > 0:
                print(f'Estimating all solutions for maximal number of events ({max_events}) with 1 pre-defined starting point and {max_starting_points-1} {method} starting points')
            event_loo_results = [self.fit_single(max_events, starting_points=max_starting_points, method=method, verbose=False)]
        else:
            event_loo_results = [max_fit]
        max_events = event_loo_results[0].event.max().values+1

        for n_events in np.arange(max_events-1,min_events,-1):
            print(f'Estimating all solutions for {n_events} number of events')
                      
            flats_prev = event_loo_results[-1].dropna('stage').parameters.values
            mags_prev = event_loo_results[-1].dropna('event').magnitudes.values
            events_temp, pars_temp = [],[]
            
            for event in np.arange(n_events + 1):#creating all possible solutions
                
                events_temp.append(mags_prev[np.arange(n_events+1) != event,])
                
                temp_flat = np.copy(flats_prev)
                temp_flat[event,1] = temp_flat[event,1] + temp_flat[event+1,1] #combine two stages into one
                temp_flat = np.delete(temp_flat, event+1, axis=0)
                pars_temp.append(temp_flat)
            if self.cpus == 1:
                event_loo_likelihood_temp = []
                for i in range(len(events_temp)):
                    event_loo_likelihood_temp.append(self.fit_single(n_events, events_temp[i],pars_temp[i],tolerance=tolerance,max_iteration=max_iteration,maximization=maximization,verbose=False))
            else:
                inputs = zip(itertools.repeat(n_events), events_temp, pars_temp,\
                            itertools.repeat([]), itertools.repeat([]),\
                            itertools.repeat(tolerance), itertools.repeat(max_iteration), \
                            itertools.repeat(maximization), itertools.repeat(1),\
                            itertools.repeat(1),itertools.repeat('random'), itertools.repeat(True),\
                            itertools.repeat(False),itertools.repeat(1))
                with mp.Pool(processes=self.cpus) as pool:
                    event_loo_likelihood_temp = pool.starmap(self.fit_single, inputs)

            lkhs = [x.likelihoods.values for x in event_loo_likelihood_temp]
            event_loo_results.append(event_loo_likelihood_temp[np.nanargmax(lkhs)])

            #remove event_loo_likelihood
            del event_loo_likelihood_temp
            # Force garbage collection
            gc.collect()

        event_loo_results = xr.concat(event_loo_results, dim="n_events")
        event_loo_results = event_loo_results.assign_coords({"n_events": np.arange(max_events,min_events,-1)})
        return event_loo_results

    def compute_max_events(self):
        '''
        Compute the maximum possible number of events given event width and mean or minimum reaction time
        '''
        return int(np.rint(np.min(self.durations)//(self.location)))


    def event_times(self, eventprobs, mean=True):
         '''
         Compute event onset times based on event probabilities
         This function is mainly kept for compatibility with previous matlab applications
         parameters
         ----------
    
         Returns
         -------
         '''
         eventprobs = eventprobs.dropna('event', how="all")
         eventprobs = eventprobs.dropna('trial_x_participant', how="all")
         onsets = np.empty((len(eventprobs.trial_x_participant),len(eventprobs.event)+1))*np.nan
         i = 0
         for trial in eventprobs.trial_x_participant.dropna('trial_x_participant', how="all").values:
             onsets[i, :len(eventprobs.event)] = np.arange(self.max_d) @ eventprobs.sel(trial_x_participant=trial).data
             onsets = onsets - self.event_width_samples//2
             onsets[i, -1] = self.ends[i] - self.starts[i]
             i += 1
         if mean:
             return np.mean(onsets, axis=0)
         else:
             return onsets


    @staticmethod        
    def compute_times(init, estimates, duration=False, fill_value=None, mean=False, mean_in_participant=True, cumulative=False, add_rt=False, extra_dim=None, as_time=False, errorbars=None, method='max'):
        '''
        Compute the likeliest onset times for each event

        parameters
        ----------
        init : 
            Initialized HMP object 
        estimates : xr.Dataset
            Estimated instance of an HMP model 
        duration : bool
            Whether to compute onset times (False) or stage duration (True)
        fill_value : float | ndarray
            What value to fill for the first onset/duration.
        mean : bool 
            Whether to compute the mean (True) or return the single trial estimates
            Note that mean and errorbars cannot both be true.
        mean_in_partipant : bool
            Whether the mean is first computed within participant before calculating the overall mean.
        cumulative : bool
            Outputs stage duration (False) or time of onset of stages (True)
        add_rt : bool
            whether to append the last stage up to the RT
        extra_dim : str
            if string the times are averaged within that dimension
        as_time : bool
            if true, return time (ms) instead of samples
        errorbars : str
            calculate 95% confidence interval ('ci'), standard deviation ('std'),
            standard error ('se') on the times or durations, or None.
            Note that mean and errorbars cannot both be true.
        method : str
            if 'mean' takes the weigthed average of the event probabilities to compute the topologies, otherwise use the time point with 'max' probability
        Returns
        -------
        times : xr.DataArray
            Transition event onset or stage duration with trial_x_participant*event dimensions or only event dimension if mean = True
            Contains nans for missing stages.
        '''

        assert not(mean and errorbars is not None), 'Only one of mean and errorbars can be set.'

        event_shift = init.event_width_samples//2
        eventprobs = estimates.eventprobs.fillna(0).copy()
        if method == "max":
            times = eventprobs.argmax('samples') - event_shift #Most likely event location
        else:
            times = xr.dot(eventprobs, eventprobs.samples, dims='samples') - event_shift
        times = times.astype('float32')#needed for eventual addition of NANs
        #in case there is a single model, but there are empty stages at the end
        #this happens with selected model from backward estimation
        if 'n_events' in times.coords and len(times.shape) == 2:
            tmp = times.mean('trial_x_participant').values
            if tmp[-1] == -event_shift:
                filled_stages = np.where(tmp != -event_shift)[0]
                times = times[:, filled_stages]
        #set to nan if stage missing
        if extra_dim == 'condition':
            times_cond = times.groupby('cond').mean('trial_x_participant').values #take average to make sure it's not just 0 on the trial-level
            for c, e in np.argwhere(times_cond == -event_shift):
                times[times['cond']==c, e] = np.nan
        elif extra_dim == 'n_events':
            times_n_events = times.mean('trial_x_participant').values
            for x, e in np.argwhere(times_n_events == -event_shift):
                times[x,:,e] = np.nan
        if as_time:
            times = times * 1000/init.sfreq
        if duration:
            fill_value=0
        if fill_value != None:            
            added = xr.DataArray(np.repeat(fill_value,len(times.trial_x_participant))[np.newaxis,:],
                                 coords={'event':[0], 
                                         'trial_x_participant':times.trial_x_participant})
            times = times.assign_coords(event=times.event+1)
            times = times.combine_first(added)
        if add_rt:             
            rts = init.named_durations
            if as_time:
                rts = rts * 1000/init.sfreq
            rts = rts.assign_coords(event=int(times.event.max().values+1))
            rts = rts.expand_dims(dim="event")
            times = xr.concat([times, rts], dim='event')
            if extra_dim == 'n_events': #move rts inside the nans of the missing stages
                for e in times['n_events'].values:
                    tmp = np.squeeze(times.isel(n_events = times['n_events'] == e).values) #seems overly complicated, but is necessary
                    #identify first nan column
                    first_nan = np.where(np.isnan(np.mean(tmp,axis=0)))[0]
                    if len(first_nan) > 0:
                        first_nan = first_nan[0]
                        tmp[:,first_nan] = tmp[:,-1]
                        tmp[:,-1] = np.nan
                        times[times['n_events'] == e,:, :] = tmp
        if duration: #taking into account missing events, hence the ugly code
            times = times.rename({'event':'stage'})
            if not cumulative:
                if not extra_dim:
                    times = times.diff(dim='stage')
                elif extra_dim == 'condition': #by cond, ignore missing events
                    for c in np.unique(times['cond'].values):
                        tmp = times.isel(trial_x_participant = estimates['cond'] == c).values
                        #identify nan columns == missing events
                        missing_evts = np.where(np.isnan(np.mean(tmp,axis=0)))[0]
                        tmp = np.diff(np.delete(tmp, missing_evts, axis=1)) #remove 0 columns, calc difference
                        #insert nan columns (to maintain shape), 
                        for missing in missing_evts:
                            tmp = np.insert(tmp, missing-1, np.nan, axis=1)
                        #add extra column to match shape
                        tmp = np.hstack((tmp,np.tile(np.nan,(tmp.shape[0],1)))) 
                        times[estimates['cond'] == c, :] = tmp
                    times = times[:,:-1] #remove extra column
                elif extra_dim == 'n_events':
                    for e in times['n_events'].values:
                        tmp = np.squeeze(times.isel(n_events = times['n_events'] == e).values) #seems overly complicated, but is necessary
                        #identify nan columns == missing events
                        missing_evts = np.where(np.isnan(np.mean(tmp,axis=0)))[0]
                        tmp = np.diff(np.delete(tmp, missing_evts, axis=1)) #remove 0 columns, calc difference
                        #insert nan columns (to maintain shape), in contrast to above,
                        #here add columns at the end, as no actually 'missing' events
                        tmp = np.hstack((tmp,np.tile(np.nan,(tmp.shape[0],len(missing_evts))))) 
                        #add extra column to match shape
                        tmp = np.hstack((tmp,np.tile(np.nan,(tmp.shape[0],1)))) 
                        times[times['n_events'] == e,:, :] = tmp
                    times = times[:,:,:-1] #remove extra column
        
        if mean:
            if extra_dim == 'condition': #calculate mean only in trials of specific condition
                if mean_in_participant:
                    times = times.groupby('cond_x_participant').mean('trial_x_participant')
                    times = times.groupby('cond').mean('cond_x_participant')       
                else:
                    times = times.groupby('cond').mean('trial_x_participant')
            else:
                if mean_in_participant:
                    times = times.groupby('participant').mean('trial_x_participant').mean('participant')
                else:
                    times = times.mean('trial_x_participant')
        elif errorbars:
            if extra_dim == 'condition':
                errorbars_model = np.zeros((len(np.unique(times['cond'])), 2,times.shape[1]))
                if errorbars == 'ci':
                    for c in np.unique(times['cond']):
                        for st in range(times.shape[1]):
                            errorbars_model[c, :, st] = utils.compute_ci(times[times['cond'] == c, st].values)
                elif errorbars == 'std':
                    std_errs = times.groupby('cond').reduce(np.std, dim='trial_x_participant').values
                    for c in np.unique(times['cond']):
                        errorbars_model[c,:,:] = np.tile(std_errs[c,:], (2,1))
                elif errorbars == 'se':
                    se_errs = times.groupby('cond_x_participant').mean('trial_x_participant').groupby('cond').reduce(sem, dim='cond_x_participant').values
                    for c in np.unique(times['cond']):
                        errorbars_model[c,:,:] = np.tile(se_errs[c,:], (2,1))
            elif extra_dim == 'n_events':
                errorbars_model = np.zeros((times.shape[0], 2, times.shape[2]))
                if errorbars == 'ci':
                    for e in np.unique(times['n_events']):
                        for st in range(times.shape[2]):
                            errorbars_model[times['n_events']==e, :, st] = utils.compute_ci(np.squeeze(times[times['n_events']==e,:, st].values))
                elif errorbars == 'std':
                    std_errs = times.reduce(np.std, dim='trial_x_participant').values
                    for e in np.unique(times['n_events']):
                        errorbars_model[times['n_events']==e,:,:] = np.tile(std_errs[times['n_events']==e,:], (2,1))
                elif errorbars == 'se':
                    se_errs = times.groupby('participant').mean('trial_x_participant').groupby('n_events').reduce(sem, dim='participant', axis=0).values
                    for c in np.unique(times['cond']):
                        errorbars_model[c,:,:] = np.tile(se_errs[c,:], (2,1))
            else:
                if errorbars == 'ci':
                    errorbars_model = np.zeros((2,times.shape[1]))
                    for st in range(times.shape[1]):
                        errorbars_model[:, st] = utils.compute_ci(times[:,st].values)
                elif errorbars == 'std':
                    errorbars_model = np.tile(times.reduce(np.std, dim='trial_x_participant').values, (2,1))
                elif errorbars == 'se':
                    errorbars_model = np.tile(times.groupby('participant').mean('trial_x_participant').reduce(sem, dim='participant').values, (2,1))
            times = errorbars_model
        return times
   
    @staticmethod
    def compute_topologies(channels, estimated, init, extra_dim=None, mean=True, mean_in_participant=True, peak=True, method='max'):
        """
        Compute topologies for each trial. 
         
        parameters
        ----------
         	channels: xr.Dataset 
                Epoched data
         	estimated: xr.Dataset 
                estimated model parameters and event probabilities
         	init : 
                Initialized HMP object 
         	extra_dim: str 
                if True the topology is computed in the extra dimension
         	mean: bool 
                if True mean will be computed instead of single-trial channel activities
            mean_in_partipant : bool
                Whether the mean is first computed within participant before calculating the overall mean.
            peak : bool
                if true, return topology at peak of the event. If false, return topologies weighted by a normalized template.
             method : str
                 if 'mean' takes the weigthed average of the event probabilities to compute the topologies, otherwise use the time point with 'max' probability
         Returns
         -------
         	event_values: xr.DataArray
                array containing the values of each electrode at the most likely transition time
                contains nans for missing events
        """
        channels = channels.rename({'epochs':'trials'}).\
                          stack(trial_x_participant=['participant','trials']).data.fillna(0).drop_duplicates('trial_x_participant')
        estimated = estimated.eventprobs.fillna(0).copy()
        n_events = estimated.event.count().values
        n_trials = estimated.trial_x_participant.count().values
        n_channels = channels.channels.count().values

        channels = channels.sel(trial_x_participant=estimated.trial_x_participant) #subset to estimated

        if peak:
            peak_shift = np.argmax(init.template)
        else:
            normed_template = init.template/np.sum(init.template)

        event_shift = init.event_width_samples // 2
        if not extra_dim or extra_dim == 'condition': #also in the condition case, only one fit per trial
            if method == "max":
                times = estimated.argmax('samples') - event_shift #Most likely event location
            else:
                times = np.round(xr.dot(estimated, estimated.samples, dims='samples')) - event_shift
        
            event_values = np.zeros((n_channels,n_trials,n_events))
            for ev in range(n_events):
                for tr in range(n_trials):
                    samp = int(times.values[tr,ev])
                    if peak:
                        event_values[:,tr,ev] = channels.values[:,samp+peak_shift,tr]
                    else:
                        vals = channels.values[:,samp:samp+init.event_width_samples,tr]
                        event_values[:,tr,ev] = np.dot(vals, normed_template[:vals.shape[1]])          
                    
            event_values = xr.DataArray(event_values, 
                        dims = ["channels","trial_x_participant","event",],
                        coords={"trial_x_participant":estimated.trial_x_participant,
                                "event": estimated.event,
                                "channels":channels.channels
                        })
            event_values = event_values.transpose("trial_x_participant","event","channels") #to maintain previous behavior

            if not extra_dim:
                #set to nan if stage missing
                times = times.mean('trial_x_participant').values
                for e in np.argwhere(times == -event_shift):
                    event_values[:,e,:] = np.nan
                        
            if extra_dim == 'condition':
                #add coords
                event_values = event_values.assign_coords({'cond_x_participant': ('trial_x_participant', channels['cond_x_participant'].values),
                                            'cond': ('trial_x_participant', channels['cond'].values)})

                #set to nan if stage missing
                times = times.groupby('cond').mean('trial_x_participant').values
                for c, e in np.argwhere(times == -event_shift):
                    event_values[event_values['cond']==c,e,:] = np.nan

        elif extra_dim == 'n_events': #here we need values per fit
            n_dim = estimated[extra_dim].count().values
            event_values = np.zeros((n_dim, n_channels, n_trials, n_events))*np.nan
            for x in range(n_dim):
                if method == "max":
                    times = estimated[x].argmax('samples') - init.event_width_samples//2
                else:
                    times = np.round(xr.dot(estimated[x], estimated.samples, dims='samples')) - init.event_width_samples//2
                for ev in range(n_events):
                    for tr in range(n_trials):
                        samp = int(times.values[tr,ev])
                        if peak:
                            event_values[x,:,tr,ev] = channels.values[:,samp+peak_shift,tr]  
                        else:
                            vals = channels.values[:,samp:samp+init.event_width_samples,tr]
                            event_values[x,:,tr,ev] = np.dot(vals, normed_template[:vals.shape[1]]) 

                #set to nan if missing
                times = times.mean('trial_x_participant').values
                for e in np.argwhere(times == -event_shift):
                    event_values[x,:,:,e] = np.nan
                
            event_values = xr.DataArray(event_values, 
                    dims = [extra_dim, "channels", "trial_x_participant","event"],
                    coords={extra_dim:estimated[extra_dim],
                            "trial_x_participant":estimated.trial_x_participant,
                            "event": estimated.event,
                            "channels":channels.channels
                    })
            event_values = event_values.transpose(extra_dim, "trial_x_participant","event","channels") #to maintain previous behavior
        else:
            print('Unknown extra dimension')

        if mean:
            if extra_dim == 'condition': #calculate mean within condition trials
                if mean_in_participant:
                    tmp = event_values.groupby('cond_x_participant').mean('trial_x_participant')
                    tmp = tmp.assign_coords({'cond': ('cond_x_participant', [x[1] for x in tmp['cond_x_participant'].data])})
                    return tmp.groupby('cond').mean('cond_x_participant')
                else:
                    return event_values.groupby('cond').mean('trial_x_participant')
            else:
                if mean_in_participant:
                    return event_values.groupby('participant').mean('trial_x_participant').mean('participant')
                else:
                    return event_values.mean('trial_x_participant')
        else:
            return event_values

    def gen_random_stages(self, n_events):
        '''
        Returns random stage duration between 0 and mean RT by iteratively drawind sample from a 
        uniform distribution between the last stage duration (equal to 0 for first iteration) and 1.
        Last stage is equal to 1-previous stage duration.
        The stages are then scaled to the mean RT
        
        parameters
        ----------
        n_events : int
            how many events
        
        Returns
        -------
        random_stages : ndarray
            random partition between 0 and mean_d
        '''
        mean_d = int(self.mean_d)
        rnd_durations = np.zeros(n_events + 1)
        while any(rnd_durations < 2): #make sure they are at least 2 samples
            rnd_events = np.random.default_rng().integers(low = 0, high = mean_d, size = n_events) #n_events between 0 and mean_d
            rnd_events = np.sort(rnd_events)
            rnd_durations = np.hstack((rnd_events, mean_d)) - np.hstack((0, rnd_events))  #associated durations
        random_stages = np.array([[self.shape, self.mean_to_scale(x, self.shape)] for x in rnd_durations])
        return random_stages
    
    def gen_mags(self, n_events, n_samples=None, lower_bound=-1, upper_bound=1, method=None, size=3, decimate=False, verbose=True):
        """
        Generate magnitudes sampled on a grid with n_events combinations. 
        This is a generator function that can be used to generate a set of magnitudes for testing different starting point to the EM algorithm
         
        parameters
        ----------
         	 n_events: int
                Number of events in the HMP model
         	 n_samples: int
                Number of samples to generate ( default : len ( grid ))
         	 lower_bound: float 
                Lower bound of the grid ( default : - 1 )
         	 upper_bound: float
                Upper bound of the grid ( default : 1 )
         	 method: str
                Method for generating the magnitudes ( default : None )
         	 size: int
                Size of the grid ( default : 3 )
         	 decimate: bool 
                If True the number of samples will be decimated to the size of the grid ( default : False )
         	 verbose: bool
                If True the number of samples will be printed to standard output ( default : True )
         
         Returns
         -------
         	 List of n_samples magnitude ( s ) sampled on
        """
        '''
        Return magnitudes sampled on a grid with n points between lower_bound and upper_bound for the n_events. All combinations are tested
        '''
        from itertools import product    
        if n_samples == 1:
            grid = np.zeros((n_events, 1, self.n_dims))
        else: 
            grid = np.array([x for x in product(np.linspace(lower_bound,upper_bound,size), repeat=self.n_dims)])

        if verbose:
            print(f'Number of potential magnitudes: {len(grid)}') 

        if n_samples is None:
            n_samples = len(grid)
        if n_samples > len(grid):
            method='random'
        if decimate:
            n_samples = int(np.rint(len(grid) / decimate))
            
        if method is None and len(grid) != n_samples:
            sort_table = np.argsort(np.abs(np.sum(grid, axis=-1)), axis=0)[::-1]
            grid = grid[sort_table[:n_samples]]
            n_samples = len(grid)
            if verbose:
                print(f'Because of decimation {len(grid)} will be estimated.')
        gen_mags = np.zeros((n_events, n_samples, self.n_dims))
        for event in range(n_events):
            if method is None:
                gen_mags[event,:,:] = grid
            elif method == "random":
                gen_mags[event,:,:] = grid[np.random.choice(range(len(grid)), size=n_samples, replace=True)]
        gen_mags = np.transpose(gen_mags, axes=(1,0,2))
        return gen_mags
    
    def _grid_search(self, n_stages, n_points=None, verbose=True, start_time=0, end_time=None, iter_limit=np.inf, step=1, offset=None, method='slide'):
        '''
        This function decomposes the mean RT into a grid with points. Ideal case is to have a grid with one sample = one search point but the number
        of possibilities badly scales with the length of the RT and the number of stages. Therefore the iter_limit is used to select an optimal number
        of points in the grid with a given spacing. After having defined the grid, the function then generates all possible combination of 
        event placements within this grid. It is faster than using random points (both should converge) but depending on the mean RT and the number 
        of events to look for, the number of combination can be really large. 
        
        parameters
        ----------
        n_stages : int
            how many event to look for +1
        n_points: int 
            how many points to look for ( default : None ). If None the number of points will be chosen from the length of the time
        verbose: bool
            Output useful print (True, default)
        start_time: int
            at what time to start the grid
        end_time: int
            at what time to stop the grid
        iter_limit: int
            How many iteration max, if grid is longer the grid will be decimated
        step: 
            How many step, i.e. samples, between two iterations
        offset:
        method:


        Returns
        -------
        parameters : ndarray
            3D array with numper of possibilities * n_stages * 2 (gamma parameters)
        '''
        from itertools import combinations_with_replacement, permutations  
        from math import comb as binomcoeff
        import more_itertools as mit
        if offset is None:
            offset = 1
        start_time = int(start_time)
        if end_time is None:
            end_time = int(self.mean_d)
        duration = end_time-start_time
        if n_points is None:
            n_points = duration//step
            duration = step*n_points
        check_n_posibilities = binomcoeff(n_points-1, n_stages-1)
        if check_n_posibilities > iter_limit:
            while binomcoeff(n_points-1, n_stages-1) > iter_limit:
                n_points = n_points-1
            step = duration//n_points#same if no points removed in the previous step
            end_time = start_time+step*(n_points-1)#Rounding up to step size
            duration = end_time-start_time
        end_time = start_time+step*(n_points)#Rounding up to step size  
        grid = np.array([x for x in np.linspace(start_time+step, end_time-step, (duration//step))-start_time])#all possible durations
        grid = grid[grid < duration - ((n_stages-2)*step)]#In case of >2 stages avoid impossible durations, just to speed up
        grid = grid[grid >= offset]
        comb = np.array([x for x in combinations_with_replacement(grid, n_stages) if np.round(np.sum(x)) == np.round(duration)])#A bit bruteforce
        if len(comb)>0:
            new_comb = []
            for c in comb:
                new_comb.append(np.array(list(mit.distinct_permutations(c))))
            comb = np.vstack(new_comb)
            parameters = np.zeros((len(comb),n_stages,2), dtype=np.float64)
            for idx, y in enumerate(comb):
                parameters[idx, :, :] = [[self.shape, self.mean_to_scale(x,self.shape)] for x in y]
            if verbose:
                if check_n_posibilities > iter_limit:
                    print(f'Initial number of possibilities is {check_n_posibilities}.') 
                    print(f'Given the number of max iterations = {iter_limit}: fitting {len(comb)} models based on all \n possibilities from grid search with a spacing of {int(step)} samples and {n_points} points  \n and durations of {grid}.')
                else:
                    print(f'Fitting {len(comb)} models using grid search')
            if method == 'grid':
                return parameters
            else:
                return parameters[np.argsort(parameters[:,0,1]),:,:]
        else:
            raise ValueError(f'No combination found given length of the data {self.mean_d}, number of events {n_stages} and a max iteration of {iter_limit}')
    
    def sliding_event_mags(self, epoch_data, step=5, decimate_grid = False, cpu=1, plot=False, tolerance=1e-4, min_iteration=10,alpha=.05):
        """
        Use the sliding_event function with different initialized magnitudes values

        parameters
        ----------
            epoch_data: xr.Dataset 
                the epoched data
            step: int 
                The number of steps to be use when sliding form 0 to mean duration
            decimate_grid: int 
                If True the grid is decimated to the given number
            cpu: int 
                The number of CPUs to use
            plot: bool
                 If True draw a plot
            tolerance: float 
                Tolerance criterion for the EM
            min_iteration: int 
                minimum iteration for the EM
            alpha: float
                transparency option for the generated plots

        Returns
        -------
            lkhs: ndarray 
                likelihoods for each sliding_event with different magnitudes
            mags: ndarray 
                magnitudes values
            channels: ndarray 
                Channel activities during the likeliest transition times for each event
            times : ndarray 
                Likeliest time of transitions for each event 
        """

        grid = np.squeeze(self.gen_mags(1, decimate = decimate_grid,size=3)) #get magn grid
        n_iter = len(grid)

        #calculate estimates, returns lkhs, mags, times
        inputs = zip(itertools.repeat((12,3)), itertools.repeat(False), itertools.repeat('search'), 
                    grid, itertools.repeat(step), itertools.repeat(False), itertools.repeat(None),
                    itertools.repeat(False),itertools.repeat(False), itertools.repeat(1), itertools.repeat(tolerance),
                    itertools.repeat(min_iteration),itertools.repeat(self.em_method))
        with mp.Pool(processes=cpu) as pool:
            estimates = list(tqdm(pool.imap(self._sliding_event_star, inputs), total=len(grid)))
            
        #topo prep
        stacked_epoch_data = epoch_data.stack(trial_x_participant=('participant','epochs')).dropna('trial_x_participant',how='all').data.to_numpy() 
        n_electrodes, _, n_trials = stacked_epoch_data.shape
        shift = int(self.event_width_samples/2)

        #collect results
        max_run_len = max([len(x[0]) for x in estimates])
        lkhs = np.zeros((n_iter * max_run_len))*np.nan
        times = np.zeros((n_iter * max_run_len))*np.nan
        mags = np.zeros((n_iter * max_run_len, self.n_dims))*np.nan
        channels = np.zeros((n_iter * max_run_len, epoch_data.sizes['channels']))*np.nan

        for i, est in enumerate(estimates):
            idx = i * max_run_len
            lkhs[idx:(idx + len(est[0]))] = est[0]
            mags[idx:(idx + len(est[0])), :] = est[1].squeeze()
            times[idx:(idx + len(est[0]))] = np.mean(est[2],axis=1)  #these should NOT be shifted, later used to calc pars
            
            #get topos per estimation per trial, then average
            topos = np.zeros((n_electrodes, n_trials, len(est[0])))
            for j, times_per_event in enumerate(est[2]):
                for trial in range(n_trials):
                    topos[:,trial,j] = stacked_epoch_data[:, int(times_per_event[trial]) + shift, trial]
                
            channels[idx:(idx+len(est[0])), :] = np.nanmean(topos,axis=1).T

        mags = mags[~np.isnan(lkhs),:]
        channels = channels[~np.isnan(lkhs),:]
        times = times[~np.isnan(lkhs)]
        lkhs = lkhs[~np.isnan(lkhs)]

        if plot:
            _, ax = plt.subplots(1,1,figsize=(20,3))
            ax.plot(times, lkhs, '.', alpha=alpha)
            ax.set_xlim(0, self.mean_d)
        
        return lkhs, mags, channels, times

    def _sliding_event_star(self, args): #for tqdm usage
        return self.sliding_event(*args)
        
    def sliding_event(self, figsize=(12,3), verbose=True, method=None, magnitudes=None, step=1, show=True, ax=None, fix_mags=False, fix_pars=False, cpus=None, tolerance=1e-4, min_iteration=1,em_method='mean'):
        """
        This method outputs the likelihood and estimates of a 1 event HMP model with each sample, from 0 to the mean epoch duration, as starting point for the first stage.
         
        parameters
        ----------
             figsize: tuple
                Size of the figure in inches.
         	 verbose: bool
                If True the method will be printed to standard output.
         	 method: str
                The method to use for fitting the model.
         	 magnitudes: ndarray
                The starting point(s) for the magnitudes
         	 step: float
                The step size for the slide between 0 and mean duration (step 2 = test every other sample)
         	 show: bool
                If True the plot will be shown.
         	 ax: plt.subplots
                The axis to plot to.
         	 fix_mags: bool
                If True the magnitudes will be fixed in the EM()
         	 fix_pars: bool
                If True the scale will be fixed in the EM()
         	 cpus: int
                The number of CPUs to use for the fitting process.
             tolerance: float
                Tolerance criterion for the EM
             min_iteration: int
                minimum iteration for the EM
         
        Returns
        -------
            lkhs: ndarray
                likelihoods for each sliding_event with different magnitudes
            mags: ndarray
                magnitudes values
        """       
        parameters = self._grid_search(2, verbose=verbose, step=step)#Looking for all possibilities with one event
        if method is None or magnitudes is None:
            magnitudes = np.zeros((len(parameters),1, self.n_dims), dtype=np.float64)
            maximization = True
            if fix_mags:
                mags_to_fix = [0]
            else:        
                mags_to_fix = []
            ls = '-'
        else:
            magnitudes = np.tile(magnitudes, (len(parameters),1,1))
            if fix_mags:
                maximization = False
                mags_to_fix = [0]
                ls = '-'
            else:
                maximization = True
                mags_to_fix = []
                ls = '.'
        if fix_pars:
            pars_to_fix = [0,1]
        else: 
            pars_to_fix = []
            ls = 'o'
        lkhs_init, mags_init, pars_init, times_init = self._estimate_single_event(magnitudes, parameters, pars_to_fix, mags_to_fix, maximization, cpus, tolerance=tolerance,min_iteration=min_iteration,em_method=em_method)
        
        if verbose:
            if ax is None:
                 _, ax = plt.subplots(figsize=figsize, dpi=100)
            ax.plot(self.scale_to_mean(pars_init[:,0,1],self.shape), lkhs_init, ls)
            plt.ylabel('Log-likelihood')
            plt.xlabel('Sample number')
            if show:
                plt.show()        
        if method is None:
            #pars, mags, lkhs = pars_init, mags_init, lkhs_init
            return ax
        else:
            return lkhs_init, mags_init, times_init
        
    def _estimate_single_event(self, magnitudes, parameters, parameters_to_fix, magnitudes_to_fix, maximization, cpus, max_iteration=1e2, tolerance=1e-4,min_iteration=1,em_method='mean'):
        filterwarnings('ignore', 'Convergence failed, estimation hitted the maximum ', )#will be the case but for a subset only hence ignore
        if cpus is None:
            cpus = self.cpus
        if cpus > 1:
            if np.shape(magnitudes) == 2:
                magnitudes = np.tile(magnitudes, (len(parameters), 1, 1))
            with mp.Pool(processes=self.cpus) as pool:
                estimates = pool.starmap(self.EM, 
                    zip(magnitudes, parameters, 
                        itertools.repeat(maximization), itertools.repeat(magnitudes_to_fix), 
                        itertools.repeat(parameters_to_fix), itertools.repeat(max_iteration),
                        itertools.repeat(tolerance), itertools.repeat(min_iteration)))
        else:
            estimates = []
            for pars, mags in zip(parameters, magnitudes):
                estimates.append(self.EM(mags, pars, maximization, magnitudes_to_fix, parameters_to_fix, max_iteration, tolerance, min_iteration=min_iteration))
        lkhs_sp = np.array([x[0] for x in estimates])
        mags_sp = np.array([x[1] for x in estimates])
        pars_sp = np.array([x[2] for x in estimates])

        #calc expected time
        if em_method == "max":
            times_sp = np.array([np.argmax(np.squeeze(x[3]),axis=0) for x in estimates]) - self.event_width_samples//2
        else:
            times_sp = np.array([np.rint(np.dot(np.squeeze(x[3]).T, np.arange(x[3].shape[0]))) for x in estimates]) - self.event_width_samples//2

        resetwarnings()
        return lkhs_sp, mags_sp, pars_sp, times_sp
    
    def fit(self, step=1, verbose=True, end=None, trace=False, fix_iter=False, max_iterations=1e3, tolerance=1e-3, grid_points=1, cpus=None, diagnostic=False, min_iteration=1, decimate=None, start=1):
        """
         Instead of fitting an n event model this method starts by fitting a 1 event model (two stages) using each sample from the time 0 (stimulus onset) to the mean RT. 
         Therefore it tests for the landing point of the expectation maximization algorithm given each sample as starting point and the likelihood associated with this landing point. 
         As soon as a starting points reaches the convergence criterion, the function fits an n+1 event model and uses the next samples in the RT as starting point for the following event
         
        parameters
        ----------
         	 step: float
                The size of the step from 0 to the mean RT
         	 verbose: bool 
                If True print information about the fit
         	 end: int
                The maximum number of samples to explore within each trial
         	 trace: bool 
                If True keep the scale and magnitudes parameters for each iteration
         	 fix_iter: bool 
                If True fix magnitudes and scale parameters for the previously found events
         	 max_iterations: int
                The maximum number of iteration in the EM() function
         	 tolerance: float
                The tolerance used for the convergence in the EM() function
         	 grid_points: int
                The number of grid points to use when testing for different magnitudes
         	 cpus: int
                The number of CPUs to use
         	 diagnostic: bool
                If True print a diagnostic plot of the EM traces for each iteration
         	 min_iteration: int 
                The minimum number of iterations for the EM() function
         	 decimate: int 
                If not None, decimate the grid search on magnitudes by the int provided
             start: float
                 Where to start the starting point search, for some reason values below 10 samples produce unexpected results
         
         Returns: 
         	 A tuple containing the fitted parameters and the fitted
        """
        if cpus is None:
            cpus = self.cpus
        if end is None:
            end = self.mean_d
        if verbose and decimate is not None:#Just for printing the info
             self.gen_mags(1, decimate=decimate, verbose=True)
        max_event_n = self.compute_max_events()
        n_points = int(np.rint(end)//step)
        if diagnostic:
            cycol = cycle(default_colors)
        pbar = tqdm(total = int(np.rint(end)))#progress bar
        n_events, j, time = 0,start,0

        #Init pars
        pars = np.zeros((max_event_n+1,2))
        pars[:,0] = self.shape #gamma parameters during estimation, shape x scale
        pars_prop = pars[:n_events+2].copy()
        pars_prop[0,1] = self.mean_to_scale(j,self.shape)#initialize gamma_parameters
        last_stage = self.mean_to_scale(end-j,self.shape)
        #Init mags
        mags = np.zeros((max_event_n, self.n_dims)) #mags during estimation
        i = 0
        mags_props = np.zeros((1, 1, self.n_dims))
        lkh_prev = self.fit_single(n_events+1, parameters_to_fix=[0,1], magnitudes_to_fix=0, verbose=False).likelihoods.values
        lkhs = self.sliding_event(fix_pars=True, fix_mags=True, method='max', verbose=False)[0]        
        delta = np.max(lkhs) - np.min(lkhs) + 1
        while self.scale_to_mean(last_stage, self.shape) >= self.event_width_samples and n_events < max_event_n-1:
            last_stage = self.mean_to_scale(end, self.shape) - np.sum(pars_prop[:n_events+1,1])
            pars_prop[n_events+1,1] = last_stage
            prev_time = time
            if fix_iter:
                to_fix = [range(n_events-1)]
            else: to_fix = []
            #Generate a grid of magnitudes as proposition 
            if decimate is None:
                mags_props = self.gen_mags(n_events+1, n_samples=grid_points, verbose=False)
            else:
                mags_props = self.gen_mags(n_events+1, decimate=decimate, verbose=False)
            #replave eventual event already found
            mags_props[:,:n_events,:] = np.tile(mags[:n_events,:], (len(mags_props), 1, 1))
            #estimate all grid_points models while fixing previous found events
            solutions = self.fit_single(n_events+1, mags_props, pars_prop, to_fix, to_fix[:-1],\
                            return_max=True, verbose=False, cpus=1,\
                            min_iteration=min_iteration, tolerance=tolerance)
            if diagnostic:#Diagnostic plot
                plt.plot(solutions.traces.T, alpha=.3, c='k')
            if solutions.likelihoods - lkh_prev > delta:#and np.diff(solutions.traces[-2:]) > 0:#Success
                lkh_prev = solutions.likelihoods.values
                if diagnostic:#Diagnostic plot
                    color = next(cycol)
                    plt.plot(solutions.traces.T, c=color, label=f'Iteration {i}')
                mags[:n_events+1], pars[:n_events+2] = solutions.magnitudes.values,\
                    solutions.parameters.values
                n_events += 1
                j = 1
                if verbose:
                    print(f'Transition event {n_events} found around sample {int(np.round(self.scale_to_mean(np.sum(pars[:n_events,1]), self.shape)))}')
            j += 1
            i += 1
            #New parameter proposition
            pars_prop = pars[:n_events+2].copy()
            pars_prop[n_events,1] = self.mean_to_scale(step*j, self.shape)
            time = int(np.round(self.scale_to_mean(np.sum(pars[:n_events,1]), self.shape)))
            pbar.update(int(np.rint(time-prev_time)))
        pbar.update(int(np.round(np.rint(end))-np.rint(time)))
        if diagnostic:
            plt.ylabel('Log-likelihood')
            plt.xlabel('EM iteration')
            plt.legend()
            plt.show()
        mags = mags[:n_events, :]
        pars = pars[:n_events+1, :]
        if n_events > 0: 
            fit = self.fit_single(n_events, parameters=pars, magnitudes=mags, verbose=verbose, cpus=1)
        else:
            warn('Failed to find more than two stages, returning None')
            fit = None#self.fit_single(n_events+1, verbose=verbose)
        return fit
    
