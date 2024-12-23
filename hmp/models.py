'''

'''

import numpy as np
import xarray as xr
import multiprocessing as mp
import itertools
from pandas import MultiIndex
from warnings import warn, filterwarnings, resetwarnings
import matplotlib.pyplot as plt
from hmp import utils
from scipy.signal import correlate
from itertools import cycle, product
from scipy.stats import norm as norm_pval 
import gc

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

default_colors =  ['cornflowerblue','indianred','orange','darkblue','darkgreen','gold', 'brown']

class hmp:
    
    def __init__(self, data, sfreq=None, cpus=1, event_width=50, shape=2, template=None, location=None, distribution='gamma'):
        '''
        This function intializes an HMP model by providing the data, the expected probability distribution for the by-trial variation in stage onset, and the expected duration of the transition event.

        parameters
        ----------
        data : xr.Dataset
            xr.Dataset obtained through the hmp.utils.transform_data() function
        sfreq : float
            (optional) Sampling frequency of the signal if not provided, inferred from the epoch_data
        cpus: int
            How many cpus to use for the functions`using multiprocessing`
        event_width : float
            width of events in milliselevels, by default 50 ms.
        shape: float
            shape of the probability distributions of the by-trial stage onset (one shape for all stages)
        template: ndarray
            Expected shape for the transition event used in the cross-correlation, should be a vector of values capturing the expected shape over the sampling frequency of the data. If None, the template is created as a half-sine shape with a frequency derived from the event_width argument
        location : float
            Minimum duration between events in samples. Default is the event_width.
        distribution : str
            Probability distribution for the by-trial onset of stages can be one of 'gamma','lognormal','wald', or 'weibull'
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
            self.location = int(self.event_width / self.steps)
        else:
            self.location = int(np.rint(location))
        if template is None:
            self.template = self._event_shape()
        else: 
            self.template = template
        # compute sequence durations based on number of samples
        durations = data.unstack().sel(component=0).rename({'epochs':'trials'})\
            .stack(trial_x_participant=['participant','trials']).dropna(dim="trial_x_participant",\
            how="all").groupby('trial_x_participant').count(dim="samples").cumsum().squeeze()
        
        if durations.trial_x_participant.count() > 1:
            dur_dropped_na = durations.dropna("trial_x_participant")
            starts = np.roll(dur_dropped_na.data, 1)
            starts[0] = 0
            ends = dur_dropped_na.data-1
            self.named_durations =  durations.dropna("trial_x_participant") - durations.dropna("trial_x_participant").shift(trial_x_participant=1, fill_value=0)
            self.coords = durations.reset_index('trial_x_participant').coords
        else: 
            dur_dropped_na = durations
            starts = np.array([0])
            ends = np.array([dur_dropped_na.data-1])
            self.named_durations = durations
            self.coords = durations.coords
            
        self.starts = starts
        self.ends = ends
        self.durations =  self.ends-self.starts+1
        self.mean_d = self.durations.mean()
        self.n_trials = durations.trial_x_participant.count().values
        self.cpus = cpus
        self.n_samples, self.n_dims = np.shape(data.data.T)
        self.crosscorr = self.cross_correlation(data.data.T)# Equation 1 in 2024 paper
        self.max_d = self.durations.max()
        self.trial_coords = data.unstack().sel(component=0,samples=0).rename({'epochs':'trials'}).\
            stack(trial_x_participant=['participant','trials']).dropna(dim="trial_x_participant",how="all").coords
    
    def _event_shape(self):
        '''
        Computes the template of a half-sine (event) with given frequency f and sampling frequency
        Equations in section 2.4 in the 2024 paper
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
        events = np.zeros(data.shape)
        for trial in range(self.n_trials):#avoids confusion of gains between trials
            for dim in np.arange(self.n_dims):
                events[self.starts[trial]:self.ends[trial]+1,dim] = correlate(data[self.starts[trial]:self.ends[trial]+1, dim], self.template, mode='same', method='direct')
        return events

    def fit_n(self, n_events=None, magnitudes=None, parameters=None, parameters_to_fix=None, 
                   magnitudes_to_fix=None, tolerance=1e-4, max_iteration=1e3, maximization=True, min_iteration=1,
                   starting_points=1, return_max=True, verbose=True, cpus=None,
                   mags_map=None, pars_map=None, level_dict=None):
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
        return_max: bool
            In the case of multiple starting points, dictates whether to only return the max loglikelihood model (True, default) or all of the models (False)
        verbose: bool
            True displays output useful for debugging, recommended for first use
        cpus: int
            number of cores to use in the multiprocessing functions
        mags_map: 2D nd_array n_level * n_events indicating which magnitudes are shared between levels.
        pars_map: 2D nd_array n_level * n_stages indicating which parameters are shared between levels.
        levels: dict | list
            if one level, use a dict with the name in the metadata and a list of the levels in the same
            order as the rows of the map(s). E.g., {'cue': ['SP', 'AC']}
            if multiple levels need to be crossed, use a list of dictionaries per level. E.g.,
            [{'cue': ['SP', 'AC',]}, {'resp': ['left', 'right']}]. These are crossed by repeating
            the first level as many times as there are levels in the selevel level. E.g., SP-left, 
            SP-right, AC-left, AC-right.
        '''
        # A dict containing all the info we want to keep, populated along the func
        infos_to_store = {}
        infos_to_store['sfreq'] = self.sfreq
        infos_to_store['event_width_samples'] = self.event_width_samples
        infos_to_store['tolerance'] = tolerance
        infos_to_store['maximization'] = int(maximization)
        
        if n_events is None:
            if parameters is not None:
                n_events = len(parameters)-1   
            elif magnitudes is not None:
                n_events = len(magnitudes)
            else:
                raise ValueError('The fit_n() function needs to be provided with a number of expected transition events')
        assert n_events <= self.compute_max_events(), f'{n_events} events do not fit given the minimum duration of {min(self.durations)} and a location of {self.location}'

        if level_dict is None:
            n_levels, levels, = 1, np.zeros(self.n_trials)
            pars_map, mags_map = np.zeros((1,n_events+1)), np.zeros((1,n_events))
        else:
            n_levels, levels, clabels, pars_map, mags_map = \
                self._level_constructor(magnitudes, parameters, mags_map, pars_map, level_dict, verbose)
            infos_to_store['mags_map'] = mags_map
            infos_to_store['pars_map'] = pars_map
            infos_to_store['clabels'] = clabels
            infos_to_store['level_dict'] = level_dict
        if verbose:
            if parameters is None:
                print(f'Estimating {n_events} events model with {starting_points} starting point(s)')
            else:
                print(f'Estimating {n_events} events model')
        if cpus is None:
            cpus = self.cpus      
            
        #Formatting parameters
        if isinstance(parameters, (xr.DataArray,xr.Dataset)):
            parameters = parameters.dropna(dim='stage').values
        if isinstance(magnitudes, (xr.DataArray,xr.Dataset)):
            magnitudes = magnitudes.dropna(dim='event').values  
        if isinstance(magnitudes, np.ndarray):
            magnitudes = magnitudes.copy()
        if isinstance(parameters, np.ndarray):
            parameters = parameters.copy()       
        if parameters_to_fix is None: 
            parameters_to_fix=[]
        else:
            infos_to_store['parameters_to_fix'] = parameters_to_fix
        if magnitudes_to_fix is None: 
            magnitudes_to_fix=[]
        else:
            infos_to_store['magnitudes_to_fix'] = magnitudes_to_fix
        
        if parameters is None:
            parameters = np.zeros((n_levels,n_events + 1, 2)) * np.nan #by default nan for missing stages
            for c in range(n_levels):
                pars_level = np.where(pars_map[c,:]>=0)[0]
                n_stage_level = len(pars_level)
                # by default starting point is to split the average duration in equal bins
                parameters[c,pars_level,:] = np.tile([self.shape, self.mean_to_scale(np.mean(self.durations[levels==c])/(n_stage_level),self.shape)], (n_stage_level,1))
        else:
            infos_to_store['sp_parameters'] = parameters
            if len(np.shape(parameters)) == 2: #broadcast provided parameters across levels
                parameters = np.tile(parameters, (n_levels, 1, 1))
            assert parameters.shape[1] == n_events + 1, f'Provided parameters ({ parameters.shape[1]} should match number of stages {n_events + 1}'

            #set params missing stages to nan to make it obvious in the results
            if (pars_map < 0).any():
                for c in range(n_levels):
                    parameters[c, np.where(pars_map[c,:]<0)[0],:] = np.nan
            
        if magnitudes is None:
            # By defaults mags are initiated to 0
            magnitudes = np.zeros((n_levels, n_events, self.n_dims), dtype=np.float64)
            if (mags_map < 0).any(): #set missing mags to nan
                for c in range(n_levels):
                    magnitudes[c, np.where(mags_map[c,:]<0)[0],:] = np.nan
        else:
            infos_to_store['sp_magnitudes'] = magnitudes
            if len(np.shape(magnitudes)) == 2: #broadcast provided magnitudes across levels
                magnitudes = np.tile(magnitudes, (n_levels, 1, 1))
            assert magnitudes.shape[1] == n_events, 'Provided magnitudes should match number of events in magnitudes map'
            
            #set mags missing events to nan to make it obvious in the results
            if (mags_map < 0).any():
                for c in range(n_levels):
                    magnitudes[c, np.where(mags_map[c,:]<0)[0],:] = np.nan
        initial_p = parameters
        initial_m = magnitudes
        parameters = [initial_p]
        magnitudes = np.tile(initial_m, (starting_points+1, 1, 1, 1))
        if starting_points > 1:
            infos_to_store['starting_points'] = starting_points
            for _ in np.arange(starting_points):
                proposal_p = np.zeros((n_levels,n_events + 1, 2)) * np.nan #by default nan for missing stages
                for c in range(n_levels):
                    pars_level = np.where(pars_map[c,:]>=0)[0]
                    n_stage_level = len(pars_level)
                    proposal_p[c,pars_level,:] = self.gen_random_stages(n_stage_level-1)
                    proposal_p[c,parameters_to_fix,:] = initial_p[0, parameters_to_fix]
                parameters.append(proposal_p)
            parameters = np.array(parameters)
            
        if cpus > 1: 
            inputs = zip(magnitudes, parameters, itertools.repeat(maximization),
                    itertools.repeat(magnitudes_to_fix),itertools.repeat(parameters_to_fix), itertools.repeat(max_iteration), itertools.repeat(tolerance), itertools.repeat(min_iteration),
                    itertools.repeat(mags_map), itertools.repeat(pars_map), itertools.repeat(levels),itertools.repeat(1))
            with mp.Pool(processes=cpus) as pool:
                if starting_points > 1:
                    estimates = list(tqdm(pool.imap(self._EM_star, inputs), total=len(magnitudes)))
                else:
                    estimates = pool.starmap(self.EM, inputs)

        else:#avoids problems if called in an already parallel function
            estimates = []
            for pars, mags in zip(parameters, magnitudes):
                estimates.append(self.EM(mags, pars, maximization,\
                magnitudes_to_fix, parameters_to_fix, max_iteration, tolerance, min_iteration,
                 mags_map, pars_map, levels, 1))
            resetwarnings()
        
        lkhs_sp = [x[0] for x in estimates]
        mags_sp = [x[1] for x in estimates]
        pars_sp = [x[2] for x in estimates]
        eventprobs_sp = [x[3] for x in estimates]
        traces_sp = [x[4] for x in estimates]
        param_dev_sp = [x[5] for x in estimates]
        if starting_points > 1 and return_max:
            max_lkhs = np.argmax(lkhs_sp)
            lkhs_sp = [lkhs_sp[max_lkhs]]
            mags_sp = [mags_sp[max_lkhs]]
            pars_sp = [pars_sp[max_lkhs]]
            eventprobs_sp = [eventprobs_sp[max_lkhs]]
            traces_sp = [traces_sp[max_lkhs]]
            param_dev_sp = [param_dev_sp[max_lkhs]]
            starting_points = 1

        #make output object
        estimated = []
        for sp in range(starting_points):
            xrlikelihoods = xr.DataArray(lkhs_sp[sp] , name="loglikelihood")
            xrtraces = xr.DataArray(traces_sp[sp], dims=("em_iteration"), name="traces",
                                    coords={'em_iteration':range(len(traces_sp[sp]))})
            xrparam_dev = xr.DataArray(param_dev_sp[sp], dims=("em_iteration","level","stage",'parameter'), 
                                       name="param_dev", coords=[range(len(param_dev_sp[sp])), range(n_levels), 
                                       range(n_events+1), ['shape','scale']])               
            xrparams = xr.DataArray(pars_sp[sp], dims=("level","stage",'parameter'), name="parameters", 
                                    coords ={'level':range(n_levels), 'stage':range(n_events+1), 'parameter':['shape','scale']})
            xrmags = xr.DataArray(mags_sp[sp], dims=("level", "event", "component"), name="magnitudes",
                         coords={'level':range(n_levels), 'event':range(n_events), "component":range(self.n_dims)})
            part, trial = self.coords['participant'].values, self.coords['trials'].values
            trial_x_part = xr.Coordinates.from_pandas_multiindex(MultiIndex.from_arrays([part,trial],
                                    names=('participant','trials')),'trial_x_participant')
            xreventprobs = xr.Dataset({'eventprobs': (('event', 'trial_x_participant','samples'),
                                             eventprobs_sp[sp].T)},
                                {'event': ('event', range(n_events)),
                                 'samples': ('samples', range(np.shape(eventprobs_sp[sp])[0]))})
            xreventprobs = xreventprobs.assign_coords(trial_x_part)
            if n_levels > 1:
                xreventprobs = xreventprobs.assign_coords(levels=("trial_x_participant", levels))
            xreventprobs = xreventprobs.transpose('trial_x_participant','samples','event')
    
            estimated.append(xr.merge((xrlikelihoods, xrparams, xrmags, xreventprobs, xrtraces, xrparam_dev)))

        if starting_points > 1:
            estimated = xr.concat(estimated, 'starting_points')
            estimated = estimated.assign_coords(starting_points=("starting_points", np.arange(starting_points)))
        else:
            estimated = estimated[0]

        # Adding infos
        estimated = estimated.assign_coords(rts=("trial_x_participant", self.named_durations.data))

        for x, y in infos_to_store.items():
            estimated.attrs[x] = y

        if n_levels == 1:
            estimated = estimated.squeeze(dim='level')
            # Drops empty coords incuced by squeeze
            estimated = estimated.drop_vars(lambda x: [v for v, da in x.coords.items() if not da.dims])

        if verbose:
            print(f"parameters estimated for {n_events} events model")
        return estimated
    
    def _level_constructor(self, magnitudes, parameters, mags_map, pars_map, level_dict, verbose):
        '''
        Adapt model to levels
        '''
        ## levels
        assert isinstance(level_dict, dict) , 'levels have to be specified as a dictionary'

        #collect level names, levels, and trial coding
        level_names = []
        level_mods = []
        level_trials = []
        for level in level_dict.keys():
            level_names.append(level)
            level_mods.append(level_dict[level])
            level_trials.append(self.trial_coords[level])
            if verbose:
                print('Level \"' + level_names[-1] + '\" analyzed, with levels:', level_mods[-1])

        level_mods = list(product(*level_mods))
        level_mods = np.array(level_mods, dtype=object)
        print(level_mods)
        n_levels = len(level_mods)

        #build level array with digit indicating the combined levels
        level_trials = np.vstack(level_trials).T
        levels = np.zeros((level_trials.shape[0])) * np.nan
        if verbose:
            print('\nCoded as follows: ')
        for i, mod in enumerate(level_mods):
            assert len(np.where((level_trials == mod).all(axis=1))[0]) > 0, f'Modality {mod} of level does not occur in the data'
            levels[np.where((level_trials == mod).all(axis=1))] = i
            if verbose:
                print(str(i) + ': ' + str(level))
        levels=np.int8(levels)
        clabels = {'level ' + str(level_names): level_mods}

        #check maps
        n_levels_mags = 0 if mags_map is None else mags_map.shape[0]
        n_levels_pars = 0 if pars_map is None else pars_map.shape[0]
        if n_levels_mags > 0 and n_levels_pars > 0: #either both maps should have the same number of levels, or 0
            assert n_levels_mags == n_levels_pars, 'magnitude and parameters maps have to indicate the same number of levels'
            #make sure nr of events correspond per row
            for c in range(n_levels):
                assert sum(mags_map[c,:] >= 0) + 1 == sum(pars_map[c,:] >= 0), 'nr of events in magnitudes map and parameters map do not correspond on row ' + str(c)
        else: #if 0, copy n_levels as zero map
            if n_levels_mags == 0:
                assert not (pars_map < 0).any(), 'If negative parameters are provided, magnitude map is required.'
                mags_map = np.zeros((n_levels, pars_map.shape[1]-1), dtype=int)
            else:
                pars_map = np.zeros((n_levels, mags_map.shape[1] + 1), dtype=int)
                if (mags_map < 0).any():
                    for c in range(n_levels):
                        pars_map[c, np.where(mags_map[c,:] < 0)[0]] = -1
                        pars_map[c, np.where(mags_map[c,:] < 0)[0]+1] = 1

        #print maps to check level/row mathcing
        if verbose:
            print('\nMagnitudes map:')
            for cnt in range(n_levels):
                print(str(cnt) + ': ', mags_map[cnt,:])

            print('\nParameters map:')
            for cnt in range(n_levels):
                print(str(cnt) + ': ', pars_map[cnt,:])

            #give explanation if negative parameters:
            if (pars_map < 0).any():
                print('\n-----')
                print('Negative parameters. Note that this stage is left out, while the parameters')
                print('of the other stages are compared column by column. In this parameter map example:')
                print(np.array([[0, 0, 0, 0],
                                [0, -1, 0, 0]]))
                print('the parameters of stage 1 are shared, as well as the parameters of stage 3 of')
                print('level 1 with stage 2 (column 3) of level 2 and the last stage of both')
                print('levels.')
                print('Given that event 2 is probably missing in level 2, it would typically')
                print('make more sense to let both stages around event 2 in level 1 vary as')
                print('compared to level 2:')
                print(np.array([[0, 0, 0, 0],
                                [0, -1, 1, 0]]))
                print('-----')

                
        #at this point, all should indicate the same number of levels
        assert n_levels == mags_map.shape[0] == pars_map.shape[0], 'number of unique levels should correspond to number of rows in map(s)'

        assert levels.shape[0] == self.durations.shape[0], 'levels parameter should contain the level per epoch.'
        return n_levels, levels, clabels, pars_map, mags_map


    def _EM_star(self, args): #for tqdm usage
        return self.EM(*args)
    
    def EM(self, magnitudes, parameters, maximization=True, magnitudes_to_fix=None, parameters_to_fix=None, max_iteration=1e3, tolerance=1e-4, min_iteration=1, mags_map=None, pars_map=None,levels=None, cpus=1):  
        '''
        Expectation maximization function underlying fit

        parameters
        ----------
        n_events : int
            how many events are estimated
        magnitudes : ndarray
            2D ndarray n_events * components (or 3D iteration * n_events * n_components), initial levelitions for events magnitudes. If magnitudes are estimated, the list provided is used as starting point,
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
        lkh : float
            Summed log probabilities
        magnitudes : ndarray
            Magnitudes of the channel contribution to each event
        parameters: ndarray
            parameters for the gammas of each stage
        eventprobs: ndarray
            Probabilities with shape max_samples*n_trials*n_events
        traces: ndarray
            Values of the log-likelihood for each EM iteration
        param_dev : ndarray
            paramters for each iteration of EM
        ''' 
        
        assert mags_map.shape[0] == pars_map.shape[0], 'Both maps need to indicate the same number of levels.'
        n_levels = mags_map.shape[0]
        
        n_events = magnitudes.shape[magnitudes.ndim-2]
        locations = np.zeros((n_events+1,),dtype=int) #location per stage
        locations[1:-1] = self.location
        locations = np.tile(locations, (n_levels, 1))
        lkh, eventprobs = self._estim_probs_levels(magnitudes, parameters, locations, mags_map, pars_map, levels, cpus=cpus)
        initial_magnitudes = magnitudes.copy()
        initial_parameters = parameters.copy()
        
        traces = [lkh]
        param_dev = [parameters.copy()] #... and parameters
        i = 0
        if not maximization:
            lkh_prev = lkh
        else:
            lkh_prev = lkh
            parameters_prev = parameters.copy()

            while i < max_iteration :#Expectation-Maximization algorithm
                if i >= min_iteration and (np.isneginf(lkh) or tolerance > (lkh-lkh_prev)/np.abs(lkh_prev)):
                    break
                
                #As long as new run gives better likelihood, go on  
                lkh_prev = lkh.copy()
                parameters_prev = parameters.copy()

                for c in range(n_levels): #get params/mags

                    mags_map_level = np.where(mags_map[c,:]>=0)[0]
                    pars_map_level = np.where(pars_map[c,:]>=0)[0]
                    epochs_level = np.where(levels == c)[0]
                    
                    #get mags/pars by level
                    magnitudes[c,mags_map_level,:], parameters[c,pars_map_level,:] = self.get_magnitudes_parameters_expectation(eventprobs[np.ix_(range(self.max_d),epochs_level, mags_map_level)], subset_epochs=epochs_level)

                    magnitudes[c,magnitudes_to_fix,:] = initial_magnitudes[c,magnitudes_to_fix,:].copy()
                    parameters[c,parameters_to_fix,:] = initial_parameters[c,parameters_to_fix,:].copy()
                
                #set mags to mean if requested in map
                for m in range(n_events):
                    for m_set in np.unique(mags_map[:,m]):
                        if m_set >= 0:
                            magnitudes[mags_map[:,m] == m_set,m,:] = np.mean(magnitudes[mags_map[:,m] == m_set,m,:],axis=0)

                #set param to mean if requested in map
                for p in range(n_events+1):
                    for p_set in np.unique(pars_map[:,p]):
                        if p_set >= 0:
                            parameters[pars_map[:,p] == p_set,p,:] = np.mean(parameters[pars_map[:,p] == p_set,p,:],axis=0)
                lkh, eventprobs = self._estim_probs_levels(magnitudes, parameters, locations, mags_map, pars_map, levels, cpus=cpus)
                traces.append(lkh)
                param_dev.append(parameters.copy())
                i += 1
        _, eventprobs = self._estim_probs_levels(magnitudes, parameters, np.zeros(locations.shape).astype(int), mags_map, pars_map, levels, cpus=cpus)
        if i == max_iteration:
            warn(f'Convergence failed, estimation hitted the maximum number of iteration ({int(max_iteration)})', RuntimeWarning)
        return lkh, magnitudes, parameters, eventprobs, np.array(traces), np.array(param_dev)


    def get_magnitudes_parameters_expectation(self,eventprobs,subset_epochs=None):
        n_events = eventprobs.shape[2]
        n_trials = eventprobs.shape[1]
        if subset_epochs is None: #all trials
            subset_epochs = range(n_trials)

        magnitudes = np.zeros((n_events, self.n_dims))

        #Magnitudes from Expectation, Eq 11 from 2024 paper
        for event in range(n_events):
            for comp in range(self.n_dims):
                event_data = np.zeros((self.max_d, len(subset_epochs)))
                for trial_idx, trial in enumerate(subset_epochs):
                    start, end = self.starts[trial], self.ends[trial]
                    duration = end - start + 1
                    event_data[:duration, trial_idx] = self.crosscorr[start:end+1, comp]
                magnitudes[event, comp] = np.mean(np.sum(eventprobs[:, :, event] * event_data, axis=0))
            # scale cross-correlation with likelihood of the transition
            # sum by-trial these scaled activation for each transition events
            # average across trials
        
        #Gamma parameters from Expectation Eq 10 from 2024 paper
        #calc averagepos here as mean_d can be level dependent, whereas scale_parameters() assumes it's general
        event_times_mean = np.concatenate([np.arange(self.max_d) @ eventprobs.mean(axis=1), [np.mean(self.durations[subset_epochs])-1]])
        parameters = self.scale_parameters(eventprobs=None, n_events=n_events, averagepos=event_times_mean)                            

        return [magnitudes, parameters]
    


    def estim_probs(self, magnitudes, parameters, locations, n_events=None, subset_epochs=None, lkh_only=False, by_trial_lkh=False):
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
        locations : ndarray
            1D ndarray of int with size n_events+1, locations for events
        n_events : int
            how many events are estimated
        subset_epochs : list
            boolean array indicating which epoch should be taken into account for level-based calcs
        lkh_only: bool
            Returning eventprobs (True) or not (False)
        
        Returns
        -------
        loglikelihood : float
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
            gains = gains + self.crosscorr[:,i][np.newaxis].T * magnitudes[:,i]-magnitudes[:,i]**2/2
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
            pmf[:,stage] = np.concatenate((np.repeat(0,locations[stage]), \
                self.distribution_pmf(parameters[stage,0], parameters[stage,1])[locations[stage]:]))
        pmf_b = pmf[:,::-1] # Stage reversed gamma pmf, same order as prob_b

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
                forward[:,trial,event] = np.convolve(forward[:,trial,event-1], pmf[:,event])[:self.max_d]
                # same but backwards
                backward[:,trial,event] = np.convolve(add_b[:,trial], pmf_b[:, event])[:self.max_d]
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

        if lkh_only:
            return likelihood
        elif by_trial_lkh:
            return forward * backward
        else:
            return [likelihood, eventprobs]

    def _estim_probs_levels(self, magnitudes, parameters, locations, mags_map, pars_map, levels, lkh_only=False, cpus=1):
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
        locations : ndarray
            2D n_level * n_events array indication locations for all events
        n_events : int
            how many events are estimated
        lkh_only: bool
            Returning eventprobs (True) or not (False)
        
        Returns
        -------
        loglikelihood : float
            Summed log probabilities
        eventprobs : ndarray
            Probabilities with shape max_samples*n_trials*n_events
        '''

        n_levels = mags_map.shape[0]
        likes_events_level = []
        if cpus > 1:
            with mp.Pool(processes=cpus) as pool:
                likes_events_level = pool.starmap(self.estim_probs, 
                    zip([magnitudes[c, mags_map[c,:]>=0, :] for c in range(n_levels)], [parameters[c, pars_map[c,:]>=0, :] for c in range(n_levels)], [locations[c, pars_map[c,:]>=0] for c in range(n_levels)],itertools.repeat(None), [levels == c for c in range(n_levels)], itertools.repeat(False)))
        else:
            for c in range(n_levels):
                magnitudes_level = magnitudes[c, mags_map[c,:]>=0, :] #select existing magnitudes
                parameters_level = parameters[c, pars_map[c,:]>=0, :] #select existing params
                likes_events_level.append(self.estim_probs(magnitudes_level, parameters_level, locations[c, pars_map[c,:]>=0], subset_epochs = (levels == c)))

        likelihood = np.sum([x[0] for x in likes_events_level])
        eventprobs = np.zeros((self.max_d, len(levels), mags_map.shape[1]))
        for c in range(n_levels):
            eventprobs[np.ix_(range(self.max_d), levels == c, mags_map[c,:]>=0)] = likes_events_level[c][1]

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

    def backward_estimation(self,max_events=None, min_events=0, max_fit=None, max_starting_points=1, tolerance=1e-4, maximization=True, max_iteration=1e3):
        '''
        First read or estimate max_event solution then estimate max_event - 1 solution by 
        iteratively removing one of the event and pick the one with the highest 
        loglikelihood
        
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
                print(f'Estimating all solutions for maximal number of events ({max_events}) with 1 pre-defined starting point and {max_starting_points-1} starting points')
            event_loo_results = [self.fit_n(max_events, starting_points=max_starting_points,  verbose=False)]
        else:
            event_loo_results = [max_fit]
        max_events = event_loo_results[0].event.max().values+1

        for n_events in np.arange(max_events-1,min_events,-1):

            #only take previous model forward when it's actually fitting ok
            if event_loo_results[-1].loglikelihood.values != -np.inf:                

                print(f'Estimating all solutions for {n_events} events')
                        
                pars_prev = event_loo_results[-1].dropna('stage').parameters.values
                mags_prev = event_loo_results[-1].dropna('event').magnitudes.values

                events_temp, pars_temp = [],[]
                
                for event in np.arange(n_events + 1):#creating all possible solutions
                    events_temp.append(mags_prev[np.arange(n_events+1) != event,])
                    
                    temp_pars = np.copy(pars_prev)
                    temp_pars[event,1] = temp_pars[event,1] + temp_pars[event+1,1] #combine two stages into one
                    temp_pars = np.delete(temp_pars, event+1, axis=0)
                    pars_temp.append(temp_pars)

                if self.cpus == 1:
                    event_loo_likelihood_temp = []
                    for i in range(len(events_temp)):
                        event_loo_likelihood_temp.append(self.fit_n(n_events, events_temp[i],pars_temp[i],tolerance=tolerance,max_iteration=max_iteration,maximization=maximization,verbose=False))
                else:
                    inputs = zip(itertools.repeat(n_events), events_temp, pars_temp,\
                                itertools.repeat([]), itertools.repeat([]),\
                                itertools.repeat(tolerance), itertools.repeat(max_iteration), \
                                itertools.repeat(maximization), itertools.repeat(1),\
                                itertools.repeat(1),itertools.repeat(True),\
                                itertools.repeat(False),itertools.repeat(1))
                    with mp.Pool(processes=self.cpus) as pool:
                        event_loo_likelihood_temp = pool.starmap(self.fit_n, inputs)

                lkhs = [x.loglikelihood.values for x in event_loo_likelihood_temp]
                event_loo_results.append(event_loo_likelihood_temp[np.nanargmax(lkhs)])

                #remove event_loo_likelihood
                del event_loo_likelihood_temp
                # Force garbage collection
                gc.collect()
            
            else: 
                print(f'Previous model did not fit well. Estimating a neutral {n_events} event model.')
                event_loo_results.append(self.fit_n(n_events, tolerance=tolerance, max_iteration = max_iteration, maximization = maximization))
        event_loo_results = xr.concat(event_loo_results, dim="n_events", fill_value=np.nan)
        event_loo_results = event_loo_results.assign_coords({"n_events": np.arange(max_events,min_events,-1)})
        event_loo_results = event_loo_results.assign_attrs(method='backward')
        if 'sp_parameters' in event_loo_results.attrs:
            del event_loo_results.attrs['sp_parameters']
            del event_loo_results.attrs['sp_magnitudes']
            del event_loo_results.attrs['maximization']
        return event_loo_results

    def compute_max_events(self):
        '''
        Compute the maximum possible number of events given event width  minimum reaction time
        '''
        return int(np.rint(np.percentile(self.durations, 10)//(self.location)))

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
        while any(rnd_durations < self.event_width_samples): #at least event_width
            rnd_events = np.random.default_rng().integers(low = 0, high = mean_d, size = n_events) #n_events between 0 and mean_d
            rnd_events = np.sort(rnd_events)
            rnd_durations = np.hstack((rnd_events, mean_d)) - np.hstack((0, rnd_events))  #associated durations
        random_stages = np.array([[self.shape, self.mean_to_scale(x, self.shape)] for x in rnd_durations])
        return random_stages    
    
    
    def fit(self, step=None, verbose=True, end=None, tolerance=1e-3, diagnostic=False, return_estimates=False, by_sample=False, pval = None):
        """
         Instead of fitting an n event model this method starts by fitting a 1 event model (two stages) using each sample from the time 0 (stimulus onset) to the mean RT. 
         Therefore it tests for the landing point of the expectation maximization algorithm given each sample as starting point and the likelihood associated with this landing point. 
         As soon as a starting points reaches the convergence criterion, the function fits an n+1 event model and uses the next samples in the RT as starting point for the following event
         
        parameters
        ----------
         	 step: float
                The size of the step from 0 to the mean RT, defaults to the widths of the expected event.
         	 verbose: bool 
                If True print information about the fit
         	 end: int
                The maximum number of samples to explore within each trial
         	 trace: bool 
                If True keep the scale and magnitudes parameters for each iteration
         	 tolerance: float
                The tolerance used for the convergence in the EM() function
         	 diagnostic: bool
                If True print a diagnostic plot of the EM traces for each iteration and several statistics at each iteration
             return_estimates : bool
                return all intermediate models
             by_sample : bool
                try every sample as the starting point, even if a later event has already
                been identified. This in case the method jumped over a local maximum in an earlier estimation.
             pval: float
                 p-value for the detection of the first event, test the first location for significance compared to a distribution of noise estimates
         
         Returns: 
         	 A the fitted HMP mo
        """
        if end is None:
            end = self.mean_d
        if step is None:
            step = self.event_width_samples
        max_event_n = self.compute_max_events()*10#not really nedded, if it fits it fits
        if diagnostic:
            cycol = cycle(default_colors)
        pbar = tqdm(total = int(np.rint(end)))#progress bar
        n_events, j, time = 1, 1, 0 #j = sample after last placed event
        #Init pars (need this for min_model)
        pars = np.zeros((max_event_n+1,2))
        pars[:,0] = self.shape #final gamma parameters during estimation, shape x scale
        pars_prop = pars[:n_events+1].copy() #gamma params of current estimation
        pars_prop[0,1] = self.mean_to_scale(j*step, self.shape) #initialize gamma_parameters at 1 sample
        last_stage = self.mean_to_scale(end-j*step, self.shape) #remainder of time
        pars_prop[-1,1] = last_stage

        #Init mags
        mags = np.zeros((max_event_n, self.n_dims)) #final mags during estimation

        # The first new detected event should be higher than the bias induced by splitting the RT in two random partition
        if pval is not None:
            lkh = self.fit_n(1, maximization=False, starting_points=100, return_max=False, verbose=False)
            lkh_prev = lkh.loglikelihood.mean() + lkh.loglikelihood.std()*norm_pval.ppf(1-pval)
        else:
            lkh_prev = -np.inf
        if return_estimates:
            estimates = [] #store all n_event solutions
        
        # Iterative fit
        while self.scale_to_mean(last_stage, self.shape) >= self.location and n_events <= max_event_n:

            prev_time = time
            
            #get new parameters
            mags_props, pars_prop = self.propose_fit_params(n_events, by_sample, step, j, mags, pars, end)
            last_stage = pars_prop[n_events,1]
            pars_prop = np.array([pars_prop])

            #Estimate model based on these propositions
            solutions = self.fit_n(n_events, mags_props, pars_prop, None, None,\
                            verbose=False, cpus=1, tolerance=tolerance)
            sol_lkh = solutions.loglikelihood.values
            sol_sample_new_event = int(np.round(self.scale_to_mean(np.sum(solutions.parameters.values[:n_events,1]), self.shape)))
            
            #Diagnostic plot
            if diagnostic:
                plt.plot(solutions.traces.T, alpha=.3, c='k')
                print()
                print('Event found at sample ' + str(sol_sample_new_event))
                print(f'Events at {np.round(self.scale_to_mean(np.cumsum(solutions.parameters.values[:,1]), self.shape)).astype(int)}')
                print('lkh change: ' + str(solutions.loglikelihood.values - lkh_prev))
            #check solution
            if sol_lkh - lkh_prev > 0:#accept solution if likelihood improved
            
                lkh_prev = sol_lkh

                #update mags, params,
                mags[:n_events] = solutions.magnitudes.values
                pars[:n_events+1] = solutions.parameters.values

                #store solution
                if return_estimates:
                    estimates.append(solutions)

                #search for an additional event, starting again at sample 1 from prev event,
                #or next sample if by_sample
                n_events += 1
                if by_sample:
                    j += 1
                    time = j * step
                else:
                    j = 1
                    time = sol_sample_new_event + j * step


                #Diagnostic plot
                if diagnostic:
                    color = next(cycol)
                    plt.plot(solutions.traces.T, c=color, label=f'n-events {n_events-1}')
                if verbose: 
                    print(f'Transition event {n_events-1} found around sample {sol_sample_new_event}')


            else: #reject solution, search on
                prev_sample = int(np.round(self.scale_to_mean(np.sum(pars[:n_events-1,1]), self.shape)))
                if not by_sample: #find furthest explored param. Note: this also work by_sample, just a tiny bit faster this way
                    max_scale = np.max([np.sum(x[:n_events,1]) for x in solutions.param_dev.values])
                    max_sample = int(np.round(self.scale_to_mean(max_scale, self.shape)))
                    j = np.max([max_sample - prev_sample +1, (j+1)*step])/step #either ffwd to furthest explored sample or add 1 to j
                    time = prev_sample + j*step
                else:
                    j += 1
                    time = j*step
            
            pbar.update(int(np.rint(time-prev_time)))

        #done estimating

        n_events = n_events-1
        if verbose:
            print()
            print('All events found, refitting final combination.')
        if diagnostic:
            plt.ylabel('Log-likelihood')
            plt.xlabel('EM iteration')
            plt.legend()
        mags = mags[:n_events, :]
        pars = pars[:n_events+1, :]
        if n_events > 0:
            fit = self.fit_n(n_events, parameters=np.array([pars]), magnitudes=np.array([mags]), verbose=verbose, cpus=1)

            fit = fit.assign_attrs(method='fit', step=step, by_sample=int(by_sample))
        else:
            warn('Failed to find more than two stages, returning None')
            fit = None
        del fit.attrs['sp_parameters']
        del fit.attrs['sp_magnitudes']
        del fit.attrs['maximization']
        pbar.update(int(np.rint(end)-int(np.rint(time))))
        if return_estimates:
            return fit, estimates
        else:
            return fit

    def propose_fit_params(self, n_events, by_sample, step, j, mags, pars, end):

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
            
            mags_props = np.zeros((1,n_events, self.n_dims)) #always 0?
            mags_props[:,:n_events-1,:] = np.tile(mags[:n_events-1,:], (len(mags_props), 1, 1))

        #in edge cases scale can get negative, make sure that doesn't happen:
        pars_prop[:,1] = np.maximum(pars_prop[:,1],self.mean_to_scale(1, self.shape)) 
       
        return mags_props, pars_prop
