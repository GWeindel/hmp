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
from tqdm.auto import tqdm as tqdm_auto
from hsmm_mvpy import utils
import tqdm

default_colors =  ['cornflowerblue','indianred','orange','darkblue','darkgreen','gold', 'brown']

                   
class hmp:
    
    def __init__(self, data, eeg_data=None, sfreq=None, offset=0, cpus=1, event_width=50, shape=2, estimate_magnitudes=True, estimate_parameters=True, template=None, location=None, distribution='gamma', em_method="mean"):
        '''
        HMP calculates the probability of data summing over all ways of 
        placing the n events to break the trial into n + 1 stages.

        Parameters
        ----------
        data : ndarray
            2D ndarray with n_samples * components 
        sfreq : int
            Sampling frequency of the signal (initially 100)
        event_width : int
            width of events in milliseconds, originally 5 samples
        location : float
            Minimum stage duration in milliseconds. 
        '''
        match distribution:
            case 'gamma':
                from scipy.stats import gamma as sp_dist
            case 'lognormal':
                from scipy.stats import lognorm as sp_dist
            case 'wald':
                from scipy.stats import invgauss as sp_dist
            case 'weibull':
                from scipy.stats import weibull_min as sp_dist
            case _:
                raise ValueError(f'Unknown Distribution {distribution}')
        self.cdf = sp_dist.cdf
            
        if sfreq is None:
            sfreq = eeg_data.sfreq
        if offset is None:
            offset = eeg_data.offset
        self.sfreq = sfreq
        self.steps = 1000/self.sfreq
        self.shape = float(shape)
        self.event_width = event_width
        self.event_width_samples = int(np.round(self.event_width / self.steps))
        if location is None:
            self.location = int(np.round(self.event_width_samples/2))
        else: self.location =  int(np.round(location / self.steps))
        durations = data.unstack().sel(component=0).swap_dims({'epochs':'trials'})\
            .stack(trial_x_participant=['participant','trials']).dropna(dim="trial_x_participant",\
            how="all").groupby('trial_x_participant').count(dim="samples").cumsum().squeeze()
        if durations.trial_x_participant.count() > 1:
            dur_dropped_na = durations.dropna("trial_x_participant")
            starts = np.roll(dur_dropped_na.data, 1)
            starts[0] = 0
            ends = dur_dropped_na.data-1 -offset
        else: 
            dur_dropped_na = durations
            starts = np.array([0])
            ends = np.array([dur_dropped_na.data-1 -offset])
        self.starts = starts
        self.ends = ends
        self.durations =  self.ends-self.starts+1
        self.max_events = self.compute_max_events()
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
            self.template = self.event_shape()
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
    
    def event_shape(self):
        '''
        Computes the template of a half-sine (event) with given frequency f and sampling frequency
        '''
        event_idx = np.arange(self.event_width_samples)*self.steps+self.steps/2
        event_frequency = 1000/(self.event_width*2)#gives event frequency given that events are defined as half-sines
        template = np.sin(2*np.pi*event_idx/1000*event_frequency)#event morph based on a half sine with given event width and sampling frequency
        template = template/np.sum(template**2)#Weight normalized
        return template
            
    def cross_correlation(self,data):
        '''
        This function puts on each sample the correlation of that sample and the next 
        x samples (depends on sampling frequency and event size) with a half sine on time domain.
        
        Parameters
        ----------
        data : ndarray
            2D ndarray with n_samples * components
        Returns
        -------
        events : ndarray
            a 2D ndarray with samples * PC components where cell values have
            been correlated with event morphology
        '''
        from scipy.signal import fftconvolve
        events = np.zeros(data.shape)
        for trial in range(self.n_trials):#avoids confusion of gains between trials
            for dim in np.arange(self.n_dims):
                events[self.starts[trial]:self.ends[trial]+1,dim] = \
                    fftconvolve(data[self.starts[trial]:self.ends[trial]+1, dim], \
                        self.template, mode='full')\
                        [len(self.template)-1:self.durations[trial]+len(self.template)+1]
        return events

    def fit_single(self, n_events=None, magnitudes=None, parameters=None, parameters_to_fix=None, 
                   magnitudes_to_fix=None, tolerance=1e-4, max_iteration=1e3, maximization=True,
                   starting_points=1, method='random', return_max=True, verbose=True, cpus=None):
        '''
        Fit HMP for a single n_events model
        
        Parameters
        ----------
        n_events : int
            how many events are estimated
        magnitudes : ndarray
            2D ndarray n_events * components, initial conditions for events magnitudes
        parameters : list
            list of initial conditions for Gamma distribution scale parameter. If parameters are estimated, the list provided is used as starting point,
            if parameters are fixed, parameters estimated will be the same as the one provided. When providing a list, stage need to be in the same order
            _n_th gamma parameter is  used for the _n_th stage
        threshold : float
            threshold for the HMP algorithm, 0 skips HMP
        '''
        assert n_events is not None, 'The fit_single() function needs to be provided with a number of expected transition events'
        assert self.location*(n_events+1) < min(self.durations), f'{n_events} events do not fit given the minimum duration of {min(self.durations)} and a location of {self.location}'
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
                parameters = np.tile([self.shape, ((np.mean(self.durations))/(n_events+1)-self.location)/self.shape], (n_events+1,1))
                parameters[0,1] = np.mean(self.durations)/(n_events+1)/self.shape #first stage has no location
            initial_p = parameters
            
            if magnitudes is None:
                magnitudes = np.zeros((n_events,self.n_dims), dtype=np.float64)
            initial_m = magnitudes
        if starting_points > 1 or len(np.shape(magnitudes)) == 3 or len(np.shape(parameters)) == 3:
            filterwarnings('ignore', 'Convergence failed, estimation hitted the maximum ', )#will be the case but for a subset only hence ignore
            if len(np.shape(initial_m)) == 2:
                parameters = [initial_p]
                magnitudes = [initial_m]
                if method == 'random':
                    for _ in np.arange(starting_points):
                        proposal_p = self.gen_random_stages(n_events, self.mean_d)
                        proposal_p[parameters_to_fix] = initial_p[parameters_to_fix]
                        #proposal_m = self._gen_mags(n_events, starting_points, method='random')
                        # proposal_m[magnitudes_to_fix] = initial_m[magnitudes_to_fix]
                        # magnitudes.append(proposal_m)
                        parameters.append(proposal_p)
                    magnitudes = self._gen_mags(n_events, starting_points, method='random')
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
                with mp.Pool(processes=cpus) as pool:

                    estimates = pool.starmap(self.EM, 
                        zip(itertools.repeat(n_events), magnitudes, parameters, itertools.repeat(maximization),
                        itertools.repeat(magnitudes_to_fix),itertools.repeat(parameters_to_fix), itertools.repeat(max_iteration), itertools.repeat(tolerance)))   
            else:#avoids problems if called in an already parallel function
                estimates = []
                for pars, mags in zip(parameters, magnitudes):
                    estimates.append(self.EM(n_events, mags, pars, maximization,\
                    magnitudes_to_fix, parameters_to_fix, max_iteration, tolerance))
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
            
        elif starting_points==1:#informed starting point
            lkh, mags, pars, eventprobs, traces = self.EM(n_events, initial_m, initial_p,\
                                        maximization, magnitudes_to_fix, parameters_to_fix, \
                                         max_iteration, tolerance)

        else:#uninitialized    
            if np.any(parameters)== None:
                parameters = np.tile([self.shape, (self.mean_d)/self.shape], (n_events+1,1))
            if np.any(magnitudes)== None:
                magnitudes = np.zeros((n_events, self.n_dims), dtype=np.float64)
            lkh, mags, pars, eventprobs, traces = self.EM(n_events, magnitudes, parameters,\
                                        threshold, magnitudes_to_fix, parameters_to_fix,\
                                        max_iteration, tolerance)
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

            xreventprobs = xr.Dataset({'eventprobs': (('event', 'trial_x_participant','samples'), 
                                             eventprobs.T)},
                             {'event':range(n_event_xreventprobs),
                              'samples':range(np.shape(eventprobs)[0]),
                            'trial_x_participant':  MultiIndex.from_arrays([part,trial],
                                    names=('participant','trials'))})
            xreventprobs = xreventprobs.transpose('trial_x_participant','samples','event')
        else: 
            n_event_xr = len(mags[0])
            xrlikelihoods = xr.DataArray(lkh , dims=("iteration"), name="likelihoods", coords={'iteration':range(len(lkh))})
            xrtraces = xr.DataArray(traces, dims=("iteration","em_iteration"), name="traces", coords={'iteration':range(len(lkh)), 'em_iteration':range(len(traces[0]))})
            xrparams = xr.DataArray(pars, dims=("iteration","stage",'parameter'), name="parameters", 
                            coords = {'iteration': range(len(lkh)), 'parameter':['shape','scale']})
            xrmags = xr.DataArray(mags, dims=("iteration","event","component"), name="magnitudes",
                        coords={'iteration':range(len(lkh)), 'event':range(n_event_xr),
                                "component":range(self.n_dims)})
            part, trial = self.coords['participant'].values, self.coords['trials'].values

            xreventprobs = xr.Dataset({'eventprobs': (('iteration','event', \
                                    'trial_x_participant','samples'), [x.T for x in eventprobs])},
                             {'iteration':range(len(lkh)),
                              'event':np.arange(n_event_xr),
                              'samples':np.arange(np.shape(eventprobs)[1]),
                              'trial_x_participant':  MultiIndex.from_arrays([part,trial],
                                    names=('participant','trials'))})
            xreventprobs = xreventprobs.transpose('iteration','trial_x_participant','samples','event')
        estimated = xr.merge((xrlikelihoods, xrparams, xrmags, xreventprobs, xrtraces))

        if verbose:
            print(f"Parameters estimated for {n_events} events model")
        return estimated
    
    def EM(self, n_events, magnitudes, parameters,  maximization=True, magnitudes_to_fix=None, parameters_to_fix=None, max_iteration=1e3, tolerance=1e-4, min_iteration=1):  
        '''
        Expectation maximization function underlying fit
        ''' 
        if not isinstance(maximization, bool):#Backward compatibility with previous versions
            warn('Deprecated use of the threshold function, use maximization and tolerance arguments. Setting tolerance at 1 for compatibility')
            maximization = {1:True, 0:False}[maximization]
            if maximization:#Backward compatibility, equivalent to previous threshold = 1
                tolerance = 1
        null_stages = np.where(parameters[:,1]<0)[0]
        wrong_shape = np.where(parameters[:,0]!=self.shape)[0]
        if len(null_stages)>0:
            raise ValueError(f'Wrong scale parameter input, provided scale parameter(s) {null_stages} should be positive but have value {parameters[null_stages,:].prod(axis=1)}')
        if len(wrong_shape)>0:
            raise ValueError(f'Wrong shape parameter input, provided parameter(s) {wrong_shape} shape is {parameters[wrong_shape,0]} but expected  expected {self.shape}')
        initial_parameters =  np.copy(parameters)
        initial_magnitudes = np.copy(magnitudes)
        
        lkh, eventprobs = self.estim_probs(magnitudes, parameters, n_events)
        traces = []
        i = 0
        if not maximization or n_events==0:
            lkh_prev = lkh
            magnitudes_prev = initial_magnitudes
            parameters_prev = initial_parameters
            eventprobs_prev = eventprobs
        else:
            lkh_prev = -np.inf
            while i < max_iteration :#Expectation-Maximization algorithm
                if i > min_iteration and tolerance > lkh - lkh_prev:
                    break
                    #As long as new run gives better likelihood, go on  
                lkh_prev = lkh.copy()
                magnitudes_prev = magnitudes.copy()
                parameters_prev = parameters.copy()
                eventprobs_prev = eventprobs.copy()
                #Magnitudes from Expectation
                event_times = np.zeros((n_events+1, self.n_trials))
                for event in range(n_events):
                    if self.em_method == "max":
                        #Take time point at maximum p() for each trial
                        #Average channel activity at those points
                        event_values = np.zeros((self.n_trials, self.n_dims))
                        for trial in range(self.n_trials):
                            event_times[event,trial]  = np.argmax(eventprobs[:, trial, event])
                            event_values[trial] = self.events[self.starts[trial] + int(event_times[event,trial])]
                        magnitudes[event] = np.mean(event_values, axis=0)
                    elif self.em_method == "mean":
                        for comp in range(self.n_dims):
                            magnitudes[event,comp] = np.mean(np.sum( \
                                eventprobs[:,:,event]*self.data_matrix[:,:,comp], axis=0))
                        # Scale cross-correlation with likelihood of the transition
                        # sum by-trial these scaled activation for each transition events
                        # average across trials
                event_times[-1,:] = self.mean_d
                magnitudes[magnitudes_to_fix,:] = initial_magnitudes[magnitudes_to_fix,:].copy()
                if self.em_method == "max":
                    parameters = self.scale_parameters(eventprobs=None, n_events=n_events, averagepos=np.mean(event_times,axis=1))#Parameters from Expectation
                elif self.em_method == "mean":
                    parameters = self.scale_parameters(eventprobs, n_events)#Parameters from Expectation
                parameters[parameters_to_fix, :] = initial_parameters[parameters_to_fix,:].copy()
                lkh, eventprobs = self.estim_probs(magnitudes, parameters, n_events)
                traces.append(lkh)
                i += 1
        if i == max_iteration:
            warn(f'Convergence failed, estimation hitted the maximum number of iteration ({int(max_iteration)})', RuntimeWarning)
        return lkh_prev, magnitudes_prev, parameters_prev, eventprobs_prev, np.array(traces)

    
    def estim_probs(self, magnitudes, parameters, n_events, lkh_only=False):
        '''
        
        Returns
        -------
        likelihood : float
            Summed log probabilities
        eventprobs : ndarray
            Probabilities with shape max_samples*n_trials*n_events
        '''
        n_stages = n_events+1
        gains = np.zeros((self.n_samples, n_events), dtype=np.float64)
        for i in range(self.n_dims):
            # computes the gains, i.e. how much the congruence between the pattern shape
            # and the data given the magnitudes of the sensors
            gains = gains + self.events[:,i][np.newaxis].T * magnitudes[:,i]
        gains = np.exp(gains)
        probs = np.zeros([self.max_d,self.n_trials,n_events], dtype=np.float64) # prob per trial
        probs_b = np.zeros([self.max_d,self.n_trials,n_events], dtype=np.float64)# Sample and state reversed
        for trial in np.arange(self.n_trials):
            # Following assigns gain per trial to variable probs 
            probs[:self.durations[trial],trial,:] = \
                gains[self.starts[trial]:self.ends[trial]+1,:] 
            # Same but samples and events are reversed, this allows to compute
            # fwd and bwd in the same way in the following steps
            probs_b[:self.durations[trial],trial,:] = \
                gains[self.starts[trial]:self.ends[trial]+1,:][::-1,::-1]

        pmf = np.zeros([self.max_d, n_stages], dtype=np.float64) # Gamma pmf for each stage parameters
        locations = np.concatenate([[-.5], np.repeat(self.location, n_events)])#all stages except first stage have a location
        locations[-1] -= 1
        for stage in range(n_stages):
            pmf[:,stage] = self.distribution_pmf(parameters[stage,0], parameters[stage,1], locations[stage])
        pmf_b = pmf[:,::-1] # Stage reversed gamma pmf, same order as prob_b

        if n_events > 0:
            forward = np.zeros((self.max_d, self.n_trials, n_events), dtype=np.float64)
            backward = np.zeros((self.max_d, self.n_trials, n_events), dtype=np.float64)
            # Computing forward and backward helper variable
            #  when stage = 0:
            forward[:,:,0] = np.tile(pmf[:,0][np.newaxis].T,\
                (1,self.n_trials))*probs[:,:,0] #first stage transition is p(B) * p(d)
            backward[:,:,0] = np.tile(pmf_b[:,0][np.newaxis].T,\
                        (1,self.n_trials)) #Reversed gamma (i.e. last stage) without probs as last event ends at time T

            for event in np.arange(1,n_events):#Following stage transitions integrate previous transitions
                add_b = backward[:,:,event-1]*probs_b[:,:,event-1]#Next stage in back
                for trial in np.arange(self.n_trials):
                    # convolution between gamma * gains at previous event and event
                    forward[:,trial,event] = self.convolution(forward[:,trial,event-1], pmf[:,event])[:self.max_d]
                    # same but backwards
                    backward[:,trial,event] = self.convolution(add_b[:,trial], pmf_b[:, event])[:self.max_d]
                forward[:,:,event] = forward[:,:,event]*probs[:,:,event]
            #re-arranging backward to the expected variable
            backward = backward[:,:,::-1]#undoes stage inversion
            for trial in np.arange(self.n_trials):#Undoes sample inversion
                backward[:self.durations[trial],trial,:] = \
                    backward[:self.durations[trial],trial,:][::-1]
            
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
            forward = np.zeros((self.max_d, self.n_trials), dtype=np.float64)
            backward = np.zeros((self.max_d, self.n_trials), dtype=np.float64)
            forward[:,:] = np.tile(pmf[:,0][np.newaxis].T,\
                (1,self.n_trials))
            backward[:,:] = np.tile(pmf_b[:,0][np.newaxis].T,\
                        (1,self.n_trials))
            for trial in np.arange(self.n_trials):#Undoes sample inversion
                backward[:self.durations[trial],trial] = \
                    backward[:self.durations[trial],trial][::-1]
            eventprobs = forward * backward
            likelihood = np.sum(np.log(eventprobs[:,:].sum(axis=0)))#sum over max_samples to avoid 0s in log
            eventprobs = eventprobs / eventprobs.sum(axis=0)

        if lkh_only:
            return likelihood
        else:
            return [likelihood, eventprobs]

    def distribution_pmf(self, shape, scale, location):
        '''
        Returns PMF of gamma or lognormal dist with shape and scale, on a range from 0 to max_length 
        
        Parameters
        ----------
        a : float
            shape parameter
        scale : float
            scale parameter      
        Returns
        -------
        p : ndarray
            probabilty mass function for a gamma with given parameters
        '''
        if scale == 0:
            warn('Convergence failed: one stage has been found to be null')
        p = self.cdf(np.arange(self.max_d), shape, scale=scale, loc=location)
        #Location is in fact +1 as np.arange starts from 0
        p = np.diff(p, prepend=0)#going to pmf
        return p
    
    def scale_parameters(self, eventprobs=None, n_events=None, averagepos=None):
        '''
        Used for the re-estimation in the EM procdure. The likeliest location of 
        the event is computed from eventprobs. The scale parameters are then taken as the average 
        distance between the events corrected for (eventual) location
        Parameters
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
        if eventprobs is not None:
            averagepos = np.concatenate([np.arange(1,self.max_d+1)@eventprobs.mean(axis=1),
                                         [self.mean_d]])#Durations
        params = np.zeros((n_events+1,2), dtype=np.float64)
        params[:,0] = self.shape
        params[:,1] = np.diff(averagepos, prepend=0)
        params[:-1,1] += 1
        params[:,1] = params[:,1]/params[:,0]
        return params


    def backward_estimation(self,max_events=None, min_events=0, max_fit=None, max_starting_points=1, method="random", tolerance=1e-4, maximization=True, max_iteration=1e3):
        '''
        First read or estimate max_event solution then estimate max_event - 1 solution by 
        iteratively removing one of the event and pick the one with the highest 
        likelihood
        
        Parameters
        ----------
        max_fit : xarray
            To avoid re-estimating the model with maximum number of events it can be provided 
            with this arguments, defaults to None
        max_starting_points: int
            how many random starting points iteration to try for the model estimating the maximal number of events
        
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
        i = 0
        for n_events in np.arange(max_events-1,min_events,-1):
            print(f'Estimating all solutions for {n_events} number of events')
            temp_best = event_loo_results[i]#previous event solution
            temp_best = temp_best.dropna('event')
            temp_best = temp_best.dropna('stage')
            n_events_list = np.arange(n_events+1)#all events from previous solution
            flats = temp_best.parameters.values
            events_temp,pars_temp = [],[]
            for event in np.arange(n_events+1):#creating all possible solutions
                events_temp.append(temp_best.magnitudes.sel(event = np.array(list(set(n_events_list) - set([event])))).values)
                flat = event + 1 #one more flat than events
                temp = np.copy(flats[:,1])
                temp[flat-1] = temp[flat-1] + temp[flat]
                temp = np.delete(temp, flat)
                pars_temp.append(np.reshape(np.concatenate([np.repeat(self.shape, len(temp)), temp]), (2, len(temp))).T)
            with mp.Pool(processes=self.cpus) as pool:
                event_loo_likelihood_temp = pool.starmap(self.fit_single, 
                    zip(itertools.repeat(n_events), events_temp, pars_temp,\
                        itertools.repeat([]), itertools.repeat([]),\
                        itertools.repeat(tolerance), itertools.repeat(max_iteration), \
                        itertools.repeat(maximization), itertools.repeat(1),\
                        itertools.repeat('random'), itertools.repeat(True),\
                        itertools.repeat(False),itertools.repeat(1)))
            models = xr.concat(event_loo_likelihood_temp, dim="iteration")
            event_loo_results.append(models.sel(iteration=[np.where(models.likelihoods == models.likelihoods.max())[0][0]]).squeeze('iteration'))
            i+=1
        bests = xr.concat(event_loo_results, dim="n_events")
        bests = bests.assign_coords({"n_events": np.arange(max_events,min_events,-1)})
        #bests = bests.squeeze('iteration')
        return bests

    def compute_max_events(self):
        '''
        Compute the maximum possible number of events given event width and mean or minimum reaction time
        '''
        return int(np.rint(np.min(self.durations)//(self.location)))-1

    def event_times(self, eventprobs, mean=True):
        '''
        Compute event onset times based on event probabilities
        This function is mainly kept for compatibility with previous matlab applications

        Parameters
        ----------
        a : float
            shape parameter
        b : float
            scale parameter
        max_length : int
            maximum length of the trials        
        Returns
        -------
        d : ndarray
            density for a gamma with given parameters
        '''
        # warn('This method is deprecated and will be removed in future version, use compute_times() instead', DeprecationWarning, stacklevel=2)
        eventprobs = eventprobs.dropna('event', how="all")
        eventprobs = eventprobs.dropna('trial_x_participant', how="all")
        onsets = np.empty((len(eventprobs.trial_x_participant),len(eventprobs.event)+1))*np.nan
        i = 0
        for trial in eventprobs.trial_x_participant.dropna('trial_x_participant', how="all").values:
            onsets[i, :len(eventprobs.event)] = np.arange(self.max_d) @ eventprobs.sel(trial_x_participant=trial).data
            onsets[i, -1] = self.ends[i] - self.starts[i]
            i += 1
        if mean:
            return np.mean(onsets, axis=0)
        else:
            return onsets

    @staticmethod        
    def compute_times(init, estimates, duration=False, fill_value=None, mean=False, cumulative=False, add_rt=False):
        '''
        Compute the likeliest onset times for each event

        Parameters
        ----------
        estimates :
            Estimated instance of an HMP model
        init : 
            Initialized HMP object  
        duration : bool
            Whether to compute onset times (False) or stage duration (True)
        fill_value : float | ndarray
            What value to fill for the first onset/duration

        Returns
        -------
        times : xr.DataArray
            Transition event onset or stage duration with trial_x_participant*event dimensions
        '''

        eventprobs = estimates.eventprobs.fillna(0).copy()
        if init.em_method == "max":
            times = times = eventprobs.argmax('samples')#Most likely event location
        else:
            times = xr.dot(eventprobs, eventprobs.samples, dims='samples')
        n = len(times[0,:].values[np.isfinite(times[0,:].values)])
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
            rts = rts.assign_coords(event=int(times.event.max().values+1))
            rts = rts.expand_dims(dim="event")
            times = xr.concat([times, rts], dim='event')
        if duration:
            #adding reaction time and treating it as the last event
            times = times.rename({'event':'stage'})
            if not cumulative:
                times = times.diff(dim='stage')
        if mean:
            times = times.mean('trial_x_participant')
        return times
   
    @staticmethod
    def compute_topologies(channels, estimated, event_width_samples, extra_dim=False, mean=True):
        shift = event_width_samples//2+1#Shifts to compute channel topology at the peak of the event
        channels = channels.rename({'epochs':'trials'}).\
                          stack(trial_x_participant=['participant','trials']).data.fillna(0).drop_duplicates('trial_x_participant')
        estimated = estimated.eventprobs.fillna(0).copy()
        n_events = estimated.event.count().values
        n_trials = estimated.trial_x_participant.count().values
        n_channels = channels.channels.count().values
        if not extra_dim:
            event_values = np.zeros((n_trials, n_events, n_channels))*np.nan
            for event in range(n_events):
                #Take time point at maximum p() for each trial
                #Average channel activity at those points            
                for t, trial in enumerate(estimated.trial_x_participant):
                    time = estimated.sel(trial_x_participant=trial, event=event).argmax('samples')
                    event_values[t, event, :] = channels.sel(trial_x_participant=trial, samples=time+shift).values
            event_values = xr.DataArray(event_values, 
                        dims = ["trial_x_participant","event","channels"],
                        coords={"trial_x_participant":estimated.trial_x_participant,
                                "event": estimated.event,
                                "channels":channels.channels
                        })
        else:
            n_dim = estimated[extra_dim].count().values
            event_values = np.zeros((n_dim, n_trials, n_events, n_channels))*np.nan
            for x in range(n_dim):
                for event in range(n_events):
                    #Take time point at maximum p() for each trial
                    #Average channel activity at those points     
                    for t, trial in enumerate(estimated[x].trial_x_participant):
                        time = estimated[x].sel(trial_x_participant=trial, event=event).argmax('samples')
                        event_values[x, t, event, :] = channels.sel(trial_x_participant=trial, samples=time+shift).values
            event_values = xr.DataArray(event_values, 
                    dims = [extra_dim, "trial_x_participant","event","channels"],
                    coords={extra_dim:estimated[extra_dim],
                            "trial_x_participant":estimated.trial_x_participant,
                            "event": estimated.event,
                            "channels":channels.channels
                    })
        if mean:
            return event_values.mean('trial_x_participant')
        else:
            return event_values


    def gen_random_stages(self, n_events, mean_d):
        '''
        Returns random stage duration between 0 and mean RT by iteratively drawind sample from a 
        uniform distribution between the last stage duration (equal to 0 for first iteration) and 1.
        Last stage is equal to 1-previous stage duration.
        The stages are then scaled to the mean RT
        Parameters
        ----------
        n_events : int
            how many events
        mean_d : float
            scale parameter
        Returns
        -------
        random_stages : ndarray
            random partition between 0 and mean_d
        '''
        mean_d = int(mean_d) - n_events * self.location #remove minimum stage duration
        rnd_durations = np.zeros(n_events + 1)
        while any(rnd_durations < 2): #make sure they are at least 2 samples
            rnd_events = np.random.default_rng().integers(low = 0, high = mean_d, size = n_events) #n_events between 0 and mean_d
            rnd_events = np.sort(rnd_events)
            rnd_durations = np.hstack((rnd_events, mean_d)) - np.hstack((0, rnd_events))  #associated durations
        random_stages = np.array([[self.shape, x / self.shape] for x in rnd_durations])
        return random_stages
    
    def _gen_mags(self, n_events, n_samples=None, lower_bound=-1, upper_bound=1, method=None, size=2, decimate=False, verbose=True):
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

        if n_samples is None or n_samples > len(grid):
            n_samples = len(grid)
        if decimate:
            n_samples = int(len(grid) / decimate)
            
        if method is None and len(grid) != n_samples:
            sort_table = np.argsort(np.abs(np.sum(grid, axis=-1)), axis=0)[::-1]
            grid = grid[sort_table[:n_samples]]
            if verbose:
                print(f'Because of decimation {len(grid)} will be estimated.')
        gen_mags = np.zeros((n_events, n_samples, self.n_dims))
        for event in range(n_events):
            if method is None:
                gen_mags[event,:,:] = grid
            elif method == "random":
                gen_mags[event,:,:] = grid[np.random.choice(range(len(grid)), size=n_samples, replace=False)]
        gen_mags = np.transpose(gen_mags, axes=(1,0,2))
        return gen_mags
    
    def _grid_search(self, n_stages, n_points=None, verbose=True, start_time=0, end_time=None, iter_limit=np.inf, step=1, offset=None, method='slide'):
        '''
        This function decomposes the mean RT into a grid with points. Ideal case is to have a grid with one sample = one search point but the number
        of possibilities badly scales with the length of the RT and the number of stages. Therefore the iter_limit is used to select an optimal number
        of points in the grid with a given spacing. After having defined the grid, the function then generates all possible combination of 
        event placements within this grid. It is faster than using random points (both should converge) but depending on the mean RT and the number 
        of events to look for, the number of combination can be really large. 
        
        Parameters
        ----------
        n_stages : int
            how many event to look for
        iter_limit : int
            How much is too much

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
                parameters[idx, :, :] = [[self.shape, x/self.shape] for x in y]
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
    
    def sliding_event_mags(self, epoch_data, step=5, decimate_grid = False, cpu=1, plot=False, tolerance=1):

        grid = np.squeeze(self._gen_mags(1, decimate = decimate_grid)) #get magn grid
        n_iter = len(grid)

        #calculate estimates, returns lkhs, mags, times
        inputs = zip(itertools.repeat((12,3)), itertools.repeat(False), itertools.repeat('search'), 
                    grid, itertools.repeat(step), itertools.repeat(False), itertools.repeat(None),
                    itertools.repeat(False),itertools.repeat(1), itertools.repeat(tolerance))
        with mp.Pool(processes=cpu) as pool:
            estimates = list(tqdm.tqdm(pool.imap(self.sliding_event_star, inputs), total=len(grid)))
        
        #topo prep
        stacked_eeg_data = epoch_data.stack(trial_x_participant=('participant','epochs')).dropna('trial_x_participant',how='all').data.to_numpy() 
        n_electrodes, _, n_trials = stacked_eeg_data.shape
        shift = int(self.event_width_samples/2)

        #collect results
        max_run_len = max([len(x[0]) for x in estimates])
        lkhs = np.zeros((n_iter * max_run_len))*np.nan
        times = np.zeros((n_iter * max_run_len))*np.nan
        mags = np.zeros((n_iter * max_run_len, self.n_dims))*np.nan
        channels = np.zeros((n_iter * max_run_len, epoch_data.dims['channels']))*np.nan

        for i, est in enumerate(estimates):
            idx = i * max_run_len
            lkhs[idx:(idx + len(est[0]))] = est[0]
            mags[idx:(idx + len(est[0])), :] = est[1].squeeze()
            times[idx:(idx + len(est[0]))] = np.mean(est[2],axis=1)  #these should NOT be shifted, later used to calc pars
            
            #get topos per estimation per trial, then average
            topos = np.zeros((n_electrodes, n_trials, len(est[0])))
            for j, times_per_event in enumerate(est[2]):
                for trial in range(n_trials):
                    topos[:,trial,j] = stacked_eeg_data[:, times_per_event[trial] + shift,trial]
                
            channels[idx:(idx+len(est[0])), :] = np.nanmean(topos,axis=1).T

        mags = mags[~np.isnan(lkhs),:]
        channels = channels[~np.isnan(lkhs),:]
        times = times[~np.isnan(lkhs)]
        lkhs = lkhs[~np.isnan(lkhs)]

        if plot:
            _, ax = plt.subplots(1,1,figsize=(20,3))
            ax.plot(times, lkhs, '.')
            
        return lkhs, mags, channels, times

    def sliding_event_star(self, args): #for tqdm usage
        return self.sliding_event(*args)
        
    def sliding_event(self, figsize=(12,3), verbose=True, method=None, magnitudes=None, step=1, show=True, ax=None, fix_mags=True, cpus=None, tolerance=1e-4):
        '''
        This method outputs the likelihood and estimated parameters of a 1 event model with each sample, from 0 to the mean 
        epoch duration. The parameters and likelihoods that are returned are 
        Take the highest likelihood, place a event by excluding event width space around it, follow with the next one
        '''
             
        parameters = self._grid_search(2, verbose=verbose, step=step, start_time=np.random.choice(range(step)))#Looking for all possibilities with one event
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
        lkhs_init, mags_init, pars_init, times_init = self.estimate_single_event(magnitudes, parameters, [], mags_to_fix, maximization, cpus, tolerance=tolerance)
        
        if verbose:
            if ax is None:
                 _, ax = plt.subplots(figsize=figsize, dpi=100)
            ax.plot(pars_init[:,0,1]*self.shape, lkhs_init, ls)
        
        if method is None:
            #pars, mags, lkhs = pars_init, mags_init, lkhs_init
            plt.ylabel('Log-likelihood')
            plt.xlabel('Sample number')
            if show:
                plt.show()
            else:
                return ax
        else:
            return lkhs_init, mags_init, times_init
        
    def estimate_single_event(self, magnitudes, parameters, parameters_to_fix, magnitudes_to_fix, maximization, cpus, max_iteration=1e2, tolerance=1e-4):
        filterwarnings('ignore', 'Convergence failed, estimation hitted the maximum ', )#will be the case but for a subset only hence ignore
        if cpus is None:
            cpus = self.cpus
        if cpus > 1:
            if np.shape(magnitudes) == 2:
                magnitudes = np.tile(magnitudes, (len(parameters), 1, 1))
            with mp.Pool(processes=self.cpus) as pool:
                estimates = pool.starmap(self.EM, 
                    zip(itertools.repeat(1), magnitudes, parameters, 
                        itertools.repeat(maximization), itertools.repeat(magnitudes_to_fix), 
                        itertools.repeat(parameters_to_fix), itertools.repeat(max_iteration),
                        itertools.repeat(tolerance)))
        else:
            estimates = []
            for pars, mags in zip(parameters, magnitudes):
                estimates.append(self.EM(1, mags, pars, maximization, magnitudes_to_fix, parameters_to_fix, max_iteration, tolerance))
        lkhs_sp = np.array([x[0] for x in estimates])
        mags_sp = np.array([x[1] for x in estimates])
        pars_sp = np.array([x[2] for x in estimates])

        #take max prob times from eventprobs, to reduce memory usage
        times_sp = np.array([np.argmax(np.squeeze(x[3]),axis=0) for x in estimates])

        resetwarnings()
        return lkhs_sp, mags_sp, pars_sp, times_sp
    
    def fit(self, step=1, verbose=True, end=None, threshold=1, trace=False, fix_iter=True, max_iterations=1e3, tolerance=1e-2, grid_points=25, cpus=None):
        '''
        Cumulative fit method.
        step = size of steps across samples
        end = max explored duration
        threshold = 
        '''
        if cpus is None:
            cpus = self.cpus
        if end is None:
            end = self.mean_d
        max_event_n = self.compute_max_events()
        n_points = int(np.rint(end)//step)
        
        pbar = tqdm_auto(total = int(np.rint(end-self.location)))#progress bar
        n_events, j, time = 0,1,0
        if trace:
            all_pars, all_mags, all_mags_prop, all_pars_prop, all_diffs = [],[],[],[],[]
        #Init pars
        pars = np.zeros((int(end),2))
        pars[:,0] = self.shape #parameters during estimation, shape x scale
        pars_prop = pars[:n_events+2].copy()
        pars_prop[0,1] = step/self.shape#initialize scale
        pars_prop[n_events+1,1] = last_stage = (end-step)/self.shape 
        pars_accepted = pars.copy()
        #Init mags
        mags = np.zeros((int(end), self.n_dims)) #mags during estimation
        mags_prop = mags[:n_events+1].copy()
        mags_prop[n_events,:] = np.zeros(self.n_dims)
        mags_accepted = mags.copy()

        while last_stage*self.shape > self.location and n_events+1 < max_event_n:
            prev_time = time
            #Generate a grid of magnitudes 
            mags_props = self._gen_mags(n_events+1, n_samples=grid_points, verbose=False)
            #replave eventual event already found
            mags_props[:,:n_events,:] = np.tile(mags_prop[:n_events,:], (len(mags_props), 1, 1))
            #estimate all grid_points models while fixing previous found events
            solutions = self.fit_single(n_events+1, mags_props, pars_prop, [range(n_events)], [range(n_events)], return_max=False, verbose=False, cpus=cpus)
            #Exclude non-converged models (negative EM LL curve)
            solutions = utils.filter_non_converged(solutions)
            if len(solutions.iteration) > 0:#Success
                if verbose:#Diagnostic plot
                    plt.plot(solutions.traces.T)
                #Average among the converged solutions and store as future starting points
                # mags[:n_events+1], pars[:n_events+2] = solutions.magnitudes.mean('iteration').values, solutions.parameters.mean('iteration').values
                nearest_solution = solutions.sel(iteration=solutions.parameters.sel(parameter="scale", \
                    stage=n_events).argmin('iteration').values)
                mags[:n_events+1], pars[:n_events+2] = nearest_solution.magnitudes.values, nearest_solution.parameters.values
                n_events += 1
                pars_accepted[:n_events+1] = pars[:n_events+1].copy()
                mags_accepted[:n_events] = mags[:n_events].copy()
                mags_accepted[n_events] = np.zeros(self.n_dims)
                j = 0
                if verbose:
                    print(f'Transition event {n_events} found around sample {int(np.round(np.sum(pars_accepted[:n_events,:].prod(axis=1))))}')
            if trace:#keep trace of algo
                all_mags_prop.append(mags[:n_events].copy())
                all_pars_prop.append(pars[:n_events+1].copy())
                all_mags.append(mags_accepted[:n_events].copy())
                all_pars.append(pars_accepted[:n_events+1].copy())
                all_diffs.append(np.abs(np.diff(mags[:n_events+1], axis=0)))

            j += 1
            #New parameter proposition
            pars_prop = pars[:n_events+2].copy()
            pars_prop[n_events,1] = step*j/self.shape
            last_stage = end/self.shape - np.sum(pars_prop[:n_events+1,1])
            pars_prop[n_events+1,1] = last_stage
            #New mag proposition
            mags_prop = mags_accepted[:n_events+1].copy()#cumulative
            mags_prop[n_events,:] = np.zeros(self.n_dims)
            time = np.sum(pars_prop[:n_events+1,1])*self.shape 
            pbar.update(int(np.round(time-prev_time)))
        pbar.update(step*2)
        if verbose:
            plt.show()
        mags = mags_accepted[:n_events, :]
        pars = pars_accepted[:n_events+1, :]
        if n_events > 1: 
            fit = self.fit_single(n_events, parameters=pars, magnitudes=mags, verbose=verbose)
        else:
            warn('Failed to find more than two stages, returning 2 stage model with default starting values')
            fit = self.fit_single(n_events)
        if trace:
            all_pars_aligned = np.tile(np.nan, (len(all_pars)+1, np.max([len(x) for x in all_pars]), 2))
            all_pars_prop_aligned = np.tile(np.nan, (len(all_pars_prop)+1, np.max([len(x) for x in all_pars_prop]), 2))
            all_mags_aligned = np.tile(np.nan, (len(all_mags)+1, np.max([len(x) for x in all_mags]), self.n_dims))
            all_mags_prop_aligned = np.tile(np.nan, (len(all_mags_prop)+1, np.max([len(x) for x in all_mags_prop]), self.n_dims))
            all_diffs_aligned = np.tile(np.nan, (len(all_diffs)+1, np.max([len(x) for x in all_diffs]), self.n_dims))
            for iteration, _i in enumerate(zip(all_pars, all_mags, all_mags_prop, all_pars_prop, all_diffs)):
                all_pars_aligned[iteration, :len(_i[0]), :] = _i[0]
                all_mags_aligned[iteration, :len(_i[1]), :] = _i[1]
                all_mags_prop_aligned[iteration, :len(_i[2]), :] = _i[2]
                all_pars_prop_aligned[iteration, :len(_i[3]), :] = _i[3]
                all_diffs_aligned[iteration, :len(_i[4]), :] = _i[4]
            traces = xr.Dataset({'parameters_accepted': (('iteration', 'stage_acc','parameter_acc'), 
                                         all_pars_aligned),
                                'magnitudes_accepted': (('iteration', 'event_acc','component_acc'), 
                                         all_mags_aligned),
                                'parameters_proposed': (('iteration', 'stage','parameter'), 
                                         all_pars_prop_aligned),
                                'magnitudes_proposed': (('iteration', 'event', 'component'), 
                                         all_mags_prop_aligned),
                                'difference_magnitudes':(('iteration','diffs', 'component'), 
                                         all_diffs_aligned)})
            return fit, traces
        else:
            return fit
    
