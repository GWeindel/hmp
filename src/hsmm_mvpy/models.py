'''

'''

import numpy as np
import xarray as xr
import multiprocessing as mp
import itertools
import math
import time#Just for speed testing
from warnings import warn, filterwarnings, resetwarnings
from scipy.stats import gamma as sp_gamma
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
default_colors =  ['cornflowerblue','indianred','orange','darkblue','darkgreen','gold', 'brown']


class hmp:
    
    def __init__(self, data, eeg_data=None, sfreq=None, offset=0, cpus=1, event_width=50, shape=2, estimate_magnitudes=True, estimate_parameters=True, template=None, location=None, distribution='gamma'):
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
        if distribution == 'gamma':
            from scipy.stats import gamma as sp_dist
        elif distribution == 'lognormal':
            from scipy.stats import lognorm as sp_dist
        elif distribution == 'wald':
            from scipy.stats import invgauss as sp_dist
        elif distribution == 'weibull':
            from scipy.stats import weibull_min as sp_dist
        else:
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
        bumbs : ndarray
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

    def fit_single(self, n_events=None, magnitudes=None, parameters=None, threshold=1, verbose=True,
            starting_points=1, parameters_to_fix=None, magnitudes_to_fix=None, method='random', multiple_n_events=None):
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
        import pandas as pd 
        if n_events is None:
            raise ValueError('The fit_single() function needs to be provided with a number of expected transition events. Look at function fit() if you want to fit a model without assuming a particular number of events.')
        if verbose:
            if parameters is None:
                print(f'Estimating {n_events} events model with {starting_points} starting point(s)')
            else:
                print(f'Estimating {n_events} events model')
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
        if starting_points > 0:#Initialize with equally spaced option
            if parameters is None:
                parameters = np.tile([self.shape, ((np.mean(self.durations))/(n_events+1)-self.location)/self.shape], (n_events+1,1))
            initial_p = parameters
            
            if magnitudes is None:
                magnitudes = np.zeros((n_events,self.n_dims), dtype=np.float64)
            initial_m = magnitudes
        
        if starting_points > 1:
            filterwarnings("ignore", category=RuntimeWarning)#Error in the generation of random see GH issue #38
            parameters = [initial_p]
            magnitudes = [initial_m]
            if method == 'random':
                for sp in np.arange(starting_points):
                    proposal_p = self.gen_random_stages(n_events, self.mean_d)
                    proposal_m = np.zeros((n_events,self.n_dims), dtype=np.float64)#Mags are NOT random but always 0
                    proposal_p[parameters_to_fix] = initial_p[parameters_to_fix]
                    proposal_m[magnitudes_to_fix] = initial_m[magnitudes_to_fix]
                    parameters.append(proposal_p)
                    magnitudes.append(proposal_m)
            elif method == 'grid':
                parameters = self._grid_search(n_events+1, iter_limit=starting_points, method='grid')
                magnitudes = np.zeros((len(parameters), n_events, self.n_dims), dtype=np.float64)
            else:
                raise ValueError('Unknown starting point method requested, use "random" or "grid"')
            with mp.Pool(processes=self.cpus) as pool:
                estimates = pool.starmap(self.EM, 
                    zip(itertools.repeat(n_events), magnitudes, parameters, itertools.repeat(1),\
                    itertools.repeat(magnitudes_to_fix),itertools.repeat(parameters_to_fix),))   
            lkhs_sp = [x[0] for x in estimates]
            mags_sp = [x[1] for x in estimates]
            pars_sp = [x[2] for x in estimates]
            eventprobs_sp = [x[3] for x in estimates]
            max_lkhs = np.where(lkhs_sp == np.max(lkhs_sp))[0][0]
            lkh = lkhs_sp[max_lkhs]
            mags = mags_sp[max_lkhs]
            pars = pars_sp[max_lkhs]
            eventprobs = eventprobs_sp[max_lkhs]
            resetwarnings()
            
        elif starting_points==1:#informed starting point
            lkh, mags, pars, eventprobs = self.EM(n_events, initial_m, initial_p,\
                                        threshold, magnitudes_to_fix, parameters_to_fix)

        else:#uninitialized    
            if np.any(parameters)== None:
                parameters = np.tile([self.shape, (self.mean_d)/self.shape], (n_events+1,1))
            if np.any(magnitudes)== None:
                magnitudes = np.zeros((n_events, self.n_dims), dtype=np.float64)
            lkh, mags, pars, eventprobs = self.EM(n_events, magnitudes, parameters,\
                                        threshold, magnitudes_to_fix, parameters_to_fix)
        if multiple_n_events is not None and len(pars) != multiple_n_events+1:#align all dimensions
            pars = np.concatenate((pars, np.tile(np.nan, (multiple_n_events+1-len(pars),2))))
            mags = np.concatenate((mags, np.tile(np.nan, 
                (multiple_n_events-len(mags), np.shape(mags)[1]))),axis=0)
            eventprobs = np.concatenate((eventprobs, np.tile(np.nan, (np.shape(eventprobs)[0],\
                    np.shape(eventprobs)[1], multiple_n_events-np.shape(eventprobs)[2]))),axis=2)
            n_events = multiple_n_events
        
        xrlikelihoods = xr.DataArray(lkh , name="likelihoods")
        xrparams = xr.DataArray(pars, dims=("stage",'parameter'), name="parameters", 
                        coords = [range(len(pars)), ['shape','scale']])
        xrmags = xr.DataArray(mags, dims=("event","component"), name="magnitudes",
                    coords = [range(len(mags)), range(np.shape(mags)[1])])
        part, trial = self.coords['participant'].values, self.coords['trials'].values
        if n_events>0:
            n_samples, n_participant_x_trials,_ = np.shape(eventprobs)
        else:
            n_samples, n_participant_x_trials = np.shape(eventprobs)
        if n_participant_x_trials >1 and n_events >0:
            xreventprobs = xr.Dataset({'eventprobs': (('event', 'trial_x_participant','samples'), 
                                         eventprobs.T)},
                         {'event':np.arange(n_events),
                          'samples':np.arange(n_samples),
                        'trial_x_participant':  pd.MultiIndex.from_arrays([part,trial],
                                names=('participant','trials'))})
            xreventprobs = xreventprobs.transpose('trial_x_participant','samples','event')
        elif n_events == 0:
            xreventprobs = xr.Dataset({'eventprobs': (('trial_x_participant','samples'), 
                                         eventprobs.T)},
                         {'samples':np.arange(n_samples),
                        'trial_x_participant':  pd.MultiIndex.from_arrays([part,trial],
                                names=('participant','trials'))})
            xreventprobs = xreventprobs.transpose('trial_x_participant','samples')

        elif n_participant_x_trials == 1: 
            xreventprobs = xr.Dataset({'eventprobs': (('event', 'trial_x_participant','samples'), 
                                         eventprobs.T)},
                         {'event':np.arange(n_events),
                          'samples':np.arange(n_samples)})
            xreventprobs = xreventprobs.transpose('trial_x_participant','samples','event')
        estimated = xr.merge((xrlikelihoods, xrparams, xrmags, xreventprobs))

        if verbose:
            print(f"Parameters estimated for {n_events} events model")
        return estimated
    
    def EM(self, n_events, magnitudes, parameters,  threshold, magnitudes_to_fix=None, parameters_to_fix=None, max_iteration = 1e3):
        '''
        Expectation maximization function underlying fit
        ''' 
        null_stages = np.where(parameters[:,1]<0)[0]
        wrong_shape = np.where(parameters[:,0]!=self.shape)[0]
        if len(null_stages)>0:
            raise ValueError(f'Wrong scale parameter input, provided scale parameter(s) {null_stages} should be positive but have value {parameters[null_stages,:].prod(axis=1)}')
        if len(wrong_shape)>0:
            raise ValueError(f'Wrong shape parameter input, provided parameter(s) {wrong_shape} shape is {parameters[wrong_shape,0]} but expected  expected {self.shape}')
        initial_parameters =  np.copy(parameters)
        initial_magnitudes = np.copy(magnitudes)
        
        lkh, eventprobs = self.estim_probs(magnitudes, parameters, n_events)
        means = np.zeros((self.max_d, self.n_trials, self.n_dims), dtype=np.float64)
        for trial in range(self.n_trials):
            means[:self.durations[trial],trial,:] = self.events[self.starts[trial]:self.ends[trial]+1,:]
            #Reorganize samples crosscorrelated with template on trial basis
        if threshold == 0 or n_events==0:
            lkh_prev = lkh
            magnitudes_prev = initial_magnitudes
            parameters_prev = initial_parameters
            eventprobs_prev = eventprobs
        else:
            for event in range(n_events):
                for comp in range(self.n_dims):
                    magnitudes[event,comp] = np.mean(np.sum( \
                        eventprobs[:,:,event]*means[:,:,comp], axis=0))
            parameters = self.scale_parameters(eventprobs, n_events)
            null_stages = np.where(parameters[:,1]<0)[0]
            if len(null_stages) == 0: 
                lkh_prev = -np.inf
                magnitudes_prev = magnitudes.copy()
                parameters_prev = parameters.copy()
                eventprobs_prev = eventprobs.copy()
            else:#Corner case in simulations
                lkh_prev = lkh
                magnitudes_prev = initial_magnitudes
                parameters_prev = initial_parameters
                eventprobs_prev = eventprobs
        i = 0
        while lkh - lkh_prev > threshold and i < max_iteration:#Expectation-Maximization algorithm
            #As long as new run gives better likelihood, go on  
            lkh_prev = lkh.copy()
            magnitudes_prev = magnitudes.copy()
            parameters_prev = parameters.copy()
            eventprobs_prev = eventprobs.copy()
            #Magnitudes from Expectation
            for event in range(n_events):
                for comp in range(self.n_dims):
                    magnitudes[event,comp] = np.mean(np.sum( \
                        eventprobs[:,:,event]*means[:,:,comp], axis=0))
                    # Scale cross-correlation with likelihood of the transition
                    # sum by-trial these scaled activation for each transition events
                    # average across trials

            magnitudes[magnitudes_to_fix,:] = initial_magnitudes[magnitudes_to_fix,:].copy()
            #Parameters from Expectation
            parameters = self.scale_parameters(eventprobs, n_events)
            parameters[parameters_to_fix, :] = initial_parameters[parameters_to_fix,:].copy()
            lkh, eventprobs = self.estim_probs(magnitudes, parameters, n_events)
            i += 1
        null_probs = np.where(eventprobs_prev<-1e-10)[0]
        if len(null_probs)>0:
            warn('Negative probabilities found after estimation, this likely refers to several events overlapping events. In case of simulated data ensure that events are enoughly separated (i.e. location parameter). In the case of real data this error is not expected, please report to the maintainers')
        if i == max_iteration:
            warn(f'Convergence failed, estimation hitted the maximum number of iteration ({int(max_iteration)})', RuntimeWarning)
        return lkh_prev, magnitudes_prev, parameters_prev, eventprobs_prev

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
        for stage in range(n_stages):
            if n_stages-1 > stage > 0:
                location = self.location
            else:
                location = 0
            pmf[:,stage] = self.distribution_pmf(parameters[stage,0], parameters[stage,1], location)
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
            eventprobs[eventprobs < 1e-10] = 0 #floating point precision error
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
    
    def scale_parameters(self, eventprobs, n_events):
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
        averagepos = np.concatenate([np.arange(1,self.max_d+1)@eventprobs.mean(axis=1),
                                     [self.mean_d]])#Durations
        params = np.zeros((n_events+1,2), dtype=np.float64)
        params[:,0] = self.shape
        params[:,1] = np.diff(averagepos, prepend=0)
        # params[[0.-1],1] += self.location
        params[0,1] -= .5#Event following starts half-sample before position
        params[-1,1] += .5# Last event terminates half-sample earlier
        params[:,1] = params[:,1]/params[:,0]
        return params
    
    def __multi_cpu_dispatch(self, list_n_events, list_mags, list_pars, threshold=1, verbose=False):
        if self.cpus > 1:
            if len(list_n_events) == 1:
                list_n_events = itertools.repeat(list_n_events)
            with mp.Pool(processes=self.cpus) as pool:
                event_loo_results = pool.starmap(self.fit_single, 
                    zip(list_n_events, list_mags, list_pars,
                        itertools.repeat(threshold),itertools.repeat(verbose)))
        else:
            event_loo_results = []
            for event_tmp, flat_tmp in zip(list_mags, list_pars):
                n_event = len(event_loo_results)+1
                event_loo_results.append(self.fit_single(n_event, event_tmp, flat_tmp, 0, False))
        return event_loo_results
        
    def loo_loglikelihood(self, estimates):
        event_loo_results = [estimates.copy()]
        n_events = event_loo_results[0].dropna('event').event.max().values
        list_values_n_events = [n_events]
        print(event_loo_results[0].parameters.values)  
        i = 0
        while n_events  > 0:
            print(f'Estimating all solutions for {n_events} number of events')
            temp_best = event_loo_results[i]#previous event solution
            temp_best = temp_best.dropna('event')
            temp_best = temp_best.dropna('stage')
            n_events_list = np.arange(n_events+1)#all events from previous solution
            flats = temp_best.parameters.values
            print(flats)
            events_temp, flats_temp = [], []
            for event in np.arange(n_events+1):#creating all possible solutions
                events_temp.append(temp_best.magnitudes.sel(event = np.array(list(set(n_events_list) - set([event])))).values)
                flat = event + 1 #one more flat than events
                temp = flats[:,1].copy()
                temp[flat-1] += temp[flat]
                temp = np.delete(temp, flat)
                flats_temp.append(np.reshape(np.concatenate([np.repeat(self.shape, len(temp)), temp]), (2, len(temp))).T)

            event_loo_likelihood_temp = self.__multi_cpu_dispatch(np.repeat(n_events,n_events+1), events_temp, 
                     flats_temp, 0, False)
            print([[x.likelihoods.values, x.parameters.values[:,1],'---------------------------------------\n'] for x in event_loo_likelihood_temp])
            print('---------------------------------------\n')
            models = xr.concat(event_loo_likelihood_temp, dim="iteration")
            event_loo_results.append(models.sel(iteration=[np.where(models.likelihoods == models.likelihoods.max())[0][0]]).squeeze('iteration'))
            n_events = event_loo_results[-1].dropna('event').event.max().values
            list_values_n_events.append(n_events)
            i += 1
        lkh = [x.likelihoods for x in event_loo_results]
        return lkh


    def backward_estimation(self,max_events=None, min_events=0, max_fit=None, max_starting_points=1, method="random", threshold=1):
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
            if max_starting_points >0:
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
            events_temp,flats_temp = [],[]
            for event in np.arange(n_events+1):#creating all possible solutions
                events_temp.append(temp_best.magnitudes.sel(event = np.array(list(set(n_events_list) - set([event])))).values)
                flat = event + 1 #one more flat than events
                temp = np.copy(flats[:,1])
                temp[flat-1] = temp[flat-1] + temp[flat]
                temp = np.delete(temp, flat)
                flats_temp.append(np.reshape(np.concatenate([np.repeat(self.shape, len(temp)), temp]), (2, len(temp))).T)
            if self.cpus > 1:
                with mp.Pool(processes=self.cpus) as pool:
                    event_loo_likelihood_temp = pool.starmap(self.fit_single, 
                        zip(itertools.repeat(n_events), events_temp, flats_temp,
                            itertools.repeat(threshold),itertools.repeat(False),itertools.repeat(1),\
                            itertools.repeat([]),itertools.repeat([]),\
                            itertools.repeat('random'),itertools.repeat(max_events)))
            else:
                raise ValueError('For loop not yet written use cpus >1')
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
        return int(np.min(self.durations)//(self.event_width_samples))

    def event_times(self, eventprobs, mean=True):
        '''
        Compute event onset times based on event probabilities
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

        eventprobs = estimates.eventprobs
        times = xr.dot(eventprobs, eventprobs.samples, dims='samples')#Most likely event location
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
        shifted_times = estimated.eventprobs.shift(samples=event_width_samples//2+1, fill_value=0).copy()#Shifts to compute channel topology at the peak of the event
        if extra_dim:
            data =  xr.dot(channels.rename({'epochs':'trials'}).\
                      stack(trial_x_participant=['participant','trials']).data.fillna(0).drop_duplicates('trial_x_participant'), \
                      shifted_times.fillna(0), dims=['samples']).\
                      transpose(extra_dim,'trial_x_participant','event','channels')
        else:
            data = xr.dot(channels.rename({'epochs':'trials'}).\
                      stack(trial_x_participant=['participant','trials']).data.fillna(0).drop_duplicates('trial_x_participant'), \
                      shifted_times.fillna(0), dims=['samples']).\
                      transpose('trial_x_participant','event','channels')
        if mean:
            data = data.mean('trial_x_participant')
        return data
    
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
        random_stages = np.array([[self.shape,x*mean_d/self.shape] for x in np.random.beta(2, 2, n_events+1)])
        return random_stages
    
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
        if binomcoeff(n_points-1, n_stages-1) > iter_limit:
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
                    print(f'Initial number of possibilities is {check_n_posibilities}. Given a number of max iteration = {iter_limit}: fitting {len(parameters)} models based on all possibilities from grid search with a spacing of {int(step)} samples and {n_points} points and durations of {grid}')
                else:
                    print(f'Fitting {len(parameters)} models using grid search')
            if method == 'grid':
                return parameters
            else:
                return parameters[np.argsort(parameters[:,0,1]),:,:]
        else:
            raise ValueError(f'No combination found given length of the data {self.mean_d}, number of events {n_stages} and a max iteration of {iter_limit}')
    
    def sliding_event(self, n_events=None, colors=default_colors, figsize=(12,3), verbose=True, method=None, plot_deriv=False, magnitudes=None, step=1):
        '''
        This method outputs the likelihood and estimated parameters of a 1 event model with each sample, from 0 to the mean 
        epoch duration. The parameters and likelihoods that are returned are 
        Take the highest likelihood, place a event by excluding event width space around it, follow with the next one
        '''
        
        from itertools import cycle
        
        mean_d = int(self.mean_d)
        init_n_events = n_events
        # if n_events == None:
        #     n_events = self.max_events
        parameters = self._grid_search(2, verbose=verbose, step=step)#Looking for all possibilities with one event
        if magnitudes is None:
            magnitudes = np.zeros((len(parameters),1, self.n_dims), dtype=np.float64)
        else:
            magnitudes = np.tile(magnitudes, (len(parameters),1, self.n_dims))
            method = 'single_event'
        lkhs_init, mags_init, pars_init, _ = \
            self.estimate_single_event(magnitudes, parameters, [0,1], [], 1)
        if verbose:
            _, ax = plt.subplots(figsize=figsize, dpi=300)
            ax.plot(pars_init[:,0,1]*self.shape, lkhs_init, '-', color='k')
        if method == 'derivative':
            deriv = np.gradient(lkhs_init)
            event_idx = np.where(np.diff(np.sign(deriv)) < 0)[0]
            n_events = len(event_idx)
            cycol = cycle(colors)
            colors = [next(cycol) for x in range(n_events)]
            pars = np.zeros((len(event_idx)+1, 2), dtype=np.float64)
            pars[:len(event_idx), :] = pars_init[:, 0, :][event_idx, :]
            pars[len(event_idx),:] = np.array([self.shape, mean_d/self.shape])#last stage defined as rt
            mags = mags_init[:, :, :][event_idx]
            lkhs = lkhs_init[event_idx]
            if verbose:
                for event in range(len(event_idx)):
                    ax.plot(np.array(pars)[event,1]*self.shape, lkhs[event], 'o', color=colors[event], label='Likelihood of Transition event %s'%(event+1))
                plt.ylabel('Log-likelihood')
                plt.xlabel('Sample number')
                plt.legend()
                plt.show()
                if plot_deriv:
                    _, ax = plt.subplots(figsize=figsize, dpi=300)
                    plt.plot(pars_init[:,0,1]*self.shape, np.gradient(lkhs_init), '-', color='k')
                    plt.hlines(0, 0, mean_d)
                    plt.show()
            
        elif method == 'estimation':
            if n_events is None:
                n_events = self.compute_max_events()
            lkhs_sp, mags_sp, pars_sp, eventprobs_sp = \
                self.estimate_single_event(np.zeros((len(parameters),1,self.n_dims), dtype=np.float64), \
                parameters, [], [], 1)
            lkhs_sp_sorting = lkhs_sp.copy()
            mags_sp_sorting = mags_sp.copy()
            pars_sp_sorting = pars_sp.copy()
            group_color = np.empty(len(lkhs_sp),dtype=str)
            max_lkhs = []
            for event in range(n_events):
                if not np.isnan(lkhs_sp_sorting).all():#Avoids problem if n_events > actual events
                    max_lkh = np.where(lkhs_sp_sorting == np.nanmax(lkhs_sp_sorting))[0][0]
                    max_lkhs.append(max_lkh)
                    gamma_max = pars_sp[max_lkh,0,1] 
                    neighbors = np.where(abs(pars_sp[:,0,1]-gamma_max) <= (self.event_width_samples/2)/self.shape)[0]
                    group = np.concatenate([[max_lkh], neighbors])
                    if verbose:
                        ax.plot(np.array(pars_sp)[group,0,1]*self.shape, lkhs_sp_sorting[group], 'o', color=colors[event])
                    lkhs_sp_sorting[group] = np.nan
                    pars_sp_sorting[group, :, :] = np.nan

            if np.isnan(lkhs_sp_sorting).all() and init_n_events != None and verbose:
                print(f'The number of requested events exceeds the number of convergence points found in the parameter space, events {event} to {n_events} were not found') 

            if not np.isnan(lkhs_sp_sorting).all() and verbose:#plot remaining points if all where not handled before
                ax.plot(pars_sp_sorting[:,0,1]*self.shape, lkhs_sp_sorting, 'o', color='gray', label='Not attributed')
                ax.legend()
            if verbose: 
                ax.set_xlabel('sample #')
                ax.set_ylabel('Loglikelihood')
                plt.show()
            pars = np.tile(np.nan, (n_events+1, 2))
            pars[:len(max_lkhs), :] = pars_sp[max_lkhs, 0, :]
            order = np.argsort(pars[:len(max_lkhs),1])#sorted index based on first stage duration
            pars[:len(max_lkhs), :] = pars[order, :]
            mags = mags_sp[max_lkhs][order]
            max_lkhs = np.array(max_lkhs)[order]
            pars[len(max_lkhs),:] = np.array([self.shape, mean_d/self.shape])#last stage defined as rt
            lkhs = np.repeat(np.nan, n_events)
            lkhs[:len(max_lkhs)] = lkhs_sp[max_lkhs]
        elif method is None:
            #pars, mags, lkhs = pars_init, mags_init, lkhs_init
            plt.ylabel('Log-likelihood')
            plt.xlabel('Sample number')
            plt.show()
        else:
            return pars, mags[:, 0, :], lkhs

    def estimate_single_event(self, magnitudes, parameters, parameters_to_fix, magnitudes_to_fix, threshold):
        if self.cpus >1:
            if np.shape(magnitudes) == 2:
                magnitudes = np.tile(magnitudes, (len(parameters), 1, 1))
            with mp.Pool(processes=self.cpus) as pool:
                estimates = pool.starmap(self.EM, 
                    zip(itertools.repeat(1), magnitudes, parameters, 
                        itertools.repeat(threshold), itertools.repeat(magnitudes_to_fix), 
                        itertools.repeat(parameters_to_fix)))
        else:
            estimates = []
            for pars, mags in zip(parameters, magnitudes):
                estimates.append(self.EM(1, mags, pars, threshold, magnitudes_to_fix, parameters_to_fix))
        lkhs_sp = np.array([x[0] for x in estimates])
        mags_sp = np.array([x[1] for x in estimates])
        pars_sp = np.array([x[2] for x in estimates])
        eventprobs_sp = np.array([x[3] for x in estimates])
        return lkhs_sp, mags_sp, pars_sp, eventprobs_sp
    
    def fit(self, step=1, verbose=True, figsize=(12,3), end=None, threshold=1, trace=False):
        '''
        '''
        if end is None:
            end = self.mean_d
        n_points = int(end//step)
        if threshold is None:
            means = np.array([np.mean(self.events[np.random.choice(range(len(self.events)), self.n_trials),:], axis=0) for x in range(1000)])
            threshold = np.abs(np.max(np.percentile(means, [0.01, 99.99], axis=0)))
        end = step*(n_points)#Rounding up to step size  
        lkh = -np.inf
        pars = np.zeros((n_points-1,2))
        pars[:,0] = self.shape
        pars[0,1] = 0.5#initialize with one event
        mags = np.zeros((n_points-1, self.n_dims))
        pbar = tqdm(total = end)
        n_events, j, time = 0,1,0
        last_stage = end
        if trace:
            all_pars, all_mags, all_mags_prop, all_pars_prop, all_diffs = [],[],[],[],[]
        pars_accepted, mags_accepted = pars.copy(), mags.copy()
        pars_prop = pars_accepted[:n_events+2].copy()#cumulative
        pars_prop[n_events,1] = step*j/self.shape
        last_stage = end/self.shape - np.sum(pars_prop[:n_events+1,1])
        pars_prop[n_events+1,1] = last_stage
        while last_stage*self.shape > self.location+step:
            prev_n_events, prev_lkh, prev_time = n_events, lkh, time
            mags_prop = mags[:n_events+1].copy()#cumulative
            lkh, mags[:n_events+1], pars[:n_events+2], _ = \
                self.EM(n_events+1, mags_prop, pars_prop.copy(), 1, [], [])
            signif = True# np.all(np.any(np.abs(mags[:n_events+1]) > threshold, axis=1))
            if n_events > 0:
                diffs = np.all(np.any(np.abs(np.diff(mags[:n_events+1], axis=0)) > threshold, axis=1))
            else:
                diffs = True
            if signif and diffs:
                n_events += 1
                pars_accepted = pars[:n_events+2].copy()
                mags_accepted = mags[:n_events+2].copy()
                mags_accepted[n_events] =  np.zeros(self.n_dims)
                j = 0
                if verbose:
                    print(f'Transition event {n_events} found around sample {int(np.round(np.sum(pars_accepted[:n_events,:].prod(axis=1))))}')#: Transition event samples = {np.round(pars[:n_events].prod(axis=1).cumsum())+self.location*np.arange(n_events)}')
            if trace:
                all_mags_prop.append(mags_prop.copy())
                all_pars_prop.append(pars_prop.copy())
                all_mags.append(mags_accepted[:n_events].copy())
                all_pars.append(pars_accepted[:n_events+1].copy())
                all_diffs.append(np.abs(np.diff(mags[:n_events+1], axis=0)))
            j += 1
            pars_prop = pars_accepted[:n_events+2].copy()
            pars_prop[n_events,1] = step*j/self.shape
            last_stage = end/self.shape - np.sum(pars_prop[:n_events+1,1])
            pars_prop[n_events+1,1] = last_stage
            time = np.sum(pars_prop[:n_events,1])*self.shape + \
                self.location*n_events + step*j/self.shape
            try:
                pbar.update(int(np.round(time-prev_time+1)))
            except:
                pbar.update(1)
        # pbar.update(int(np.round(end-prev_time)+self.location+step))
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
    
    def _bwd_fit(self, step=1, verbose=True, figsize=(12,3), end=None, threshold=None, bwd=True):
        '''
        '''
        if threshold is None:
            threshold = .05*self.n_dims
        if end is None:
            end = int(self.mean_d)
        n_points = int(end//step)
        lkh = np.repeat(-np.inf, n_points*3)
        pars, mags = np.zeros((n_points-1,2)),np.zeros((n_points-1, self.n_dims))
        new_pars, new_mags = pars.copy(), mags.copy()
        pbar = tqdm(total = n_points-1)
        n_events, i,j = 0,1,0
        event_width = self.event_width_samples / self.shape
        all_diffs = []
        while pars[-n_events:].prod(axis=1).sum() < self.mean_d:
            if bwd:
                index = range(-n_events-2,0)
            else:
                index = range(0,n_events+2)[::-1]
            if j == 0:
                print(np.round(pars[index[1:]].prod(axis=1).sum()))
                next_pars = self._grid_search(2, verbose=False, start_time=(np.round(pars[index[:n_events]].prod(axis=1).sum())), step=step, end_time=end)#next steps in parameter space
                if bwd:
                    next_pars = next_pars[:,::-1,:]
                    
                next_mags = np.zeros(self.n_dims)#reinitialisze mags
            if j < len(next_pars):
                pars_prop = np.concatenate([pars[index[2:]], next_pars[j]])
                mags_prop = np.concatenate([mags[index[2:]], [next_mags]])
                print(mags_prop)
                print(pars_prop)
                # next_pars[j, 1, 1] = end/self.shape-np.sum(pars_prop[:n_events+1, 1])
                lkh[i], new_mags[index[1:],:], new_pars[index,:], _ = \
                    self.EM(n_events+1, mags_prop.copy(), pars_prop.copy(), 1, [], [])
                print(new_mags[index[1:],:])
                print(new_pars[index,:])

                diffs = np.sum(np.diff(new_mags[index[1:]],axis=0, prepend=0)**2,axis=1)
                all_diffs.append(diffs[-1])
                print(diffs)
                # print(f'Times = {np.round(pars[:n_events].prod(axis=1).cumsum())}')
                if (diffs > threshold).all() and np.all(new_mags[index[1:]][-1]>0):#valid iteration
                    j = 0
                    n_events += 1
                    pars[index], mags[index] = new_pars[index], new_mags[index]
                    if verbose:
                        print(f'Transition event {n_events} found around time {np.round(np.sum(pars[index[1:]].prod(axis=1)))} (step {i})')
                    print(f'New pars : {pars[index[0]]}')
                    print(int((pars[-n_events,1] - np.sum(pars_prop[index[1:],1]))*self.shape))
                    pbar.update(int((np.sum(pars[-n_events:,1]) - np.sum(pars_prop[-n_events:,1]))*self.shape))
                else:
                    pbar.update(1)
                    j += 1
                i += 1
            else:
                break
        all_diffs = np.array(all_diffs)
        plt.hist(all_diffs, bins=40)
        plt.xlim(0, threshold+threshold/2)
        plt.show()
        plt.plot(all_diffs)
        plt.show()
        mags = mags[index[2:], :]
        pars = pars[index[1:], :]
        pars[-1, :] = np.concatenate([[self.shape], [self.mean_d/self.shape-np.sum(pars[:-1, 1])]])
        # if pars[-1, 1] <= 0:
        #     print(True)
        #     pars[-1, 1] = .5
        if verbose:
            _, ax = plt.subplots(figsize=figsize, dpi=300)
            ax.plot(lkh, '-', color='k')
            ax.set_ylabel('Log-likelihood')
            ax.set_xlabel('Sample number')
            plt.show()
        print(mags)
        print(pars)
        fit = self.fit_single(len(pars)-1, parameters=pars, magnitudes=mags)
        return fit, lkh    
    
    def event_gain_plot(self, lkh, pars, mags, colors=default_colors, figsize=(12,3)):
        
        from itertools import cycle
        n_events = len(mags)
        _, ax = plt.subplots(figsize=figsize, dpi=300)
        cycol = cycle(colors)
        colors = [next(cycol) for x in range(n_events)]
        parameters = self._grid_search(2, verbose=True)
        for event in range(n_events):
            event_lkh = np.zeros(len(parameters))
            iteration = 0
            print(mags[event,:])
            for pars_event in parameters:
                event_lkh[iteration], _, _, _ = \
                    self.EM(1,  np.array([mags[event,:]]), pars_event,1, [0], [0,1])
                iteration += 1
            ax.plot(parameters[:,0,1], event_lkh, color=colors[event])
        plt.show()

