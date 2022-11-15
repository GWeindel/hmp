'''

'''

import numpy as np
import xarray as xr
import multiprocessing as mp
import itertools
import math
from warnings import warn

class hsmm:
    
    def __init__(self, data, sf, cpus=1, bump_width = 50, shape=2, estimate_magnitudes=True, estimate_parameters=True):
        '''
        HSMM calculates the probability of data summing over all ways of 
        placing the n bumps to break the trial into n + 1 flats.

        Parameters
        ----------
        data : ndarray
            2D ndarray with n_samples * components 
        starts : ndarray
            1D array with start of each trial
        ends : ndarray
            1D array with end of each trial
        sf : int
            Sampling frequency of the signal (initially 100)
        width : int
            width of bumps in milliseconds, originally 5 samples
        '''
        durations = data.unstack().sel(component=0).swap_dims({'epochs':'trials'})\
            .stack(trial_x_participant=['participant','trials']).dropna(dim="trial_x_participant",\
            how="all").groupby('trial_x_participant').count(dim="samples").cumsum().squeeze()
        dur_dropped_na = durations.dropna("trial_x_participant")
        starts = np.roll(dur_dropped_na.data, 1)
        starts[0] = 0
        self.starts = starts
        self.ends = dur_dropped_na.data-1 
        self.durations =  self.ends-self.starts+1
        self.named_durations =  durations.dropna("trial_x_participant") - durations.dropna("trial_x_participant").shift(trial_x_participant=1, fill_value=0)
        self.sf = sf
        self.tseps = 1000/self.sf
        self.n_trials = len(self.durations)  
        self.bump_width = bump_width
        self.cpus = cpus
        self.bump_width_samples = int(self.bump_width * (self.sf/1000))
        self.offset = self.bump_width_samples//2#offset on data linked to the choosen width how soon the first peak can be or how late the last,
        self.coords = durations.reset_index('trial_x_participant').coords
        data = data.data.T
        self.n_samples, self.n_dims = np.shape(data)
        self.bumps = self.calc_bumps(data)#adds bump morphology
        self.max_d = np.max(self.durations)
        self.max_bumps = self.compute_max_bumps()
        self.shape = shape
        self.estimate_magnitudes = estimate_magnitudes
        self.estimate_parameters = estimate_parameters
    
    def calc_bumps(self,data):
        '''
        This function puts on each sample the correlation of that sample and the previous
        five samples with a Bump morphology on time domain.  Will be used fot the likelihood 
        of the EEG data given that the bumps are centered at each time point

        Parameters
        ----------
        data : ndarray
            2D ndarray with n_samples * components

        Returns
        -------
        bumbs : ndarray
            a 2D ndarray with samples * PC components where cell values have
            been correlated with bump morphology
        '''
        bump_idx = np.arange(0,self.bump_width_samples)*self.tseps+self.tseps/2
        bump_frequency = 1000/(self.bump_width*2)#gives bump frequency given that bumps are defined as half-sines
        template = np.sin(2*np.pi*np.linspace(0,1,1000)*bump_frequency)[[int(x) for x in bump_idx]]#bump morph based on a half sine with given bump width and sampling frequency #previously np.array([0.3090, 0.8090, 1.0000, 0.8090, 0.3090]) 
        
        template = template/np.sum(template**2)#Weight normalized to sum(P) = 1.294
        
        bumps = np.zeros(data.shape)

        for j in np.arange(self.n_dims):#For each PC
            temp = np.zeros((self.n_samples,self.bump_width_samples))
            temp[:,0] = data[:,j]#first col = samples of PC
            for i in np.arange(1,self.bump_width_samples):
                temp[:,i] = np.concatenate((temp[1:, i-1], [0]), axis=0)
                # puts the component in a [n_samples X length(bump)] matrix shifted.
                # each column is a copy of the first one but shifted one sample
                # upwards
            bumps[:,j] = temp @ template
            # for each PC we calculate its correlation with bump temp(data samples * 5) *  
            # template(sine wave bump in samples - 5*1)
        bumps[self.offset:,:] = bumps[:-self.offset,:]#Centering
        bumps[:self.offset,:] = 0 #Centering
        bumps[-self.offset:,:] = 0 #Centering
        return bumps

    def fit_single(self, n_bumps, magnitudes=None, parameters=None, threshold=1, mp=False, verbose=True, starting_points=1,
                  parameters_to_fix = [], magnitudes_to_fix = [], method='random'):
        '''
        Fit HsMM for a single n_bumps model

        Parameters
        ----------
        n_bumps : int
            how many bumps are estimated
        magnitudes : ndarray
            2D ndarray components * n_bumps, initial conditions for bumps magnitudes
        parameters : list
            list of initial conditions for Gamma distribution scale parameter. If parameters are estimated, the list provided is used as starting point,
            if parameters are fixed, parameters estimated will be the same as the one provided. When providing a list, stage need to be in the same order
            _n_th gamma parameter is  used for the _n_th stage
        threshold : float
            threshold for the HsMM algorithm, 0 skips HsMM

        '''
        import pandas as pd 
        if verbose:
            print(f'Estimating {n_bumps} bumps model with {starting_points-1} random starting points')
        if mp==True: #PCG: Dirty temporarilly needed for multiprocessing in the iterative backroll estimation...
            magnitudes = magnitudes.T
        
        if self.estimate_magnitudes == False:#Don't need to manually fix pars if not estimated
            magnitudes_to_fix = np.arange(n_bumps+1)
        if self.estimate_parameters == False:#Don't need to manually fix pars if not estimated
            parameters_to_fix = np.arange(n_bumps+1)            
        #Formatting parameters
        if isinstance(parameters, (xr.DataArray,xr.Dataset)):
            parameters = parameters.dropna(dim='stage').values
        if isinstance(magnitudes, (xr.DataArray,xr.Dataset)):
            magnitudes = magnitudes.dropna(dim='bump').values  
        if starting_points > 0:#Initialize with equally spaced option
            if np.any(parameters) == None:
                parameters = np.tile([self.shape, math.ceil(np.mean(self.durations)/(n_bumps+1)/self.shape)], (n_bumps+1,1))
            initial_p = parameters
            
            if np.any(magnitudes) == None:
                magnitudes = np.zeros((self.n_dims,n_bumps))
            initial_m = magnitudes
        
        if starting_points > 1:
            import multiprocessing as mp
            parameters = [initial_p]
            magnitudes = [initial_m]
            if method == 'random':
                for sp in np.arange(starting_points):
                    proposal_p = self.gen_random_stages(n_bumps, np.mean(self.durations))
                    proposal_m = np.zeros((self.n_dims,n_bumps))#Mags are NOT random but always 0
                    proposal_p[parameters_to_fix] = initial_p[parameters_to_fix]
                    proposal_m[magnitudes_to_fix] = initial_m[magnitudes_to_fix]
                    parameters.append(proposal_p)
                    magnitudes.append(proposal_m)
            elif method == 'grid':
                parameters = self.grid_search(n_bumps+1, starting_points)
                magnitudes = np.zeros((len(parameters), self.n_dims,n_bumps))
            else:
                raise ValueError('Unknown starting point method requested, use "random" or "grid"')
            with mp.Pool(processes=self.cpus) as pool:
                estimates = pool.starmap(self.fit, 
                    zip(itertools.repeat(n_bumps), magnitudes, parameters, itertools.repeat(1),\
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
            
        elif starting_points==1:#informed starting point
            lkh, mags, pars, eventprobs = self.fit(n_bumps, initial_m, initial_p,\
                                        threshold, magnitudes_to_fix, parameters_to_fix)

        else:#uninitialized    
            if np.any(parameters)== None:
                parameters = np.tile([self.shape, self.durations.mean()/2], (n_bumps+1,1))
            if np.any(magnitudes)== None:
                magnitudes = np.zeros((self.n_dims,n_bumps))
            lkh, mags, pars, eventprobs = self.fit(n_bumps, magnitudes, parameters,\
                                        threshold, magnitudes_to_fix, parameters_to_fix)
        
        if len(pars) != self.max_bumps+1:#align all dimensions
            pars = np.concatenate((pars, np.tile(np.nan, (self.max_bumps+1-len(pars),2))))
            mags = np.concatenate((mags, np.tile(np.nan, (np.shape(mags)[0], \
                self.max_bumps-np.shape(mags)[1]))),axis=1)
            eventprobs = np.concatenate((eventprobs, np.tile(np.nan, (np.shape(eventprobs)[0],\
                                        np.shape(eventprobs)[1], self.max_bumps-np.shape(eventprobs)[2]))),axis=2)
        
        xrlikelihoods = xr.DataArray(lkh , name="likelihoods")
        xrparams = xr.DataArray(pars, dims=("stage",'params'), name="parameters")
        xrmags = xr.DataArray(mags, dims=("component","bump"), name="magnitudes")
        #xreventprobs = xr.DataArray(eventprobs, dims=("samples",'trialxPart','bump'), name="eventprobs")
        #xreventprobs = xreventprobs.trialxPart.set_index(self.coords.values)
        part, trial = self.coords['participant'].values, self.coords['trials'].values

        n_samples, n_participant_x_trials, bumps_n = np.shape(eventprobs)
        xreventprobs = xr.Dataset({'eventprobs': (('bump', 'trial_x_participant','samples'), 
                                         eventprobs.T)},
                         {'bump':np.arange(bumps_n),
                          'samples':np.arange(n_samples),
                        'trial_x_participant':  pd.MultiIndex.from_arrays([part,trial],
                                names=('participant','trials'))})
        #xreventprobs = xreventprobs.unstack('trial_x_participant')
        xreventprobs = xreventprobs.transpose('trial_x_participant','samples','bump')
        estimated = xr.merge((xrlikelihoods, xrparams, xrmags, xreventprobs))

        if verbose:
            print(f"Parameters estimated for {n_bumps} bumps model")
        return estimated
        
    def fit(self, n_bumps, magnitudes, parameters,  threshold, magnitudes_to_fix=[], parameters_to_fix=[]):
        '''
        Fitting function underlying single and iterative fit
        '''
        lkh1 = -np.inf#initialize likelihood     
        lkh, eventprobs = self.calc_EEG_50h(magnitudes, parameters, n_bumps)
        if threshold == 0:
            lkh1 = np.copy(lkh)
            magnitudes1 = np.copy(magnitudes)
            parameters1 = np.copy(parameters)
            eventprobs1 = np.copy(eventprobs)
        else : 
            means = np.zeros((self.max_d, self.n_trials, self.n_dims))
            for i in np.arange(self.n_trials):
                means[:self.durations[i],i,:] = self.bumps[self.starts[i]:self.ends[i]+1,:]
                # arrange bumps dimensions by trials [max_d*trial*PCs]
            while (lkh - lkh1) > threshold:
                #As long as new run gives better likelihood, go on  
                lkh1 = np.copy(lkh)
                magnitudes1 = np.copy(magnitudes)
                parameters1 = np.copy(parameters)
                eventprobs1 = np.copy(eventprobs)
                for i in np.arange(n_bumps):
                    for j in np.arange(self.n_dims):
                        magnitudes[j,i] = np.mean(np.sum( \
                        eventprobs[:,:,i]*means[:,:,j], axis=0))
                        # 2) sum of all samples in a trial
                        # 3) mean across trials of the sum of samples in a trial
                        # repeated for each PC (j) and later for each bump (i)
                        # magnitudes [nPCAs, nBumps]
                    if i in magnitudes_to_fix:
                        magnitudes[:,i] = magnitudes1[:,i]
                parameters = self.gamma_parameters(eventprobs, n_bumps)

                #Ensure constrain of gammas > bump_width, note that contrary to the matlab code this is not applied on the first and last stages
                for i in np.arange(n_bumps+1): #PCG: seems unefficient likely slows down process, isn't there a better way to bound the estimation??
                    if i in parameters_to_fix:
                        parameters[i,:] = parameters1[i,:]
                    if 0 < i < n_bumps+1 and parameters[i,:].prod() < self.bump_width_samples:
                        # multiply scale and shape parameters to get 
                        # the mean distance of the gamma-2 pdf. 
                        # It constrains that bumps are separated at 
                        # least a bump length
                        parameters[i,:] = parameters1[i,:]
                lkh, eventprobs = self.calc_EEG_50h(magnitudes, parameters, n_bumps)
        return lkh1, magnitudes1, parameters1, eventprobs1


    def calc_EEG_50h(self, magnitudes, parameters, n_bumps, lkh_only=False):
        '''
        Defines the likelihood function to be maximized as described in Anderson, Zhang, Borst and Walsh, 2016

        Returns
        -------
        likelihood : float
            likelihoods
        eventprobs : ndarray
            [samples(max_d)*n_trials*n_bumps] = [max_d*trials*nBumps]
        '''
        if isinstance(parameters, (xr.DataArray,xr.Dataset)):
            parameters = parameters.dropna(dim='stage').values
        if isinstance(magnitudes, (xr.DataArray,xr.Dataset)):
            magnitudes = magnitudes.dropna(dim='bump').values
        gains = np.zeros((self.n_samples, n_bumps))

        for i in np.arange(self.n_dims):
            # computes the gains, i.e. how much the bumps reduce the variance at 
            # the location where they are placed for all samples, see Appendix Anderson,Zhang, 
            # Borst and Walsh, 2016, last equation, right hand side parenthesis 
            # (S^2 -(S -B)^2) (Sb- B2/2). And sum over all PCA
            gains = gains + self.bumps[:,i][np.newaxis].T * magnitudes[i,:] - \
                    np.tile((magnitudes[i,:]**2),(self.n_samples,1))/2 
            # bump*magnitudes-> gives [n_samples*nBumps] It scales bumps prob. by the
            # global magnitudes of the bumps topology 'magnitudes' of each bump. 
            # tile append vertically the (estimated bump-magnitudes)^2 of one PC 
            # for all samples divided by 2.
            # gain(n,sum(pca)) = gain(n,pca) + corrBump(n,pca) * 
            # estBumpsMorph(pca,bumps) - (estBumpMorph(pca,bumps)^2)/2
            # sum for all PCs of the 'normalized' correlation of P(having a sin)
            # and bump morphology
        gains = np.exp(gains)
        probs = np.zeros([self.max_d,self.n_trials,n_bumps]) # prob per trial
        probs_b = np.zeros([self.max_d,self.n_trials,n_bumps])
        for i in np.arange(self.n_trials):
            # Following assigns gain per trial to variable probs 
            # in direct and reverse order
            probs[:self.ends[i] - self.starts[i]+1,i,:] = \
                gains[self.starts[i]: self.ends[i]+1,:] 
            for j in np.arange(n_bumps): # PCG: for-loop IMPROVE
                probs_b[:self.ends[i]- self.starts[i]+1 ,i,j] = \
                np.flipud(gains[self.starts[i]: self.ends[i]+1,\
                n_bumps-j-1])
                # assign reversed gains array per trial

        LP = np.zeros([self.max_d, n_bumps + 1]) # Gamma pdf for each stage parameters
        for j in np.arange(n_bumps + 1):
            LP[:,j] = self.gamma_EEG(parameters[j,0], parameters[j,1], self.max_d)
            # Compute Gamma pdf from 0 to max_d with parameters
        BLP = LP[:,::-1] # States reversed gamma pdf
        forward = np.zeros((self.max_d, self.n_trials, n_bumps))
        forward_b = np.zeros((self.max_d, self.n_trials, n_bumps))
        backward = np.zeros((self.max_d, self.n_trials, n_bumps))
        # eq1 in Appendix, first definition of likelihood
        # For each trial (given a length of max duration) compute gamma pdf * gains
        # Start with first bump as first stage only one gamma and no bumps
        forward[:self.max_d,:,0] = np.tile(LP[:self.max_d,0][np.newaxis].T,\
            (1,self.n_trials))*probs[:self.max_d,:,0]

        forward_b[:self.max_d,:,0] = np.tile(BLP[:self.max_d,0][np.newaxis].T,\
                    (1,self.n_trials)) # reversed Gamma pdf

        for i in np.arange(1,n_bumps):#continue with other bumps
            next_ = np.concatenate((np.zeros(self.bump_width_samples), LP[:self.max_d - \
                    self.bump_width_samples, i]), axis=0)
            # next_ bump width samples followed by gamma pdf (one state)
            next_b = np.concatenate((np.zeros(self.bump_width_samples), BLP[:self.max_d - \
                    self.bump_width_samples, i]), axis=0)
            # next_b same with reversed gamma
            add_b = forward_b[:,:,i-1] * probs_b[:,:,i-1]
            for j in np.arange(self.n_trials):
                temp = np.convolve(forward[:,j,i-1],next_)
                # convolution between gamma * gains at previous states and state i
                forward[:,j,i] = temp[:self.max_d]
                temp = np.convolve(add_b[:,j],next_b)
                # same but backwards
                forward_b[:,j,i] = temp[:self.max_d]
            forward[:,:,i] = forward[:,:,i] * probs[:,:,i]
        forward_b = forward_b[:,:,::-1] # undoes inversion
        for j in np.arange(self.n_trials): # TODO : IMPROVE
            for i in np.arange(n_bumps):
                backward[:self.durations[j],j,i] = np.flipud(forward_b[:self.durations[j],j,i])
        backward[:self.offset,:,:] = 0
        temp = forward * backward # [max_d,n_trials,n_bumps] .* [max_d,n_trials,n_bumps];
        likelihood = np.sum(np.log(temp[:,:,0].sum(axis=0)))# why 0 index? shouldn't it also be for all dim??
        # sum(log(sum of 'temp' by columns, samples in a trial)) 
        eventprobs = temp / np.tile(temp.sum(axis=0), [self.max_d, 1, 1])
        #normalization [-1, 1] divide each trial and state by the sum of the n points in a trial
        if lkh_only:
            return likelihood
        else:
            return [likelihood, eventprobs]

    def gamma_parameters(self, eventprobs, n_bumps):
        '''
        Given that the shape is fixed the calculation of the maximum likelihood
        scales becomes simple.  One just calculates the means expected lengths 
        of the flats and divides by the shape

        Parameters
        ----------
        eventprobs : ndarray
            [samples(max_d)*n_trials*n_bumps] = [max_d*trials*nBumps]
        durations : ndarray
            1D array of trial length
        mags : ndarray
            2D ndarray components * nBumps, initial conditions for bumps magnitudes
        shape : float
            shape parameter for the gamma, defaults to 2  

        Returns
        -------
        params : ndarray
            shape and scale for the gamma distributions
        '''
        width = self.bump_width_samples #unaccounted samples -1?
        # Expected value, time location
        averagepos = np.hstack((np.sum(np.tile(np.arange(self.max_d)[np.newaxis].T,\
            (1, n_bumps)) * np.mean(eventprobs, axis=1).reshape(self.max_d, n_bumps,\
                order="F"), axis=0), np.mean(self.durations)))
        # 1) mean accross trials of eventprobs -> mP[max_l, nbump]
        # 2) global expected location of each bump
        # concatenate horizontaly to last column the length of each trial
        averagepos = averagepos - (self.offset+np.hstack(np.asarray([\
                np.append(np.arange(0,n_bumps*width,width),n_bumps*width-self.offset)],dtype='object')))
        # correction for time locations with number of bumps and size in samples
        flats = averagepos - np.hstack((0,averagepos[:-1]))
        params = np.zeros((n_bumps+1,2))
        params[:,0] = self.shape #PCG shape is hardcoded
        params[:,1] = flats.T / self.shape
        # correct flats between bumps for the fact that the gamma is 
        # calculated at midpoint
        #params[:,1] = params[:,1] - .5 /self.shape
        # first flat is bounded on left while last flat may go 
        # beyond on right
        params[0,1] = params[0,1] - .5 /self.shape
        params[-1,1] = params[-1,1] - .5 /self.shape
        return params

    def backward_estimation(self,max_fit=None, max_starting_points=1, method="random"):
        '''
        First read or estimate max_bump solution then estimate max_bump - 1 solution by 
        iteratively removing one of the bump and pick the one with the highest 
        likelihood
        
        Parameters
        ----------
        max_fit : xarray
            To avoid re-estimating the model with maximum number of bumps it can be provided 
            with this arguments, defaults to None
        max_starting_points: int
            how many random starting points iteration to try for the model estimating the maximal number of bumps
        
        '''
        if not max_fit:
            if max_starting_points >0:
                print(f'Estimating all solutions for maximal number of bumps ({self.max_bumps}) with 1 pre-defined starting point and {max_starting_points-1} {method} starting points')
            bump_loo_results = [self.fit_single(self.max_bumps, starting_points=max_starting_points, method=method, verbose=False)]
        else:
            bump_loo_results = [max_fit]
        i = 0
        for n_bumps in np.arange(self.max_bumps-1,0,-1):
            print(f'Estimating all solutions for {n_bumps} number of bumps')
            temp_best = bump_loo_results[i]#previous bump solution
            temp_best = temp_best.dropna('bump')
            temp_best = temp_best.dropna('stage')
            n_bumps_list = np.arange(n_bumps+1)#all bumps from previous solution
            flats = temp_best.parameters.values
            bumps_temp,flats_temp = [],[]
            for bump in np.arange(n_bumps+1):#creating all possible solutions
                bumps_temp.append(temp_best.magnitudes.sel(bump = np.array(list(set(n_bumps_list) - set([bump])))).values.T)
                flat = bump + 1 #one more flat than bumps
                temp = np.copy(flats[:,1])
                temp[flat-1] = temp[flat-1] + temp[flat]
                temp = np.delete(temp, flat)
                flats_temp.append(np.reshape(np.concatenate([np.repeat(self.shape, len(temp)), temp]), (2, len(temp))).T)
            if self.cpus > 1:
                with mp.Pool(processes=self.cpus) as pool:
                    bump_loo_likelihood_temp = pool.starmap(self.fit_single, 
                        zip(itertools.repeat(n_bumps), bumps_temp, flats_temp,
                            itertools.repeat(1),itertools.repeat(True),itertools.repeat(False)))
            else:
                raise ValueError('For loop not yet written use cpus >1')
            models = xr.concat(bump_loo_likelihood_temp, dim="iteration")
            bump_loo_results.append(models.sel(iteration=[np.where(models.likelihoods == models.likelihoods.max())[0][0]]).squeeze('iteration'))
            i+=1
        bests = xr.concat(bump_loo_results, dim="n_bumps")
        bests = bests.assign_coords({"n_bumps": np.arange(self.max_bumps,0,-1)})
        #bests = bests.squeeze('iteration')
        return bests

    
    @staticmethod
    def gamma_EEG(a, b, max_length):
        '''
        Returns PDF of gamma dist with shape = a and scale = b, 
        on a range from 0 to max_length 

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
        from scipy.stats import gamma
        d = [gamma.pdf(t+.5,a,scale=b) for t in np.arange(max_length)]
        d = d/np.sum(d)
        return d
    
    def gen_random_stages(self, n_bumps, mean_rt):
        '''
        Returns random stage duration between 0 and mean RT by iteratively drawind sample from a 
        uniform distribution between the last stage duration (equal to 0 for first iteration) and 1.
        Last stage is equal to 1-previous stage duration.
        The stages are then scaled to the mean RT
        Parameters
        ----------
        n_bumps : int
            how many bumps
        mean_rt : float
            scale parameter
        Returns
        -------
        random_stages : ndarray
            random partition between 0 and mean_rt
        '''
        #random_stages = [0]
        #for stage in np.arange(n_bumps):
        random_stages= np.array([[2,x*mean_rt/2] for x in np.random.beta(2, 2, n_bumps+1)])
        #random_stages.append(1)#last one is defined as 1 - previous
        #random_stages = np.diff(random_stages)
        #random_stages = np.array([[self.shape, np.round(x*mean_rt)+1] for x in random_stages])#Remove 0 duration stage
        return random_stages
    
    def compute_max_bumps(self):
        '''
        Compute the maximum possible number of bumps given bump width and mean or minimum reaction time
        '''
        return int(np.round(np.min(self.durations)/self.bump_width_samples))
        # if not min_rt:
        #     return int(np.mean(self.durations)/self.bump_width_samples)
        # else:
        #     return int(np.min(self.durations)/self.bump_width_samples)

    def bump_times(self, eventprobs, mean=True):
        '''
        Compute bump onset times based on bump probabilities

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
        warn('This method is deprecated and will be removed in future version, use onset_times() instead', DeprecationWarning, stacklevel=2)
        eventprobs = eventprobs.dropna('bump', how="all")
        eventprobs = eventprobs.dropna('trial_x_participant', how="all")
        onsets = np.empty((len(eventprobs.trial_x_participant),len(eventprobs.bump)+1))
        i = 0
        for trial in eventprobs.trial_x_participant.dropna('trial_x_participant', how="all").values:
            onsets[i, :len(eventprobs.bump)] = np.arange(self.max_d) @ eventprobs.sel(trial_x_participant=trial).data - self.bump_width_samples/2#Correcting for centerning, thus times represents bump onset
            onsets[i, -1] = self.ends[i] - self.starts[i]
            i += 1
        if mean:
            return np.mean(onsets, axis=0)
        else:
            return onsets
        
    def compute_times(self, estimates, duration=False, fill_value=None, mean=False, cumulative=False, add_rt=False):
        '''
        Compute the likeliest onset times for each bump

        Parameters
        ----------
        estimates :
            Estimated instance of an hsmm model
        init : 
            Initialized HsMM object  
        duration : bool
            Whether to compute onset times (False) or stage duration (True)
        fill_value : float | ndarray
            What value to fill for the first onset/duration

        Returns
        -------
        times : xr.DataArray
            Bump onset or stage duration with bump and trial_x_participant dimensions
        '''

        eventprobs = estimates.eventprobs
        eventprobs = eventprobs.dropna('bump')
        times = xr.dot(eventprobs, eventprobs.samples, dims='samples')#Most likely bump location
        times[:] = times - self.bump_width_samples/2#Correcting for centerning, thus times represents bump onset
        if duration: fill_value=0
        if fill_value != None:            #times = times.shift(bump=1, fill_value=fill_value)
            added = xr.DataArray(np.repeat(fill_value,len(times.trial_x_participant))[np.newaxis,:],
                                 coords={'bump':[0], 
                                         'trial_x_participant':times.trial_x_participant})
            times = times.assign_coords(bump=times.bump+1)
            times = times.combine_first(added)
        if add_rt:             
            rts = self.named_durations
            rts = rts.assign_coords(bump=int(times.bump.max().values+1))
            rts = rts.expand_dims(dim="bump")
            times = xr.concat([times, rts], dim='bump')
        if duration:
            #adding reaction time and treating it as the last bump
            times = times.rename({'bump':'stage'})
            if not cumulative:
                times = times.diff(dim='stage')
        if mean:
            times = times.mean('trial_x_participant')
        return times
   
    
    def compute_topo(self, data, eventprobs, mean=True):
        '''
        DEPRECATED
        '''
        warn('This method is deprecated and will be removed in future version, use onset_times() instead', DeprecationWarning, stacklevel=2)
        if 'trial_x_participant' not in data:
            data = data.stack(trial_x_participant=['participant','epochs'])
        eventprobs = eventprobs.dropna('bump', how="all")
        eventprobs = eventprobs.dropna('trial_x_participant', how="all")
        topologies = np.empty((len(eventprobs.trial_x_participant.data), 
                               len(eventprobs.bump.data),
                               len(data.electrodes.data)))
        i = 0
        for trial_x_participant in eventprobs.trial_x_participant:
            z = 0
            for bump in eventprobs.bump:
                trial_samples = np.arange(self.named_durations.sel(trial_x_participant=trial_x_participant).data)
                topologies[i,z, :] = data.sel(trial_x_participant = trial_x_participant,
                                              samples=trial_samples).data @ \
                    eventprobs.sel(trial_x_participant = trial_x_participant,  bump=bump, samples=trial_samples)
                z += 1
            i += 1
        if mean:
            return np.mean(topologies, axis=0)
        else:
            return topologies
        
    def multiple_topologies(self, data, eventprobs, mean=True):
        '''
        DEPRECATED
        '''
        warn('This method is deprecated and will be removed in future version, use onset_times() instead', DeprecationWarning, stacklevel=2)        
        topo = np.tile(np.nan, (len(eventprobs.n_bumps), len(eventprobs.n_bumps), len(data.electrodes)))
        for n_bumps in eventprobs.n_bumps:
            topo[n_bumps-1, :n_bumps.values, :] = self.compute_topo(data, eventprobs.sel(n_bumps=n_bumps))
        return topo
    
    def grid_search(self, n_stages, iter_limit=1e3):
        '''
        This function decomposes the mean RT into a grid with points. Ideal case is to have a grid with one sample = one search point but the number
        of possibilities badly scales with the length of the RT and the number of stages. Therefore the iter_limit is used to select an optimal number
        of points in the grid with a given spacing. After having defined the grid, the function then generates all possible combination of 
        bump placements within this grid. It is faster than using random points (both should converge) but depending on the mean RT and the number 
        of bumps to look for, the number of combination can be really large. 
        
        Parameters
        ----------
        n_stages : int
            how many bump to look for
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

        mean_rt = self.durations.mean()
        bumps_width = self.bump_width_samples
        n_points = int(mean_rt - bumps_width*(n_stages-1))
        check_n_posibilities = binomcoeff(n_points-1, n_stages-1)
        while binomcoeff(n_points-1, n_stages-1) > iter_limit:
            n_points = n_points-1
        spacing = mean_rt//n_points
        mean_rt = spacing*n_points
        grid = (np.arange(n_points)+1)*spacing
        grid = grid[grid < mean_rt - ((n_stages-2)*spacing)]
        comb = np.array([x for x in combinations_with_replacement(grid, n_stages) if np.sum(x) == mean_rt])#A bit bruteforce
        new_comb = []
        for c in comb:
            new_comb.append(np.array(list(mit.distinct_permutations(c))))
        comb = np.vstack(new_comb)

        parameters = np.zeros((len(comb),n_stages,2))
        for idx, y in enumerate(comb):
            parameters[idx, :, :] = [[self.shape, x/self.shape] for x in y]
        print(f'Fitting {len(parameters)} models based on all possibilities from grid search with a spacing of {int(spacing)} samples and {int(n_points)} points')
        return parameters
    
