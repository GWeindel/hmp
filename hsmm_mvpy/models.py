'''

'''

import numpy as np
import xarray as xr
import multiprocessing as mp
import itertools
import math


class hsmm:
    
    def __init__(self, data, starts, ends, sf, cpus=1, bump_width = 50, shape=2, estimate_magnitudes=True, estimate_parameters=True,
                parameters_to_fix = [], magnitudes_to_fix = []):
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
        self.starts = starts
        self.ends = ends    
        self.sf = sf
        self.tseps = 1000/sf
        self.n_trials = len(self.starts)  #number of trials
        self.bump_width = bump_width
        self.cpus = cpus
        self.bump_width_samples = int(self.bump_width * (self.sf/1000))
        self.offset = self.bump_width_samples//2#offset on data linked to the choosen width how soon the first peak can be or how late the last,
        self.n_samples, self.n_dims = np.shape(data)
        self.bumps = self.calc_bumps(data)#adds bump morphology
        self.durations = self.ends - self.starts+1#length of each trial
        self.max_d = np.max(self.durations)
        self.max_bumps = self.compute_max_bumps()
        self.shape = shape
        self.estimate_magnitudes = estimate_magnitudes 
        self.estimate_parameters = estimate_parameters
        self.parameters_to_fix = parameters_to_fix
        self.magnitudes_to_fix = magnitudes_to_fix
        
    
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

    def fit_single(self, n_bumps, magnitudes=None, parameters=None, threshold=1, mp=False, verbose=True, starting_points=1):
        '''
        Fit HsMM for a single n_bumps model

        Parameters
        ----------
        n_bumps : int
            how many bumps are estimated
        magnitudes : ndarray
            2D ndarray components * n_bumps, initial conditions for bumps magnitudes
        parameters : list
            list of initial conditions for Gamma distribution scale parameter
        threshold : float
            threshold for the HsMM algorithm, 0 skips HsMM

        '''
        if verbose:
            print(f"Estimating parameters for {n_bumps} bumps model")
        if mp==True: #PCG: Dirty temporarilly needed for multiprocessing in the iterative backroll estimation...
            magnitudes = magnitudes.T
        if isinstance(parameters, (xr.DataArray,xr.Dataset)):
            parameters = parameters.dropna(dim='stage').values
        if isinstance(magnitudes, (xr.DataArray,xr.Dataset)):
            magnitudes = magnitudes.dropna(dim='bump').values
        
        if self.cpus > 1 and starting_points > 1:
            import multiprocessing as mp
            parameters = []
            magnitudes = []
            for sp in np.arange(starting_points):
                parameters.append(np.array([[self.shape,x] for x in np.random.uniform(0,(np.mean(self.durations)/self.shape),n_bumps+1)]))
                magnitudes.append(np.random.normal(0, .5, (self.n_dims,n_bumps)))
            with mp.Pool(processes=self.cpus) as pool:
                estimates = pool.starmap(self.fit, 
                    zip(itertools.repeat(n_bumps), magnitudes, parameters, itertools.repeat(1)))
            lkhs_sp = [x[0] for x in estimates]
            mags_sp = [x[1] for x in estimates]
            pars_sp = [x[2] for x in estimates]
            eventprobs_sp = [x[3] for x in estimates]
            max_lkhs = np.where(lkhs_sp == np.max(lkhs_sp))[0][0]
            lkh = lkhs_sp[max_lkhs]
            mags = mags_sp[max_lkhs]
            pars = pars_sp[max_lkhs]
            eventprobs = eventprobs_sp[max_lkhs]
        elif starting_points > 1:
            likelihood_prev = -np.inf
            for sp in np.arange(starting_points):
                if sp  > 1:
                    #For now the random starting point are uninformed, might be worth to switch to a cleverer solution
                    parameters = np.array([[self.shape,x] for x in np.random.uniform(0,(np.mean(self.durations)/self.shape),n_bumps+1)])
                    magnitudes = np.random.normal(0, .5, (self.n_dims,n_bumps))
                likelihood, magnitudes_, parameters_, eventprobs_ = \
                    self.fit(n_bumps, magnitudes, parameters, threshold)
                if likelihood > likelihood_prev:
                    lkh, mags, pars, eventprobs = likelihood, magnitudes_, parameters_, eventprobs_
                    likelihood_prev = likelihood
        elif np.any(parameters)== None or np.any(magnitudes)== None :
            if np.any(parameters)== None:
                parameters = np.tile([self.shape, math.ceil(np.mean(self.durations)/(n_bumps+1)/self.shape)], (n_bumps+1,1))
            if np.any(magnitudes)== None:
                magnitudes = np.zeros((self.n_dims,n_bumps))
            lkh, mags, pars, eventprobs = self.fit(n_bumps, magnitudes, parameters, threshold)
        else:
            lkh, mags, pars, eventprobs = self.fit(n_bumps, magnitudes, parameters, threshold)
        #Comparing to uninitialized gamma parameters
        #parameters = np.tile([self.shape, 50], (n_bumps+1,1))
        #likelihood, magnitudes_, parameters_, eventprobs_ = \
        #        self.fit(n_bumps, magnitudes, parameters, threshold)
        #if likelihood > lkh:
        #    lkh, mags, pars, eventprobs = likelihood, magnitudes_, parameters_, eventprobs_

        
        if len(pars) != self.max_bumps+1:#align all dimensions
            pars = np.concatenate((pars, np.tile(np.nan, (self.max_bumps+1-len(pars),2))))
            mags = np.concatenate((mags, np.tile(np.nan, (np.shape(mags)[0], \
                self.max_bumps-np.shape(mags)[1]))),axis=1)
            eventprobs = np.concatenate((eventprobs, np.tile(np.nan, (np.shape(eventprobs)[0],np.shape(eventprobs)[1], self.max_bumps-np.shape(eventprobs)[2]))),axis=2)
        
        xrlikelihoods = xr.DataArray(lkh , name="likelihoods")
        xrparams = xr.DataArray(pars, dims=("stage",'params'), name="parameters")
        xrmags = xr.DataArray(mags, dims=("component","bump"), name="magnitudes")
        xreventprobs = xr.DataArray(eventprobs, dims=("samples",'trial','bump'), name="eventprobs")
        estimated = xr.merge((xrlikelihoods, xrparams, xrmags, xreventprobs))#,xreventprobs))
        if verbose:
            print(f"Parameters estimated for {n_bumps} bumps model")
        return estimated
        
    def fit(self, n_bumps, magnitudes, parameters,  threshold):
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
                if self.estimate_magnitudes:
                    for i in np.arange(n_bumps):
                        for j in np.arange(self.n_dims):
                            magnitudes[j,i] = np.mean(np.sum( \
                            eventprobs[:,:,i]*means[:,:,j], axis=0))
                            # 2) sum of all samples in a trial
                            # 3) mean across trials of the sum of samples in a trial
                            # repeated for each PC (j) and later for each bump (i)
                            # magnitudes [nPCAs, nBumps]
                        if i in self.magnitudes_to_fix:
                            magnitudes[:,i] = magnitudes1[:,i]
                if self.estimate_parameters:
                    parameters = self.gamma_parameters(eventprobs, n_bumps)

                    #Ensure constrain of gammas > bump_width, note that contrary to the matlab code this is not applied on the first stage (np.arange(1,n_bumps) 
                    for i in np.arange(1,n_bumps): #PCG: seems unefficient likely slows down process, isn't there a better way to bound the estimation??
                        if parameters[i,:].prod() < self.bump_width_samples:
                            # multiply scale and shape parameters to get 
                            # the mean distance of the gamma-2 pdf. 
                            # It constrains that bumps are separated at 
                            # least a bump length
                            parameters[i,:] = parameters1[i,:]
                        if i in self.parameters_to_fix:
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
            probs[self.offset:self.ends[i] - self.starts[i]+1 - self.offset,i,:] = \
                gains[self.starts[i]+ self.offset : self.ends[i] - self.offset+1,:] 
            for j in np.arange(n_bumps): # PCG: for-loop IMPROVE
                probs_b[self.offset:self.ends[i]- self.starts[i]+1 - self.offset,i,j] = \
                np.flipud(gains[self.starts[i]+ self.offset : self.ends[i]- self.offset+1,\
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
        forward[self.offset:self.max_d,:,0] = np.tile(LP[:self.max_d-self.offset,0][np.newaxis].T,\
            (1,self.n_trials))*probs[self.offset:self.max_d,:,0]

        forward_b[self.offset:self.max_d,:,0] = np.tile(BLP[:self.max_d-self.offset,0][np.newaxis].T,\
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
        #params[:,1] = params[:,1] - .5 /shape
        # first flat is bounded on left while last flat may go 
        # beyond on right
        params[0,1] = params[0,1] + .5 /self.shape
        params[-1,1] = params[-1,1] - .5 /self.shape
        return params

    def backward_estimation(self,max_fit=None, max_starting_points=1):
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
            print(f'Estimating all solutions for maximal number of bumps ({self.max_bumps}) with {max_starting_points-1} random starting points')
            bump_loo_results = [self.fit_single(self.max_bumps, starting_points=max_starting_points)]
        else:
            bump_loo_results = [max_fit]
        i = 0
        for n_bumps in np.arange(self.max_bumps-1,0,-1):
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
                        zip(itertools.repeat(n_bumps), bumps_temp, flats_temp,#itertools.repeat(np.tile([self.shape,50], (n_bumps+1,1))),# ##
                            #temp_best.parameters.values[possible_flats,:],
                            #itertools.repeat(self.get_init_parameters(n_bumps)),
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
    
    def compute_max_bumps(self):
        '''
        Compute the maximum possible number of bumps given bump width and minimum epoch size
        '''
        return int(np.min(self.durations)/self.bump_width_samples)

    def bump_times(self, eventprobs):
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
        
        eventprobs = eventprobs.dropna('bump')
        onsets = np.empty((len(eventprobs.trial),len(eventprobs.bump)+1))
        i = 0
        for trial in eventprobs.trial.values:
            onsets[i, :len(eventprobs.bump)] = np.arange(self.max_d) @ eventprobs.sel(trial=trial).data - self.bump_width_samples/2#Correcting for centerning, thus times represents bump onset
            onsets[i, -1] = self.ends[i] - self.starts[i]
            i += 1
        return onsets
    