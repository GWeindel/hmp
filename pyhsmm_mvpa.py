'''
Copyright (C) 2018, Qiong Zhang, Matthew M Walsh, John R Anderson
Modification and comments by H.Berberyan, ALICE, RUG
Python adaptation and additional comments by Gabriel Weindel

TODO :
- Object oriented programming (add classes and classmethod to avoid redundnant parameters
- Implement multiprocessing
- soft-code the size of bumps/sampling rate
- 
'''

import numpy as np
import scipy.stats as stats
import xarray as xr
import multiprocessing as mp
import warnings
import math

class hsmm:
    
    def __init__(self, data, starts, ends, sf, bump_width = 50):
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
        self.n_trials = len(self.starts)  #number of trials
        self.bump_width = bump_width
        self.bump_width_samples = int(self.bump_width * (self.sf/1000))
        self.offset = self.bump_width_samples//2#offset on data linked to the choosen width
        # Offset is how soon the first peak can be or how late the last,originaly offset = 2
        self.n_samples, self.n_dims = np.shape(data)
        self.bumps = self.calc_bumps(data)#adds bump morphology
        self.durations = self.ends - self.starts+1#length of each trial
        self.max_d = np.max(self.durations)
    
    def calc_bumps(self,data):
        '''
        This function puts on each sample the correlation of that sample and the previous
        five samples with a Bump morphology on time domain.

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
        bump_idx = np.arange(0,self.bump_width_samples)*(1000/self.sf)+(1000/self.sf)/2
        bump_frequency = 1000/(self.bump_width*2)#gives bump frequency given that bumps are defined as hal-sines
        template = np.sin(2*np.pi*np.linspace(0,1,1000)*bump_frequency)[[int(x) for x in bump_idx]]#bump morph based on a half sine with given bump width and sampling frequency #previously np.array([0.3090, 0.8090, 1.0000, 0.8090, 0.3090]) 
        
        ### TESTING : plt.plot(np.linspace(0,1,1000),np.sin((2*np.pi*np.linspace(0,1,1000)*bump_frequency)))
#plt.plot(bump_idx/1000, np.sin(2*np.pi*np.linspace(0,1,1000)*bump_frequency)[[int(x) for x in bump_idx]],'.')
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

    def fit_single(self, n_bumps, magnitudes=None, parameters=None, threshold=1, mp=False,xarr=False):
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
        print(f"Estimating parameters for {n_bumps} bumps model")
        if mp==True: #PCG: Dirty temporarilly needed for multiprocessing in the iterative backroll estimation...
            magnitudes = magnitudes.T
        if xarr==True:
            magnitudes = magnitudes.dropna(dim='bump').values
            parameters = parameters.dropna(dim='stage').values
        lkh,mags,pars,eventprobs = \
            self.__fit(n_bumps, magnitudes, parameters, threshold)
        
        max_bumps = self.max_bumps()
        if len(pars) != max_bumps+1:#align all dimensions
            pars = np.concatenate((pars, np.tile(np.nan, (max_bumps+1-len(pars),2))))
            mags = np.concatenate((mags, np.tile(np.nan, (np.shape(mags)[0], \
                max_bumps-np.shape(mags)[1]))),axis=1)
           # eventprobs = np.concatenate((eventprobs, np.tile(np.nan, (np.shape(eventprobs)[0],np.shape(eventprobs)[1], \
            #    max_bumps-np.shape(eventprobs)[2]))),axis=1)
        
        xrlikelihoods = xr.DataArray(lkh , name="likelihoods")
        xrparams = xr.DataArray(pars, dims=("stage",'params'), name="parameters")
        xrmags = xr.DataArray(mags, dims=("component","bump"), name="magnitudes")
        #xreventprobs = xr.DataArray(eventprobs, dims=("samples",'trial','bump'), name="eventprobs")
        estimated = xr.merge((xrlikelihoods,xrparams,xrmags))#,xreventprobs))
        print(f"Parameters estimated for {n_bumps} bumps model")
        return estimated
    
    def fit_iterative(self, max_bumps, magnitudes=None, parameters=None, threshold=1):
        xrlikelihoods, xrparams, xrmags = [],[],[]
        for n_bumps in np.arange(1, max_bumps+1):
            if np.any(magnitudes)!= None:
                magnitudes_n = magnitudes[n_bumps-1,:,:n_bumps]
                parameters_n = parameters[n_bumps-1,:n_bumps+1,:]
            else: 
                magnitudes_n = magnitudes
                parameters_n = parameters
            lkh,mags,pars,_ = \
                self.__fit(n_bumps, magnitudes_n,parameters_n,threshold)

            if len(pars) != max_bumps+1:#Xarray needs same dimension size for merging
                pars = np.concatenate((pars, np.tile(np.nan, \
                    (max_bumps+1-len(pars),2))))
                mags = np.concatenate((mags, \
                    np.tile(np.nan, (np.shape(mags)[0], \
                    max_bumps-np.shape(mags)[1]))),axis=1)
            xrlikelihoods.append(xr.DataArray(lkh, name="likelihood"))
            xrparams.append(xr.DataArray(pars, dims=("stage",'params'), name="parameters"))
            xrmags.append(xr.DataArray(mags, dims=("component","bump"), name="magnitudes"))
            
        xrlikelihoods = xr.concat(xrlikelihoods, dim="n_bumps")
        xrparams = xr.concat(xrparams, dim="n_bumps")
        xrmags = xr.concat(xrmags, dim="n_bumps")
        #xreventprobs =  xr.DataArray(self.eventprobs, dims=("bumps","samples",'trial','bump'), name="eventprobs")
        return xr.merge((xrlikelihoods,xrparams,xrmags))
    
    #def fit_loo(self, max_bumps, magnitudes=None, parameters=None, threshold=1):

    def get_init_parameters(self, n_bumps):
        parameters = np.tile([2, math.ceil(self.max_d)/(n_bumps+1)/2], (n_bumps+1,1))
        return parameters
        
    def __fit(self, n_bumps, magnitudes, parameters,  threshold):
        '''
        Hidden fitting function underlying single and iterative fit
        '''
        
        try:
            if np.any(parameters)== None:
                warnings.warn('Using default parameters value for gamma parameters')
                parameters = self.get_init_parameters(n_bumps)
        except:
            print("Using magnitudes provided")
        try:
            if np.any(magnitudes)== None:
                warnings.warn('Using default parameters value for magnitudes')
                magnitudes = np.zeros((self.n_dims,n_bumps))
        except:
            print("Using parameters provided")
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
                parameters, averagepos = self.gamma_parameters(eventprobs, n_bumps)

                for i in np.arange(n_bumps + 1): #PCG: seems unefficient
                    if parameters[i,:].prod() < self.bump_width_samples:
                        # multiply scale and shape parameters to get 
                        # the mean distance of the gamma-2 pdf. 
                        # It constrains that bumps are separated at 
                        # least a bump length
                        parameters[i,:] = parameters1[i,:]
                lkh, eventprobs = self.calc_EEG_50h(magnitudes,parameters,n_bumps)
        return lkh1,magnitudes1,parameters1,eventprobs1


    def calc_EEG_50h(self, magnitudes, parameters, n_bumps, lkh_only=False, xarr=False):
        '''
        Defines the likelihood function to be maximized as described in Anderson, Zhang, Borst and Walsh, 2016

        Returns
        -------
        likelihood : float
            likelihoods
        eventprobs : ndarray
            [samples(max_d)*n_trials*n_bumps] = [max_d*trials*nBumps]
        '''
        if xarr == True:
            magnitudes = magnitudes.dropna(dim='bump').values
            parameters = parameters.dropna(dim='stage').values
        gains = np.zeros((self.n_samples, n_bumps))
        #if len(np.shape(magnitudes)) <2:
        #    magnitudes = magnitudes[np.newaxis].T
        for i in np.arange(self.n_dims):
            # computes the gains, i.e. how much the bumps reduce the variance at 
            # the location where they are placed for all samples, see Appendix Anderson,Zhang, 
            # Borst and Walsh, 2016, last equation, right hand side parenthesis 
            # (S^2 -(S -B)^2) (Sb- B2/2). And sum over all PCA
            gains = gains + self.bumps[:,i][np.newaxis].T * magnitudes[i,:] - \
                    np.tile((magnitudes[i,:]**2),(self.n_samples,1))/2 
            #Previous : gains + bumps[:,i][np.newaxis].T * magnitudes[i,:] -  \
            #        np.tile((magnitudes[i,:]**2),(n_samples,1))/2 
            # bump*magnitudes-> gives [n_samples*nBumps] It scales bumps prob. by the
            # global magnitudes of the bumps topology 'magnitudes' of each bump. 
            # tile append vertically the (estimated bump-magnitudes)^2 of one PC 
            # for all samples divided by 2.
            # n -> Total N of samples
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
                gains[self.starts[i]+ self.offset-1 : self.ends[i] - self.offset,:] 
            for j in np.arange(n_bumps): # PCG: for-loop IMPROVE
                probs_b[self.offset:self.ends[i]- self.starts[i]+1 - self.offset,i,j] = \
                np.flipud(gains[self.starts[i]+ self.offset-1 : self.ends[i]- self.offset,\
                n_bumps-1-j])
                # assign the reverse of gains per trial

        LP = np.zeros([self.max_d, n_bumps + 1]) # Gamma pdf for each stage parameters
        for j in np.arange(n_bumps + 1):
            LP[:,j] = self.gamma_EEG(parameters[j,0], parameters[j,1], self.max_d)
            # Compute Gamma pdf from 0 to max_d with parameters 'parameters'
        BLP = np.zeros([self.max_d, n_bumps + 1]) 
        BLP[:,:] = LP[:,::-1] # States reversed gamma pdf
        forward = np.zeros((self.max_d, self.n_trials, n_bumps))
        forward_b = np.zeros((self.max_d, self.n_trials, n_bumps))
        backward = np.zeros((self.max_d, self.n_trials, n_bumps))
        # eq1 in Appendix, first definition of likelihood
        # For each trial (given a length of max duration) compute gamma pdf * gains
        # Start with first bump
        forward[self.offset:self.max_d,:,0] = np.tile(LP[:self.max_d-self.offset,0],\
            (self.n_trials,1)).T*probs[self.offset:self.max_d+1,:,0]

        forward_b[self.offset:self.max_d,:,0] = np.tile(BLP[:self.max_d-self.offset,0],\
                    (self.n_trials,1)).T # reversed Gamma pdf

        for i in np.arange(1,n_bumps):#continue with other bumps
            next_ = np.concatenate((np.zeros(self.bump_width_samples), LP[:self.max_d - \
                    self.bump_width_samples, i]), axis=0)
            # next_ bump width samples followed by gamma pdf
            next_b = np.concatenate((np.zeros(self.bump_width_samples), BLP[:self.max_d - \
                    self.bump_width_samples, i]), axis=0)
            # next_b same with reversed gamma
            add_b = forward_b[:,:,i-1] * probs_b[:,:,i-1]
            for j in np.arange(self.n_trials):
                temp = np.convolve(forward[:,j,i-1],next_)
                # convolution between gamma * gains at state i and 
                # gamma at state i-1
                forward[:,j,i] = temp[:self.max_d]
                temp = np.convolve(add_b[:,j],next_b)
                # same but backwards
                forward_b[:,j,i] = temp[:self.max_d]
            forward[:,:,i] = forward[:,:,i] * probs[:,:,i]
        forward_b = forward_b[:,:,::-1]
        for j in np.arange(self.n_trials): #PCG: IMPROVE
            for i in np.arange(n_bumps):
                backward[:self.durations[j],j,i] = np.flipud(forward_b[:self.durations[j],j,i])
        backward[:self.offset,:,:] = 0
        temp = forward * backward # [max_d,n_trials,n_bumps] .* [max_d,n_trials,n_bumps];
        likelihood = np.sum(np.log(temp[:,:,0].sum(axis=0)))# why 0 index? PCG shouldn't it also be for all dim??
        # sum(log(sum of 'temp' by columns, samples in a trial)) 
        eventprobs = temp / np.tile(temp.sum(axis=0), [self.max_d, 1, 1])
        #normalization [-1, 1] divide each trial and state by the sum of the n points in a trial
        if lkh_only == False:
            return [likelihood, eventprobs]
        else:
            return likelihood
    

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
        width = self.offset-1 #unaccounted samples
        # Expected value, time location
        averagepos = np.hstack((np.sum(np.tile(np.arange(self.max_d)[np.newaxis].T,\
            (1, n_bumps)) * np.mean(eventprobs, axis=1).reshape(self.max_d, n_bumps,\
                order="F"), axis=0), np.mean(self.durations)))
        #PCG 2???
        # 1) mean accross trials of eventprobs -> mP[max_l, nbump]
        # 2) global expected location of each bump
        # concatenate horizontaly to last column the length of each trial
        averagepos = averagepos - (np.hstack(width/2+np.asarray([np.arange(0, (n_bumps-1)*width+1, width), (n_bumps-1)*width+1],dtype='object')))
        
        #PCG unsure about the width/2, is it the shape parameter or width/2
        # correction for time locations
        flats = averagepos - np.array(np.hstack((0,averagepos[:-1])))
        params = np.zeros((n_bumps+1,2))
        params[:,0] = 2 #PCG shape is hardcoded
        params[:,1] = flats.T / 2 
        # correct flats between bumps for the fact that the gamma is 
        # calculated at midpoint
        params[1:-1,1] = params[1:-1,1] + .5 / 2  #PCG: previous params[1:-1,1] = params[1:-1,1] + .5 / 2 
        # first flat is bounded on left while last flat may go 
        # beyond on right
        params[0,1] = params[0,1] - .5 / 2 
        return params, averagepos

    
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
        d = [stats.gamma.pdf(t-.5,a,scale=b) for t in np.arange(1,max_length+1)]
        d = d/np.sum(d)
        return d
        
        
    @staticmethod
    def load_fit(filename):
        return xr.open_dataset(filename+'.nc')

    @staticmethod
    def save_fit(data, filename):
        data.to_netcdf(filename+".nc")
        
    def extract_results(self):
        xrlikelihoods = xr.DataArray(self.likelihoods , name="likelihoods")
        xrparams = xr.DataArray(self.parameters, dims=("stage",'params'), name="parameters")
        xrmags = xr.DataArray(self.magnitudes, dims=("component","bump"), name="magnitudes")
        xreventprobs =  xr.DataArray(self.eventprobs, dims=("samples",'trial','bump'), name="eventprobs")
        estimated = xr.merge((xrlikelihoods,xrparams,xrmags,xreventprobs))
        return estimated
    
    def bump_times(self, fit, n_bumps):
        params = fit.parameters
        params = params.where(np.isfinite(params))
        scales = [(bump[-1])*(2000/self.sf) for bump in params[:n_bumps+1]]
        return scales
    
    
    def max_bumps(self):
        max_bumps = np.min(self.ends - self.starts + 1)//self.bump_width_samples
        return max_bumps

    def save_fit(self, filename):
        estimated = self.extract_results()
        estimated.to_netcdf(filename+".nc")

    def load_fit(self, filename):
        return xr.open_dataset(filename+'.nc')

class iterative_fit(hsmm):
    
    def __init__(self, data, starts, ends, max_bumps,initializing, \
                 magnitudes = None, parameters = None, \
                 width = 5, threshold = 1, gamma_shape = 2):
        
        #super().__init__(data, starts, ends, n_bumps=max_bumps,
        #         magnitudes = None, parameters = None, \
        #         width = 5, threshold = 1, gamma_shape = 2)
        #max_bumps = math.floor(np.min(ends - starts + 1)/5)
        self.max_bumps = max_bumps
        estimated = []
        for n_bumps in np.arange(self.max_bumps)+1:
            print(f'Fitting {n_bumps} bump model')
            super().__init__(data, starts, ends, n_bumps=n_bumps,initializing=initializing, \
                 magnitudes = None, parameters = None, \
                 width = 5, threshold = 1, gamma_shape = 2)

            hsmm.fit(self)
            
            estimated.append(self.extract_results_iterative())
        self.estimated = estimated#xr.concat(estimated, dim="n_bumps", join='left')

    def extract_results_iterative(self):
        if len(self.parameters) != self.max_bumps+1:
            self.parameters = np.concatenate((self.parameters, np.tile(np.nan, \
                (self.max_bumps+1-len(self.parameters),2))))
            self.magnitudes = np.concatenate((self.magnitudes, \
                np.tile(np.nan, (np.shape(self.magnitudes)[0], \
                self.max_bumps-np.shape(self.magnitudes)[1]))),axis=1)
        #    self.eventprobs
        xrlikelihoods = xr.DataArray(self.likelihoods, name="likelihoods")
        xrparams = xr.DataArray(self.parameters, dims=("stage",'params'), name="parameters")
        xrmags = xr.DataArray(self.magnitudes, dims=("component","bump"), name="magnitudes")
        #xreventprobs =  xr.DataArray(self.eventprobs, dims=("bumps","samples",'trial','bump'), name="eventprobs")
        estimated = xr.merge((xrlikelihoods,xrparams,xrmags))#,xreventprobs))
        return estimated
                                      
    def get_results(self):
        return xr.concat(estimated, dim="bumps")
      
    def bump_times(self, n_bumps):
        params = self.estimated.parameters.sel(bumps=n_bumps).values
        scales = [(bump[-1]+self.offset)*(2000/self.sf) for bump in params[:n_bumps+1]]
        return scales
    
class generate_pcs():
    pass

class results():
    
    def __init__(self, estimated, data, starts, ends, width=5,sf=100):
        self.estimated = estimated
        self.data = data
        self.starts = starts
        self.ends = ends
        self.sf = sf
        self.width = width
        self.offset = width//2
        self.n_bumps = np.shape(estimated.magnitudes) 
        self.durations = starts - ends + 1

    
class LOOCV(hsmm):
    
    def __init__(self, data, starts, ends, n_bumps, subjects,\
                 initializing = True, magnitudes = None, parameters = None, \
                 width = 5, threshold = 1, gamma_shape = 2, subject=1):
        subjects_idx = np.unique(subjects)
        subjects_idx_loo = subjects_idx[subjects_idx != subject]
        subjects_loo = np.array([s for s in subjects if s not in subjects_idx_loo])
        starts_left_out_idx = np.array([starts[idx] for idx, s in enumerate(subjects) if s not in subjects_idx_loo])
        ends_left_out_idx = np.array([ends[idx] for idx, s in enumerate(subjects) if s not in subjects_idx_loo])

        #Correct starts indexes to account for reoved subject, whole indexing needs improvement
        starts_loo = np.concatenate([starts[starts < starts_left_out_idx[0]], starts[starts > ends_left_out_idx[-1]]-ends_left_out_idx[-1]+1])
        ends_loo = np.concatenate([ends[ends < starts_left_out_idx[0]], ends[ends > ends_left_out_idx[-1]]-ends_left_out_idx[-1]])
        starts_left_out = np.array([start - starts_left_out_idx[0]  for start in starts if start >= starts_left_out_idx[0] and start <= ends_left_out_idx[-1]])
        ends_left_out = np.array([end - starts_left_out_idx[0]  for end in ends if end >= starts_left_out_idx[0] and end <= ends_left_out_idx[-1]])


        samples_loo = np.array([sample for idx,sample in enumerate(data) if idx < starts_left_out_idx[0] or idx > ends_left_out_idx[-1]])
        samples_left_out = np.array([sample for idx,sample in enumerate(data) if idx >= starts_left_out_idx[0] and idx <= ends_left_out_idx[-1]])
        
        #Fitting the HsMM using previous estimated parameters as initial parameters
        super().__init__(samples_loo, starts_loo, ends_loo, n_bumps= n_bumps,initializing=initializing, \
                 magnitudes = magnitudes[:,:n_bumps],
                 parameters = parameters[:n_bumps+1,:])
        hsmm.fit(self)
        
        super().__init__(samples_left_out, starts_left_out, ends_left_out, n_bumps, initializing=False,\
            magnitudes = self.magnitudes[:,:n_bumps].values,
            parameters = self.parameters[:n_bumps+1,:].values,
            threshold=0)
        hsmm.fit(self)        

    def extract_results(self):
        xrlikelihoods = xr.DataArray(self.likelihoods , name="likelihoods")
        xrparams = xr.DataArray(self.parameters, dims=("stage",'params'), name="parameters")
        xrmags = xr.DataArray(self.magnitudes, dims=("component","bump"), name="magnitudes")
        xreventprobs =  xr.DataArray(self.eventprobs, dims=("samples",'trial','bump'), name="eventprobs")
        estimated = xr.merge((xrlikelihoods,xrparams,xrmags,xreventprobs))
        return estimated