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
import math
import scipy.stats as stats
import xarray as xr
import multiprocessing as mp

class hsmm:
    
    
    def __init__(self, data, starts, ends, width = 5, gamma_shape = 2):
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
        magnitudes : ndarray
            2D ndarray components * nBumps, initial conditions for bumps magnitudes
        parameters : list
            list of initial conditions for Gamma distribution scale parameter
        width : int
            width of bumps, originally 5 samples
        threshold : float
            threshold for the HsMM algorithm, 0 skips HsMM

        Returns
        -------
        TODO
        '''
        self.starts = starts
        self.ends = ends    
        self.n_trials = len(self.starts)  #number of trials
        self.width = width#width of the bumps in samples
        self.offset = self.width//2#offset on data linked to the choosen width
        self.gamma_shape = gamma_shape
        self.n_samples, self.n_dims = np.shape(data)
        self.bumps = self.calc_bumps(data)#bump morphology added
        self.durations = self.ends - self.starts+1#length of each trial
        self.max_d = np.max(self.durations)

    def fit_single(self, n_bumps, initializing, magnitudes=None, parameters=None, threshold=0):
        lkh,magnitudes,parameters,eventprobs = \
            self.__fit(n_bumps, initializing, magnitudes, parameters, threshold)
        xrlikelihoods = xr.DataArray(self.likelihoods , name="likelihoods")
        xrparams = xr.DataArray(self.parameters, dims=("stage",'params'), name="parameters")
        xrmags = xr.DataArray(self.magnitudes, dims=("component","bump"), name="magnitudes")
        xreventprobs =  xr.DataArray(self.eventprobs, dims=("samples",'trial','bump'), name="eventprobs")
        estimated = xr.merge((xrlikelihoods,xrparams,xrmags,xreventprobs))
        return estimated
        
    def __fit(self, n_bumps, initializing, magnitudes, parameters, threshold):
        '''
        Hidden fitting function underlying sngle and iterative fit
        '''
        if initializing == True:
            parameters = np.tile([2, math.ceil(self.max_d)/(n_bumps+1)/2], (n_bumps+1,1))
            magnitudes = np.zeros((self.n_dims,n_bumps))
        
        lkh1 = -np.inf#initialize likelihood     
        lkh, eventprobs = self.calc_EEG_50h(parameters, magnitudes, n_bumps)
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
                    if parameters[i,:].prod() < self.width:
                        # multiply scale and shape parameters to get 
                        # the mean distance of the gamma-2 pdf. 
                        # It constrains that bumps are separated at 
                        # least a bump length
                        parameters[i,:] = parameters1[i,:]
                lkh, eventprobs = self.calc_EEG_50h(parameters,magnitudes,n_bumps)
        return lkh1,magnitudes1,parameters1,eventprobs1
    
    def fit_iterative(self, max_bumps, initializing, magnitudes=None, parameters=None, threshold=0):
        xrlikelihoods, xrparams, xrmags = [],[],[]
        for n_bumps in np.arange(1, max_bumps+1):
            lkh,magnitudes,parameters,eventprobs = \
                self.__fit(n_bumps,initializing,magnitudes,parameters,threshold)

            if len(parameters) != max_bumps+1:#Xarray needs same dimension size for merging
                parameters = np.concatenate((self.parameters, np.tile(np.nan, \
                    (self.max_bumps+1-len(self.parameters),2))))
                magnitudes = np.concatenate((self.magnitudes, \
                    np.tile(np.nan, (np.shape(self.magnitudes)[0], \
                    self.max_bumps-np.shape(self.magnitudes)[1]))),axis=1)
            xrlikelihoods.append(xr.DataArray(lkh, name="likelihood"))
            xrparams.append(xr.DataArray(parameters, dims=("stage",'params'), name="parameters"))
            xrmags.append(xr.DataArray(magnitudes, dims=("component","bump"), name="magnitudes"))
            
        xrlikelihoods = xr.concat(xrlikelihoods, dim="n_bumps")
        xrparams = xr.concat(xrparams, dim="n_bumps")
        xrmags = xr.concat(xrmags, dim="n_bumps")
        #xreventprobs =  xr.DataArray(self.eventprobs, dims=("bumps","samples",'trial','bump'), name="eventprobs")
        return xr.merge((xrlikelihoods,xrparams,xrmags))
    
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

        template = np.array([0.3090, 0.8090, 1.0000, 0.8090, 0.3090])#bump morph #PCG HARD CODED TO 5 SAMPLES
        template = template/np.sum(template**2)#Weight normalized to sum(P) = 1.294
        
        bumps = np.zeros(data.shape)

        for j in np.arange(self.n_dims):#For each PC
            temp = np.zeros((self.n_samples,5))
            temp[:,0] = data[:,j]#first col = samples of PC
            for i in np.arange(1,self.width):
                temp[:,i] = np.concatenate((temp[1:, i-1], [0]), axis=0)
                # puts the component in a [n_samples X length(bump)] matrix shifted.
                # each column is a copy of the first one but shifted one sample
                # upwards
            bumps[:,j] = temp @ template
            # for each PC we calculate its correlation with bump temp(data samples * 5) *  
            # template(sine wave bump in samples - 5*1)
        bumps[2:,:] = bumps[:-2,:]#Centering
        bumps[[0,1,-2,-1],:] = 0 #Centering
        return bumps

    def calc_EEG_50h(self, parameters, magnitudes, n_bumps):
        '''
        Defines the likelihood function to be maximized as described in Anderson, Zhang, Borst and Walsh, 2016

        Parameters
        ----------
        bumps : ndarray
            2D ndarray with n_samples * ncomponents obtained with the calc_bump function, 
            each sample is adapted to a 5-sample bump morphology
        starts : ndarray
            1D array with start of each trial
        ends : ndarray
            1D array with end of each trial
        magnitudes : ndarray
            2D ndarray components * nBumps, initial conditions for bumps magnitudes
        parameters : list
            list of initial conditions for Gamma distribution scale parameter


        Returns
        -------
        likelihood : float
            likelihoods
        eventprobs : ndarray
            [samples(max_d)*n_trials*n_bumps] = [max_d*trials*nBumps]
        '''
        gains = np.zeros((self.n_samples, n_bumps))

        for i in np.arange(self.n_dims):
            # computes the gains, i.e. how much the bumps reduce the variance at 
            # the location where they are placed, see Appendix Anderson,Zhang, 
            # Borst and Walsh, 2016, last equation, right hand side parenthesis 
            # (S^2 -(S -B)^2) (Sb- B2/2). And sum over all PCA
            gains = gains + self.bumps[:,i][np.newaxis].T * magnitudes[i,:] - \
                    np.tile((magnitudes[i,:]**2),(self.n_samples,1))/2 
            #Previous : gains + bumps[:,i][np.newaxis].T * magnitudes[i,:] -  \
            #        np.tile((magnitudes[i,:]**2),(n_samples,1))/2 
            #MATLAB:gains=gains + bumps(:,i)*magnitudes(i,:) - repmat(magnitudes(i,:).^2,nsamples,1)/2;
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
            probs[self.offset+1:self.ends[i] - self.starts[i]+1 - self.offset,i,:] = \
                gains[self.starts[i]+ self.offset : self.ends[i] - self.offset,:] 
            for j in np.arange(n_bumps): # PCG: for-loop IMPROVE
                probs_b[self.offset+1:self.ends[i]- self.starts[i]+1 - self.offset,i,j] = \
                np.flipud(gains[self.starts[i]+ self.offset : self.ends[i]- self.offset,\
                n_bumps-1-j])
                # assign the reverse of gains per trial

        LP = np.zeros([self.max_d, n_bumps + 1]) # Gamma pdf with each stage parameters
        for j in np.arange(n_bumps):
            LP[:,j] = self.gamma_EEG(parameters[j,0], parameters[j,1], self.max_d)
            # Compute Gamma pdf from 0 to max_d with parameters 'parameters'
        BLP = np.zeros([self.max_d, n_bumps + 1]) 
        BLP[:,:] = LP[:,::-1] # States reversed gamma pdf

        forward = np.zeros((self.max_d, self.n_trials, n_bumps))
        forward_b = np.zeros((self.max_d, self.n_trials, n_bumps))
        backward = np.zeros((self.max_d, self.n_trials, n_bumps))

        # eq1 in Appendix, first definition of likelihood

        forward[self.offset:self.max_d,:,0] = np.tile(LP[:self.max_d-self.offset,0],\
            (self.n_trials,1)).T*probs[self.offset:self.max_d+1,:,0]#Gamma pdf * gains

        forward_b[self.offset:self.max_d,:,0] = np.tile(BLP[:self.max_d-self.offset,0], (self.n_trials,1)).T # reversed Gamma pdf

        for i in np.arange(1,n_bumps):
            next_ = np.concatenate((np.zeros(self.width), LP[:self.max_d - self.width, i]), axis=0)
            # next_ 5 bump width samples followed by gamma pdf
            next_b = np.concatenate((np.zeros(self.width), BLP[:self.max_d - self.width, i]), axis=0)
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
        likelihood = np.sum(np.log(temp[:,:,0].sum(axis=0)))
        # sum(log(sum of 'temp' by columns, samples in a trial)) 
        eventprobs = temp / np.tile(temp.sum(axis=0), [self.max_d, 1, 1])
        #normalization [-1, 1] divide each trial and state by the sum of the n points in a trial
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

        n_bumps = np.size(eventprobs,2) #number of bumps 

        # Expected value, time location
        averagepos = np.hstack((np.sum( \
                np.tile(np.arange(1,self.max_d+1)[np.newaxis].T, (1, n_bumps))\
                * np.mean(eventprobs, axis=1).reshape(self.max_d, n_bumps,\
                order="F").copy(), axis=0), np.mean(self.durations)))
        # 1) mean accross trials of eventprobs -> mP[max_l, nbump]
        # 2) global expected location of each bump
        # concatenate horizontaly to last column the length of each trial
        averagepos = averagepos - (self.offset + \
                        np.hstack([np.arange(0, n_bumps*self.width, self.width), \
                        (n_bumps-1)*self.width+self.offset]))
        # correction for time locations
        flats = averagepos - np.hstack((0,averagepos[:-1]))
        params = np.zeros((n_bumps+1,2))
        params[:,0] = self.gamma_shape 
        params[:,1] = flats.T / self.gamma_shape 
        # correct flats between bumps for the fact that the gamma is 
        # calculated at midpoint
        params[1:-1,1] = params[1:-1,1] + .5 / self.gamma_shape  
        # first flat is bounded on left while last flat may go 
        # beyond on right
        params[0,1] = params[0,1] - .5 / self.gamma_shape 
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
    
    def bump_times(self, n_bumps):
        params = self.estimated.parameters.sel(bumps=n_bumps).values
        scales = [(bump[-1]+self.offset)*(2000/self.sf) for bump in params[:n_bumps+1]]
        return scales
    
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