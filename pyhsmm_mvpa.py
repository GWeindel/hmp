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
import itertools
import warnings
import math

warnings.filterwarnings('ignore', 'Degrees of freedom <= 0 for slice.', )#weird warning, likely due to nan in xarray, not important but better fix it later

def read_mne_EEG(pfiles, event_id, resp_id, sfreq, events=None, resampling=False, \
                 tmin=-.2, tmax=2.2, offset_after_resp = .1, low_pass=.5, \
                 high_pass = 30, upper_limit_RT=2, lower_limit_RT=.2):
    ''' 
    Reads EEG data using MNE's integrated function. This function 
    '''
    import mne
    tstep = 1/sfreq
    epoch_data = [] 
    if isinstance(pfiles,str):#only one participant
        pfiles = [pfiles]
    for participant in pfiles:
        data = mne.io.read_raw_fif(participant, preload=False, verbose=False)
        data.load_data()
        data.filter(low_pass, high_pass, fir_design='firwin', verbose=False)#Filtering out frequency outside range .5 and 30Hz, as study by Anderson et al. (Berberyan used 40 Hz)
        # Loading events (in our case one event = one trial)
        if events is None:
            events = mne.find_events(data, verbose=False)
        else:
            if sfreq < data.info['sfreq']:
                raise ValueError('Cannot provide events and specify downsampling')

        if sfreq < data.info['sfreq']:
            print(f'Downsampling to {sfreq} Hz')
            data, events = data.resample(sfreq, events=events)#100 Hz is the standard used for previous applications of HsMM
        if events[0,1] > 0:#bug from some stim channel, should be 0 otherwise indicated offset in the trggers
            events[:,2] = events[:,2]-events[:,1]#correction on event value
        events_wresp = events

        #Only pick electrodes placed on the scalp:
        picks = mne.pick_types(data.info, eeg=True, stim=False, eog=False, misc=False,
                           exclude='bads') 
        offset_after_resp_samples = int(offset_after_resp*tstep)

        metadata, events, event_id = mne.epochs.make_metadata(
            events=events, event_id= event_id,
            tmin=tmin, tmax=tmax, sfreq=data.info['sfreq'])

        epochs = mne.Epochs(data, events, event_id, tmin, tmax, proj=False,
                        picks=picks, baseline=(None, 0), preload=True,
                        verbose=False,detrend=1,on_missing = 'warn',
                        metadata=metadata,reject_by_annotation=True)
        data_epoch = epochs.get_data()

        valid_epochs_idx = [x for x in np.arange(len(epochs.drop_log)) if epochs.drop_log[x] == ()]

        rts=[]#reaction times
        trigger = []
        i,j = 0,0
        while i < len(events_wresp):
            if events_wresp[i,2] in event_id.values() :
                if j in valid_epochs_idx:
                    if events_wresp[i+1,2] in resp_id.values():# and events_wresp[i-1,2] == 2:#2 for high force condition 
                        rts.append(events_wresp[i+1,0] - events_wresp[i,0] )
                    #if events_wresp[i+1,2] in resp_id.values() and events_wresp[i-1,2] ==1:#1 for low force condition 
                        #rts.append(0)
                    elif events_wresp[i+1,2] not in resp_id.values(): #trials without resp
                        rts.append(0)
                j += 1
            i += 1
        rts = np.array(rts)
        rts[rts > sfreq*upper_limit_RT] = 0 #removes RT above x sec
        rts[rts < sfreq*lower_limit_RT] = 0 #removes RT below x sec, important as determines max bumps

        triggers = epochs.metadata["event_name"].reset_index(drop=True)
        cropped_data_epoch = np.empty([len(rts[rts!= 0]), len(epochs.ch_names), max(rts)+offset_after_resp_samples])
        cropped_data_epoch[:] = np.nan
        cropped_trigger = []
        i, j = 0, 0
        for i in np.arange(len(data_epoch)):
            if rts[i] != 0:
            #Crops the epochs to time 0 (stim onset) up to RT
                cropped_data_epoch[j,:,:rts[i]+offset_after_resp_samples] = \
                (data_epoch[i,:,epochs.time_as_index(0)[0]:\
                epochs.time_as_index(0)[0]+int(rts[i])+offset_after_resp_samples])
                cropped_trigger.append(triggers[i])
                j += 1
        # recover actual data points in a 3D matrix with dimensions trials X electrodes X sample
        epoch_data.append(xr.Dataset(
            {
                "data": (["epochs", "electrodes", "samples"],cropped_data_epoch),
                "event": (["epochs"], cropped_trigger),
            },
            coords={
                "epochs" : np.arange(len(cropped_data_epoch)),
                "electrodes":  epochs.ch_names,
                # TODO When time "electrodes": (['name','x','y','z'], epochs.ch_names,
                "samples": np.arange(max(rts)+offset_after_resp_samples)#+1)
            },
            attrs={'sfreq':epochs.info['sfreq']}
            )
        )

    epoch_data = xr.concat(epoch_data, dim="participant")
    return epoch_data

def standardize(x):
    # Scaling variances to mean variance of the group
    return ((x.data / x.data.std(dim=...)*x.mean_std))

def vcov_mat(x):
    x = x.dropna(dim="samples").squeeze().data
    xT = x.T.data
    return x @ xT

def zscore(data):
    return (data - data.mean()) / data.std()

def transform_data(data, subjects_variable, apply_standard=True,  apply_zscore=True, method='pca', n_comp=10, stack=True, single=False, return_weights=False):
    #Extract durations of epochs (equivalent to RTs) to partition the stacked data

    from sklearn.decomposition import PCA
    if apply_standard:
        mean_std = data.groupby(subjects_variable).std(dim=...).data.mean()
        data = data.assign(mean_std=mean_std.data)
        data = data.groupby(subjects_variable).map(standardize)
    if method == 'pca':
        var_cov_matrices = []
        # Computing cov matrices by trial and take the average of those
        if not single:
            for i,trial_dat in data.stack(trial=("participant", "epochs")).groupby('trial'):
                var_cov_matrices.append(vcov_mat(trial_dat)) #Would be nice not to have a for loop but groupby.map seem to fal
            var_cov_matrix = np.mean(var_cov_matrices,axis=0)
        else:
            for i,trial_dat in data.groupby('epochs'):
                var_cov_matrices.append(vcov_mat(trial_dat)) #Would be nice not to have a for loop but groupby.map seem to fal
            var_cov_matrix = np.mean(var_cov_matrices,axis=0)    
        # Performing spatial PCA on the average var-cov matrix
        pca = PCA(n_components=n_comp, svd_solver='arpack')#selecting Principale components (PC)
        pca_data = pca.fit_transform(var_cov_matrix)

        #Rebuilding pca PCs as xarray to ease computation
        coords = dict(electrodes=("electrodes", data.coords["electrodes"].values),
                     component=("component", np.arange(10)))
        pca_data = xr.DataArray(pca_data, dims=("electrodes","component"), coords=coords)
        data = data @ pca_data
        if apply_zscore and not single:
            data = data.stack(trial=[subjects_variable,'epochs','component']).groupby('trial').map(zscore).unstack()
        elif apply_zscore and single :
            data = data.stack(trial=['epochs','component']).groupby('trial').map(zscore).unstack()
            for comp in data.component:
                if data.sel(component=comp).std() > 0:#corner case in  (some) simulations
                    data.sel(component=comp).groupby('epochs').map(zscore).unstack()
                else:
                    data.sel(component=comp)+ 1e-10
#    else:
#        raise NameError('Method unknown')
    #data = data.reset_index(dims_or_levels="epochs")
    if single:
        durations = np.unique(data.isel(component=0).\
           groupby('epochs').count(dim="samples").data.cumsum())
        while durations[0] == 0:#Dirty due to repeition should be fixed
            durations = durations[1:]
        starts = np.insert(durations[:-1],0,0)
        starts = xr.DataArray(starts, coords={'trial':np.arange(len(durations))})
        ends = durations-1
        ends = xr.DataArray(ends, coords={'trial':np.arange(len(durations))})
        if stack:
            data = data.stack(all_samples=['epochs',"samples"]).dropna(dim="all_samples")
    else:
        durations = np.unique(data.isel(component=0).stack(trial=\
           [subjects_variable,'epochs']).reset_index([subjects_variable,'epochs']).\
           groupby('trial').count(dim="samples").data.cumsum())
        while durations[0] == 0:
            durations = durations[1:]
        starts = np.insert(durations[:-1],0,0)
        starts = xr.DataArray(starts, coords={'trial':np.arange(len(durations))})
        ends = durations-1
        ends = xr.DataArray(ends, coords={'trial':np.arange(len(durations))})
        if stack:
            data = data.stack(all_samples=[subjects_variable,'epochs',"samples"]).dropna(dim="all_samples")
    
    if return_weights:
        return xr.Dataset({'data':data, 'starts':starts, 'ends':ends, 'PCs':pca_data})
    else:
        return data.squeeze(),starts.values,ends.values

def LOOCV(data, subject, n_bumps, iterative_fits, sfreq):
    #Looping over possible number of bumps
    subjects_idx = data.participant.values
    likelihoods_loo = []
    
    #Extracting data without left out subject
    stacked_loo, starts_loo, ends_loo = transform_data(data.sel(participant= subjects_idx[subjects_idx!=subject],drop=False),\
                           'participant', apply_standard=False,  apply_zscore=False, method='', n_comp=10, stack=True)
    #Fitting the HsMM using previous estimated parameters as initial parameters
    model_loo = hsmm(stacked_loo.to_numpy().T, starts_loo, ends_loo, sf=sfreq)
    fit = model_loo.fit_single(n_bumps, iterative_fits.magnitudes, iterative_fits.parameters, 1, False, True)

    #Evaluating likelihood for left out subject
    #Extracting data of left out subject
    stacked_left_out, starts_left_out, ends_left_out = transform_data(data.sel(participant=subject, drop=False),\
                           'participant', apply_standard=False,  apply_zscore=False, method='', n_comp=10, stack=True,single=True)

    model_left_out = hsmm(stacked_left_out.to_numpy().T, starts_left_out, ends_left_out, sf=sfreq)
    likelihood = model_left_out.calc_EEG_50h(fit.magnitudes, fit.parameters, n_bumps,True,True)
    return likelihood, subject


class hsmm:
    
    def __init__(self, data, starts, ends, sf, cpus=1, bump_width = 50):
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
#        if self.cpus > 1:
#            import itertools
#            import multiprocessing
        self.bump_width_samples = int(self.bump_width * (self.sf/1000))
        self.offset = self.bump_width_samples//2#offset on data linked to the choosen width
        # Offset is how soon the first peak can be or how late the last,originaly offset = 2
        self.n_samples, self.n_dims = np.shape(data)
        self.bumps = self.calc_bumps(data)#adds bump morphology
        self.durations = self.ends - self.starts+1#length of each trial
        self.max_d = np.max(self.durations)
        self.max_bumps = self.compute_max_bumps()
    
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
        
        if len(pars) != self.max_bumps+1:#align all dimensions
            pars = np.concatenate((pars, np.tile(np.nan, (self.max_bumps+1-len(pars),2))))
            mags = np.concatenate((mags, np.tile(np.nan, (np.shape(mags)[0], \
                self.max_bumps-np.shape(mags)[1]))),axis=1)
            eventprobs = np.concatenate((eventprobs, np.tile(np.nan, (np.shape(eventprobs)[0],np.shape(eventprobs)[1], self.max_bumps-np.shape(eventprobs)[2]))),axis=2)
        
        xrlikelihoods = xr.DataArray(lkh , name="likelihoods")
        xrparams = xr.DataArray(pars, dims=("stage",'params'), name="parameters")
        xrmags = xr.DataArray(mags, dims=("component","bump"), name="magnitudes")
        xreventprobs = xr.DataArray(eventprobs, dims=("samples",'trial','bump'), name="eventprobs")
        estimated = xr.merge((xrlikelihoods,xrparams,xrmags,xreventprobs))#,xreventprobs))
        print(f"Parameters estimated for {n_bumps} bumps model")
        return estimated
    
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
                gains[self.starts[i]+ self.offset : self.ends[i] - self.offset+1,:] 
            for j in np.arange(n_bumps): # PCG: for-loop IMPROVE
                probs_b[self.offset:self.ends[i]- self.starts[i]+1 - self.offset,i,j] = \
                np.flipud(gains[self.starts[i]+ self.offset : self.ends[i]- self.offset+1,\
                n_bumps-j-1])
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
        forward[self.offset:self.max_d,:,0] = np.tile(LP[:self.max_d-self.offset,0][np.newaxis].T,\
            (1,self.n_trials))*probs[self.offset:self.max_d,:,0]

        forward_b[self.offset:self.max_d,:,0] = np.tile(BLP[:self.max_d-self.offset,0][np.newaxis].T,\
                    (1,self.n_trials)) # reversed Gamma pdf

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
        Gives the average positions of each bump 
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
        width = self.bump_width_samples-1 #unaccounted samples -1?
        # Expected value, time location
        averagepos = np.hstack((np.sum(np.tile(np.arange(self.max_d)[np.newaxis].T,\
            (1, n_bumps)) * np.mean(eventprobs, axis=1).reshape(self.max_d, n_bumps,\
                order="F"), axis=0), np.mean(self.durations)))
        # 1) mean accross trials of eventprobs -> mP[max_l, nbump]
        # 2) global expected location of each bump
        # concatenate horizontaly to last column the length of each trial
        averagepos = averagepos# - np.hstack(np.asarray([self.offset+np.append(np.arange(0,(n_bumps-1)*width+1, width),(n_bumps-1)*width+self.offset)],dtype='object'))
        # PCG hat part is sensible and should be carefully checked
        # correction for time locations with number of bumps and size in samples
        flats = averagepos - np.hstack((0,averagepos[:-1]))
        params = np.zeros((n_bumps+1,2))
        params[:,0] = 2 #PCG shape is hardcoded
        params[:,1] = flats.T / 2 
        # correct flats between bumps for the fact that the gamma is 
        # calculated at midpoint
        #params[1:-1,1] = params[1:-1,1] + .5 / 2  
        # first flat is bounded on left while last flat may go 
        # beyond on right
        #params[0,1] = params[0,1] - .5 / 2 
        return params, averagepos

    def backward_estimation(self):
        '''
        first estimate max_bump solution then estimate max_bump - 1 solution by 
        iteratively removing one of the bump and pick the one with the highest 
        likelihood
        '''
        bump_loo_results = [self.fit_single(self.max_bumps)]

        i = 1
        for n_bumps in np.arange(self.max_bumps-1,0,-1):
            temp_best = bump_loo_results[-i]
            n_bumps_list = np.arange(n_bumps+1)
            possible_bumps = np.array(list(itertools.combinations(n_bumps_list, n_bumps)))
            possible_flats = [x + 1 for x in possible_bumps]
            possible_flats = [np.insert(x,0,0) for x in possible_flats]
            if self.cpus > 1:
                with mp.Pool(processes=self.cpus) as pool:
                    bump_loo_likelihood_temp = pool.starmap(self.fit_single, 
                        zip(itertools.repeat(n_bumps),temp_best.magnitudes.values.T[possible_bumps,:],
                            temp_best.parameters.values[possible_flats,:],
                            itertools.repeat(1),itertools.repeat(True)))
            else:
                raise ValueError('For loop not yet written use cpus >1')
            models = xr.concat(bump_loo_likelihood_temp, dim="iteration")
            bump_loo_results.append(models.sel(iteration=[np.where(models.likelihoods == models.likelihoods.max())[0][0]]))
            i+=1
        bests = xr.concat(bump_loo_results, dim="n_bumps")
        bests = bests.assign_coords({"n_bumps": np.arange(self.max_bumps,0,-1)})
        bests = bests.squeeze('iteration')
        return bests

    
    def mean_bump_times(self,fit, time=True):
        samples = np.where(fit.eventprobs.mean(dim=['trial']).dropna(dim='bump') == 
                np.max(fit.eventprobs.mean(dim=['trial']).dropna(dim='bump'),axis=0))[0]
        if time:
            times = samples*self.tseps
        else: times = samples
        return times
    
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
        d = [stats.gamma.pdf(t+.5,a,scale=b) for t in np.arange(max_length)]
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
    
    
    def bump_times(self, fit, time=True):
        params = fit.parameters.copy(deep=True).dropna(dim="stage")
        if time:
            scales = [(bump[-1])*2*self.tseps for bump in params]
        else:
            scales = [(bump[-1])*2 for bump in params]
        return scales
    
    def compute_max_bumps(self):
        max_bumps = np.min(self.ends - self.starts + 1)//self.bump_width_samples
        return max_bumps

    def save_fit(self, filename):
        estimated = self.extract_results()
        estimated.to_netcdf(filename+".nc")

    def load_fit(self, filename):
        return xr.open_dataset(filename+'.nc')
