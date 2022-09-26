'''

'''

import numpy as np
import scipy.stats as stats
import xarray as xr
import multiprocessing as mp
import itertools
import warnings

warnings.filterwarnings('ignore', 'Degrees of freedom <= 0 for slice.', )#weird warning, likely due to nan in xarray, not important but better fix it later  

def read_mne_EEG(pfiles, event_id, resp_id, sfreq, subj_idx=None, events_provided=None, verbose=True,
                 tmin=-.2, tmax=5, offset_after_resp = .1, high_pass=.5, \
                 low_pass = 30, upper_limit_RT=5, lower_limit_RT=0.001, reject_threshold=None):
    ''' 
    Reads EEG data format (.fif or .bdf) using MNE's integrated function .
    
    Notes: 
    - Only EEG data are selected (other channel types are discarded)
    - All times are expressed on the second scale.
    - If multiple files in pfiles the data of the group is read and seqentially processed.
    - Reaction Times are only computed if response trigger is in the epoch window (determined by tmin and tmax)
    
    Procedure:
    1) the data is filtered with filters specified in low_pass and high_pass. Parameters of the filter are
        determined by MNE's filter function.
    2) if no events is provided, detect events in stumulus channel and keep events with id in event_id and resp_id.
    3) eventual downsampling is performed if sfreq is lower than the data's sampling frequency. The event structure is
        passed at the resample() function of MNE to ensure that events are approriately timed after downsampling.
    4) epochs are created based on stimulus onsets (event_id) and tmin and tmax. Epoching removes any epoch where a 
        'BAD' annotiation is present and all epochs where an electrode exceeds reject_threshold. Epochs are baseline 
        corrected from tmin to stimulus onset (time 0).
    5) Reaction times (RT) are computed based on the sample difference between onset of stimulus and response triggers. 
        If no response event happens after a stimulus or if RT > upper_limit_RT & < upper_limit_RT, RT is 0.
    6) all the non-rejected epochs with positive RTs are cropped to stimulus onset to stimulus_onset + RT.
    
    Parameters
    ----------
    pfiles : str or list
        list of EEG files to read
    event_id : dict
        Dictionary containing the correspondance of named condition [keys] and event code [values]
    resp_id : ndarray
        Dictionary containing the correspondance of named response [keys] and event code [values]
    sfreq : float
        Desired sampling frequency
    subj_idx : list
        List of subject names
    events_provided : float
        np.array with 3 columns -> [samples of the event, initial value of the channel, event code]. To use if the
        automated event detection method of MNE is not appropriate 
    verbose : bool
        Whether to display MNE's message
    tmin : float
        Time taken before stimulus onset to compute baseline
    tmax : float
        Time taken after stimulus onset
    offset_after_resp : float
        Time taken after onset of the response
    low_pass : float
        Value of the low pass filter
    high_pass : float
        Value of the high pass filter
    upper_limit_RT : float
        Upper limit for RTs. Longer RTs are discarded
    lower_limit_RT : float
        Lower limit for RTs. Shorter RTs are discarded
    reject_threshold : float
        Rejection threshold to apply when creating epochs, expressed in microvolt
    
        
    Returns
    -------
    epoch_data : xarray
        Returns an xarray Dataset with all the data, events, electrodes, participant. 
        All eventual participant/electrodes naming and epochs index are kept. 
        The choosen sampling frequnecy is stored as attribute.
    '''
    import mne
    tstep = 1/sfreq
    epoch_data = [] 
    if isinstance(pfiles,str):#only one participant
        pfiles = [pfiles]
    if not subj_idx:
        subj_idx = ["S"+str(x) for x in np.arange(len(pfiles))]
    y = 0
    for participant in pfiles:
        print(f'Processing participant {participant}')
        if '.fif' in participant:
            data = mne.io.read_raw_fif(participant, preload=False, verbose=verbose)
        elif '.bdf' in participant:
            data = mne.io.read_raw_bdf(participant, preload=False, verbose=verbose)
        else:
            raise ValueError(f'Unknown EEG file format for participant {participant}')
        data.load_data()
        data.filter(high_pass, low_pass, fir_design='firwin', verbose=verbose)#Filtering out frequency outside range .5 and 30Hz, as study by Anderson et al.
        # Loading events (in our case one event = one trial)
        if events_provided is None:
            events = mne.find_events(data, verbose=verbose, min_duration = 1 / data.info['sfreq'])
            if events[0,1] > 0:#bug from some stim channel, should be 0 otherwise indicates offset in the trggers
                events[:,2] = events[:,2]-events[:,1]#correction on event value
            events_values = np.concatenate([np.array([x for x in event_id.values()]), np.array([x for x in resp_id.values()])])
            events = np.array([list(x) for x in events if x[2] in events_values])#only keeps events with stim or response
        else:
            if len(np.shape(events_provided)) == 2:
                events_provided = events_provided[np.newaxis]
            events = events_provided[y]
        if sfreq < data.info['sfreq']:#Downsampling
            print(f'Downsampling to {sfreq} Hz')
            data, events = data.resample(sfreq, events=events)#100 Hz is the standard used for previous applications of HsMM


        #Only pick eeg electrodes
        picks = mne.pick_types(data.info, eeg=True, stim=False, eog=False, misc=False,
                           exclude='bads') 
        offset_after_resp_samples = int(offset_after_resp*tstep)
        metadata, meta_events, event_id = mne.epochs.make_metadata(
            events=events, event_id= event_id,
            tmin=tmin, tmax=tmax, sfreq=data.info['sfreq'])
        epochs = mne.Epochs(data, meta_events, event_id, tmin, tmax, proj=False,
                        picks=picks, baseline=(None, 0), preload=True,
                        verbose=verbose,detrend=1, on_missing = 'warn', event_repeated='drop',
                        metadata=metadata, reject_by_annotation=True, reject=reject_threshold)
        data_epoch = epochs.get_data()

        valid_epochs_idx = [x for x in np.arange(len(epochs.drop_log)) if epochs.drop_log[x] == ()]
        correct_stim_timing  = np.array([list(x) for x in events if x[2] in event_id.values()])[valid_epochs_idx,0]
        stim_events = np.array([x for x in np.arange(len(events)) if events[x,0] in correct_stim_timing])
        rts=[]#reaction times
        trigger = []
        without_rt = 0
        for i in stim_events:
            if events[i+1,2] in resp_id.values():
                rts.append(events[i+1,0] - events[i,0] )

            else:
                rts.append(0)
                without_rt += 1
        print(f'N trials without response event: {without_rt}')
        rts = np.array(rts)
        print(f'Applying reaction time trim to keep RTs between {lower_limit_RT} and {upper_limit_RT} seconds')
        rts[rts > sfreq*upper_limit_RT] = 0 #removes RT above x sec
        rts[rts < sfreq*lower_limit_RT] = 0 #removes RT below x sec, important as determines max bumps
        print(f'{len(rts)} RTs kept of {len(stim_events)} clean epochs')
        triggers = epochs.metadata["event_name"].reset_index(drop=True)
        cropped_data_epoch = np.empty([len(rts[rts> 0]), len(epochs.ch_names), max(rts)+offset_after_resp_samples])
        cropped_data_epoch[:] = np.nan
        cropped_trigger = []
        epochs_idx = []
        j = 0
        for i in np.arange(len(data_epoch)):
            if rts[i] > 0:
                cropped_trigger.append(triggers[i])
            #Crops the epochs to time 0 (stim onset) up to RT
                cropped_data_epoch[j,:,:rts[i]+offset_after_resp_samples] = \
                (data_epoch[i,:,epochs.time_as_index(0)[0]:\
                epochs.time_as_index(0)[0]+int(rts[i])+offset_after_resp_samples])
                epochs_idx.append(valid_epochs_idx[j])
                j += 1
        x = 0
        while np.isnan(cropped_data_epoch[-1]).all():#Weird bug I guess it is perhps due to too long epoch?
            cropped_data_epoch = cropped_data_epoch[:-1]
            x += 1
        if x > 0:
            print(f'RTs > 0 longer than expected ({x})')
        print(f'{len(cropped_data_epoch)} trials were retained for participant {participant}')
        print(f'End sampling frequency is {sfreq} Hz')
        epoch_data.append(hsmm_data_format(cropped_data_epoch, cropped_trigger, epochs.info['sfreq'], epochs=[int(x) for x in epochs_idx], electrodes = epochs.ch_names))
        y += 1
        
    epoch_data = xr.concat(epoch_data, dim = xr.DataArray(subj_idx, dims='participant'),
                          fill_value={'event':'', 'data':np.nan})
    return epoch_data

def hsmm_data_format(data, events, sfreq, participants=[], epochs=None, electrodes=None):
    '''
    Converting 3D matrix with dimensions (participant) * trials * electrodes * sample into xarray Dataset
    
    Parameters
    ----------
    data : ndarray
        4/3D matrix with dimensions (participant) * trials * electrodes * sample  
    events : float
        np.array with 3 columns -> [samples of the event, initial value of the channel, event code]. To use if the
        automated event detection method of MNE is not appropriate 
    sfreq : float
        Sampling frequency of the data
    participants : list
        List of participant index
    epochs : list
        List of epochs index
    electrodes : list
        List of electrode index
    '''
    if len(np.shape(data)) == 4:#means group
        n_subj, n_epochs, n_electrodes, n_samples = np.shape(data)
    elif len(np.shape(data)) == 3:
        n_epochs, n_electrodes, n_samples = np.shape(data)
    else:
        raise ValueError(f'Unknown data format with dimensions {np.shape(data)}')
    if events is None:
        events = np.repeat(np.nan, n_epochs)
    if electrodes is None:
        electrodes = np.arange(n_electrodes)
    if epochs is None:
         epochs = np.arange(n_epochs)
    if len(participants) < 2:
        data = xr.Dataset(
                {
                    "data": (["epochs", "electrodes", "samples"],data),
                    "event": (["epochs"], events),
                },
                coords={
                    "epochs" :epochs,
                    "electrodes":  electrodes,
                    "samples": np.arange(n_samples)
                },
                attrs={'sfreq':sfreq}
                )
    else:
        data = xr.Dataset(
                {
                    "data": (['participant',"epochs", "electrodes", "samples"],data),
                    "event": (['participant',"epochs"], events),
                },
                coords={
                    'participant':participants,
                    "epochs" :epochs,
                    "electrodes":  electrodes,
                    "samples": np.arange(n_samples)
                },
                attrs={'sfreq':sfreq}
                )
    return data

def standardize(x):
    '''
    Scaling variances to mean variance of the group
    '''
    return ((x.data / x.data.std(dim=...))*x.mean_std)

def vcov_mat(x):
    '''
    Computes Variance-Covariance matrix
    '''
    x = x.dropna(dim="samples").squeeze().data
    xT = x.T.data
    return x @ xT

def zscore(data):
    '''
    zscore of the data
    '''
    return (data - data.mean()) / data.std()

def transform_data(data, subjects_variable="participant", apply_standard=True,  apply_zscore=True, method='pca', n_comp=None, single=False, return_weights=False):
    '''
    Adapts EEG epoched data (in xarray format) to the expected data format for hsmms. 
    First this code can apply standardization of individual variances (if apply_standard=True).
    Second, a spatial PCA on the average variance-covariance matrix is performed (if method='pca', more methods in development)
    Third,stacks the data going from format [participant * epochs * samples * electrodes] to [samples * electrodes]
    Last, performs z-scoring on each epoch and for each principal component (PC)
    
    
    Parameters
    ----------
    data : xarray
        unstacked xarray data from transform_data() or anyother source yielding an xarray with dimensions 
        [participant * epochs * samples * electrodes] 
    subjects_variable : str
        name of the dimension for subjects ID
    apply_standard : bool 
        Whether to apply standardization
    apply_zscore : bool 
        Whether to apply z-scoring
    method : str
        Method to apply, for now limited to 'pca'
    n_comp : int
        How many components to select from the PC space, if None plots the scree plot and a prompt requires user
        to specify how many PCs should be retained
    single : bool 
        Whether participant is unique (True) or a group of participant (False)

    Returns
    -------
    data : xarray.Dataset
        xarray dataset [n_samples * n_comp] data expressed in the PC space, ready for hsMM fit
    pca_data : xarray.Dataset
        loadings of the PCA, used to retrieve electrode space
    pca.explained_variance_ : ndarray
        explained variance for each component
    means : xarray.DataArray
        means of the electrodes before PCA/zscore
    '''
    from sklearn.decomposition import PCA
    #var = data.var(...)
    if apply_standard and not single:
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
        if n_comp == None:
            import matplotlib.pyplot as plt
            n_comp = np.shape(var_cov_matrix)[0]-1
            fig, ax = plt.subplots(1,2, figsize=(.2*n_comp, 4))
            pca = PCA(n_components=n_comp, svd_solver='full')#selecting Principale components (PC)
            pca_data = pca.fit_transform(var_cov_matrix)
            var = pca.transform(var_cov_matrix)
            var = np.var(var, axis=0)
            ax[0].plot(np.arange(pca.n_components)+1, var/np.sum(var),'.-')
            ax[0].set_ylabel('Normalized explained variance')
            ax[0].set_xlabel('Component')
            ax[1].plot(np.arange(pca.n_components)+1, np.cumsum(var/np.sum(var)),'.-')
            ax[1].set_ylabel('Cumulative normalized explained variance')
            ax[1].set_xlabel('Component')
            plt.tight_layout()
            plt.show()
            n_comp = int(input(f'How many PCs (80 and 95% explained variance at component n{np.where(np.cumsum(var/np.sum(var)) >= .80)[0][0]+1} and n{np.where(np.cumsum(var/np.sum(var)) >= .95)[0][0]+1})?'))

        pca = PCA(n_components=n_comp, svd_solver='full')#selecting Principale components (PC)

        pca_data = pca.fit_transform(var_cov_matrix)/pca.explained_variance_ # divided by explained var for compatibility with matlab's PCA
        
        #Rebuilding pca PCs as xarray to ease computation
        coords = dict(electrodes=("electrodes", data.coords["electrodes"].values),
                     component=("component", np.arange(n_comp)))
        pca_data = xr.DataArray(pca_data, dims=("electrodes","component"), coords=coords)
        means = data.groupby('electrodes').mean(...)
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
        if return_weights:
            return data, pca_data, pca.explained_variance_, means
        else:
            return data
    else:
        return data
    

def stack_data(data, subjects_variable='participant', electrode_variable='component', single=False):
    '''
    Stacks the data going from format [participant * epochs * samples * electrodes] to [samples * electrodes]
    with sample indexes starts and ends to delimitate the epochs.
    
    
    Parameters
    ----------
    data : xarray
        unstacked xarray data from transform_data() or anyother source yielding an xarray with dimensions 
        [participant * epochs * samples * electrodes] 
    subjects_variable : str
        name of the dimension for subjects ID
    single : bool 
        Whether participant is unique (True) or a group of participant (False)
    
    Returns
    -------
    data : xarray.Dataset
        xarray dataset [samples * electrodes]
    '''    
    if isinstance(data, (xr.DataArray,xr.Dataset)) and 'component' not in data.dims:
        data = data.rename_dims({'electrodes':'component'})
        print(data.participant)
    if "participant" not in data.dims:
        data = data.expand_dims("participant")
    durations = data.isel(component=0).rename({'epochs':'trials', subjects_variable:'subjects'})\
    .stack(trial_x_participant=['subjects','trials']).dropna(dim="trial_x_participant", how="all").\
    groupby('trial_x_participant').count(dim="samples").cumsum()

    #durations = data.isel(component=0).rename({'epochs':'trials', subjects_variable:'subjects'}).stack(trial=\
    #   ['subjects','trials']).dropna(dim="trial", how='all').\
    #   groupby('trial').count(dim="samples").cumsum().unstack()

    data = data.stack(all_samples=[subjects_variable,'epochs',"samples"]).dropna(dim="all_samples")
    return xr.Dataset({'data':data, 'durations':durations})
    #return data, durations



def LOOCV(data, subject, n_bumps, initial_fit, sfreq, bump_width=50):
    '''
    Performs Leave-one-out cross validation, removes one participant from data, estimate n_bumps HsMM parameters, 
    compute the likelihood of the data from the left out participant with the estimated parameters. The model is fit
    using initial fit as starting points for magnitudes and parameters
    
    
    Parameters
    ----------
    data : xarray
        xarray data from transform_data() 
    subject : str
        name of the subject to remove
    n_bumps : int 
        How many bumps in the model
    initial_fit : xarray
        Fit of the model with the same number of bumps and all participants
    sfreq : float
        Sampling frequency of the data
    bump_width : float
        length of the bumps in milliseconds
    
    Returns
    -------
    likelihood : float
        likelihood computed for the left-out participant
    subject : str
        name of the subject to remove
    '''    
    from hsmm_mvpy.models import hsmm
    #Looping over possible number of bumps
    subjects_idx = data.participant.values
    likelihoods_loo = []
    #Extracting data without left out subject
    stacked_loo = stack_data(data.sel(participant= subjects_idx[subjects_idx!=subject],drop=False))
    #Fitting the HsMM using previous estimated parameters as initial parameters
    model_loo = hsmm(stacked_loo, sf=sfreq, bump_width=bump_width)
    fit = model_loo.fit_single(n_bumps, initial_fit.magnitudes, initial_fit.parameters, 1, False, verbose=False)

    #Evaluating likelihood for left out subject
    #Extracting data of left out subject
    stacked_left_out = stack_data(data.sel(participant=subject, drop=False))
    model_left_out = hsmm(stacked_left_out, sf=sfreq, bump_width=bump_width)
    likelihood = model_left_out.calc_EEG_50h(fit.magnitudes, fit.parameters, n_bumps,True)
    return likelihood, subject


def reconstruct(magnitudes, PCs, eigen, means):
    '''
    Reconstruct electrode activity from PCA
    
    Parameters
    ----------
    magnitudes :  
        2D or 3D ndarray with [n_components * n_bumps] can also contain several estimations [estimation * n_components * n_bumps]
    PCs : 
        2D ndarray with PCA loadings [channels x n_components]
    eigen : 
        PCA eigenvalues of the covariance matrix of data [n_components]
    means : 
        Grand mean [channels]
        
    Returns
    -------
    electrodes : ndarray
        a 2D ndarray with [electrodes * bumps]
    '''
    if len(np.shape(magnitudes))==2:
        magnitudes = np.array([magnitudes])
    n_iter, n_comp, n_bumps = np.shape(magnitudes)
    list_electrodes = []
    for iteration in np.arange(n_iter): 
        #electrodes = np.array(magnitudes[iteration].T * 
        electrodes =  np.array(magnitudes[iteration, ].T * np.tile(np.sqrt(eigen[:n_comp]).T, (n_bumps,1))) @ np.array(PCs[:,:n_comp]).T
        list_electrodes.append(electrodes + np.tile(means,(n_bumps,1)))#add means for each electrode
    return np.array(list_electrodes)

def stage_durations(times):
    '''
    Returns the stage durations from the bump onset times by substracting each stage to the previous
    
    Parameters
    ----------
    times : ndarray
        2D ndarray with [n_trials * n_bumps]
    
    Returns
    -------
    times : ndarray
        2D ndarray with [n_trials * n_bumps]
    '''
    times = np.diff(times, axis=1, prepend=0)
    return times

def save_fit(data, filename):
    '''
    Save fit
    '''
    data.unstack().to_netcdf(filename)
    print(f"{filename} saved")

def load_fit(filename):
    '''
    Load fit
    '''
    data = xr.open_dataset(filename)
    if 'trials' in data:
        data = data.stack(trial_x_participant=["participant","trials"]).dropna(dim="trial_x_participant", how='all')
    return data

def save_eventprobs(eventprobs, filename):
    '''
    Saves eventprobs to filename csv file
    '''
    eventprobs = eventprobs.unstack()
    eventprobs.to_dataframe().to_csv(filename)
    print(f"Saved at {filename}")