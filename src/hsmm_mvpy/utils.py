'''

'''

import numpy as np
import scipy.stats as stats
import xarray as xr
import multiprocessing as mp
import itertools
import pandas as pd
import warnings
from warnings import warn, filterwarnings

filterwarnings('ignore', 'Degrees of freedom <= 0 for slice.', )#weird warning, likely due to nan in xarray, not important but better fix it later 

def read_mne_EEG(pfiles, event_id=None, resp_id=None, epoched=False, sfreq=None, 
                 subj_idx=None, metadata = None, events_provided=None, rt_col='response',
                 verbose=True, tmin=-.2, tmax=5, offset_after_resp = 0, 
                 high_pass=.5, low_pass = None, pick_channels = 'eeg', baseline=(None, 0),
                 upper_limit_RT=5, lower_limit_RT=0.001, reject_threshold=None):
    warn('This method is deprecated and will be removed in future version, use read_mne_data instead', DeprecationWarning, stacklevel=2)
    return read_mne_data(pfiles, event_id, resp_id, epoched, sfreq, 
                 subj_idx, metadata, events_provided, rt_col,
                 verbose, tmin, tmax, offset_after_resp, 
                 high_pass, low_pass, pick_channels, baseline,
                 upper_limit_RT, lower_limit_RT, reject_threshold)

def read_mne_data(pfiles, event_id=None, resp_id=None, epoched=False, sfreq=None, 
                 subj_idx=None, metadata=None, events_provided=None, rt_col='rt', rts=None,
                 verbose=True, tmin=-.2, tmax=5, offset_after_resp = 0, 
                 high_pass=.5, low_pass = None, pick_channels = 'eeg', baseline=(None, 0),
                 upper_limit_RT=np.inf, lower_limit_RT=0, reject_threshold=None, scale=1):
    ''' 
    Reads EEG/MEG data format (.fif or .bdf) using MNE's integrated function .
    
    Notes: 
    - Only EEG or MEG data are selected (other channel types are discarded)
    - All times are expressed on the second scale.
    - If multiple files in pfiles the data of the group is read and seqentially processed.
    - For non epoched data: Reaction Times are only computed if response trigger is in the epoch window (determined by tmin and tmax)
    
    Procedure:
    if data not already epoched:
        0.1) the data is filtered with filters specified in low_pass and high_pass. Parameters of the filter are
        determined by MNE's filter function.
        0.2) if no events is provided, detect events in stumulus channel and keep events with id in event_id and resp_id.
        0.3) eventual downsampling is performed if sfreq is lower than the data's sampling frequency. The event structure is
        passed at the resample() function of MNE to ensure that events are approriately timed after downsampling.
        0.4) epochs are created based on stimulus onsets (event_id) and tmin and tmax. Epoching removes any epoch where a 
        'BAD' annotiation is present and all epochs where an channel exceeds reject_threshold. Epochs are baseline 
        corrected from tmin to stimulus onset (time 0).)
    1) Reaction times (RT) are computed based on the sample difference between onset of stimulus and response triggers. 
        If no response event happens after a stimulus or if RT > upper_limit_RT & < upper_limit_RT, RT is 0.
    2) all the non-rejected epochs with positive RTs are cropped to stimulus onset to stimulus_onset + RT.
    
    Parameters
    ----------
    pfiles : str or list
        list of EEG files to read,
    event_id : dict
        Dictionary containing the correspondance of named condition [keys] and event code [values]
    resp_id : dict
        Dictionary containing the correspondance of named response [keys] and event code [values]
    sfreq : float
        Desired sampling frequency
    to_merge_id: dict
        Dictionary containing the correspondance of named condition [keys] and event code [values] that needs to be
        merged with the stimuli event in event_id
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
        Time taken after onset of the response in seconds
    low_pass : float
        Value of the low pass filter
    high_pass : float
        Value of the high pass filter
    pick_channels: list 
        'eeg' (default) to keep only EEG channels or  list of channel names to keep
    baseline : tuple
        Time values to compute the baseline and substract to epoch data (usually some time before stimulus onset)
    upper_limit_RT : float
        Upper limit for RTs. Longer RTs are discarded
    lower_limit_RT : float
        Lower limit for RTs. Shorter RTs are discarded
    reject_threshold : float
        Rejection threshold to apply when creating epochs, expressed in microvolt
    scale: float
        Scale to apply to the RT data (e.g. 1000 if ms)
        
    Returns
    -------
    epoch_data : xarray
        Returns an xarray Dataset with all the data, events, channels, participant. 
        All eventual participant/channels naming and epochs index are kept. 
        The choosen sampling frequnecy is stored as attribute.
    '''
    import mne
    dict_datatype = {False:'continuous', True:'epoched'}
    epoch_data = [] 
    if isinstance(pfiles,str):#only one participant
        pfiles = [pfiles]
    if not subj_idx:
        subj_idx = ["S"+str(x) for x in np.arange(len(pfiles))]
    if isinstance(subj_idx,str):
        subj_idx = [subj_idx]
    if upper_limit_RT<0 or lower_limit_RT<0:
        raise ValueError('Limit to RTs cannot be negative')
    y = 0
    if metadata is not None:
        if len(pfiles)> 1 and len(metadata) != len(pfiles):
            raise ValueError(f'Incompatible dimension between the provided metadata {len(metadata)} and the number of eeg files provided {len(pfiles)}')
    else:
        metadata_i = None
    for participant in pfiles:

        print(f"Processing participant {participant}'s {dict_datatype[epoched]} {pick_channels}")

        # loading data
        if epoched == False:# performs epoching on raw data
            if '.fif' in participant:
                data = mne.io.read_raw_fif(participant, preload=False, verbose=verbose)
            elif '.bdf' in participant:
                data = mne.io.read_raw_bdf(participant, preload=False, verbose=verbose)
            else:
                raise ValueError(f'Unknown EEG file format for participant {participant}')
            if sfreq is None: 
                sfreq = data.info['sfreq']

            if 'response' not in list(resp_id.keys())[0]:
                resp_id = {f'response/{k}': v for k, v in resp_id.items()}
            if events_provided is None:
                try:
                    events = mne.find_events(data, verbose=verbose, min_duration = 1 / data.info['sfreq'])
                except:
                    events = mne.events_from_annotations(data, verbose=verbose)
                data = _pick_channels(pick_channels,data, stim=True)
                if events[0,1] > 0:#bug from some stim channel, should be 0 otherwise indicates offset in the trggers
                    print(f'Correcting event values as trigger channel has offset {np.unique(events[:,1])}')
                    events[:,2] = events[:,2]-events[:,1]#correction on event value                
                events_values = np.concatenate([np.array([x for x in event_id.values()]), np.array([x for x in resp_id.values()])])
                events = np.array([list(x) for x in events if x[2] in events_values])#only keeps events with stim or response
                events_stim = np.array([list(x) for x in events if x[2] in event_id.values()])#only stim
            else:
                if len(np.shape(events_provided)) == 2:
                    events_provided = events_provided[np.newaxis]
                events = events_provided[y]

            data.load_data()
            data.filter(high_pass, low_pass, fir_design='firwin', verbose=verbose)

            if sfreq < data.info['sfreq']:#Downsampling
                print(f'Downsampling to {sfreq} Hz')
                data, events = data.resample(sfreq, events=events)
            combined =  {**event_id, **resp_id}#event_id | resp_id 
            stim = list(event_id.keys())
            
            if verbose:
                print(f'Creating epochs based on following event ID :{np.unique(events[:,2])}')

            offset_after_resp_samples = int(offset_after_resp*sfreq)
            if metadata is None:
                metadata_i, meta_events, event_id = mne.epochs.make_metadata(
                    events=events, event_id=combined, tmin=tmin, tmax=tmax,
                    sfreq=data.info["sfreq"], row_events=stim, keep_first=["response"])
                metadata_i = metadata_i[["event_name","response"]]#only keep event_names and rts
            else:
                metadata_i = metadata[y]
            epochs = mne.Epochs(data, meta_events, event_id, tmin, tmax, proj=False,
                    baseline=baseline, preload=True, picks=pick_channels,
                    verbose=verbose, detrend=1, on_missing = 'warn', event_repeated='drop',
                    metadata=metadata_i, reject_by_annotation=True, reject=reject_threshold)
            epochs.metadata.rename({'response':'rt'}, axis=1, inplace=True)
            metadata_i = epochs.metadata
        else:
            if '.fif' in participant:
                epochs = mne.read_epochs(participant, preload=True, verbose=verbose)
                if sfreq is None: 
                    sfreq = epochs.info['sfreq']
                elif sfreq  < epochs.info['sfreq']:
                    if verbose:
                        print(f'Resampling data at {sfreq}')
                    epochs = epochs.resample(sfreq)
            else:
                raise ValueError('Incorrect file format')
            _pick_channels(pick_channels,epochs, stim=False)
            if metadata is None:
                try:
                    metadata_i = epochs.metadata#accounts for dropped epochs
                except:
                    raise ValueError('Missing metadata in the epoched data')
            elif isinstance(metadata, pd.DataFrame):
                if len(pfiles)>1:
                    metadata_i = metadata[y].copy()#TODO, better account for participant's wide provided metadata
                else:
                    metadata_i = metadata.copy()
            else:
                raise ValueError('Metadata should be a pandas data-frame as generated by mne or be contained in the passed epoch data')
        offset_after_resp_samples = np.rint(offset_after_resp*sfreq).astype(int)
        valid_epoch_index = [x for x,y in enumerate(epochs.drop_log) if len(y) == 0]
        data_epoch = epochs.get_data()#preserves index
        rts = metadata_i[rt_col]
        if isinstance(metadata_i, pd.DataFrame):
            if len(metadata_i) > len(data_epoch):#assumes metadata contains rejected epochs
                metadata_i = metadata_i.loc[valid_epoch_index]
                rts = metadata_i[rt_col]
            try:
                rts = rts/scale
            except:
                raise ValueError(f'Expected column named {rt_col} in the provided metadata file, alternative names can be passed through the rt_col parameter')
        elif rts is None:
            raise ValueError(f'Expected either a metadata Dataframe or an array of Reaction Times')
        rts_arr = np.array(rts)
        if verbose:
            print(f'Applying reaction time trim to keep RTs between {lower_limit_RT} and {upper_limit_RT} seconds')
        rts_arr[rts_arr > upper_limit_RT] = 0 #removes RT above x sec
        rts_arr[rts_arr < lower_limit_RT] = 0 #removes RT below x sec, important as determines max events
        rts_arr[np.isnan(rts_arr)] = 0#too long trial
        rts_arr = np.rint(rts_arr*sfreq).astype(int)
        if verbose:
            print(f'{len(rts_arr[rts_arr > 0])} RTs kept of {len(rts_arr)} clean epochs')
        triggers = metadata_i.iloc[:,0].values#assumes first col is trigger
        cropped_data_epoch = np.empty([len(rts_arr[rts_arr>0]), len(epochs.ch_names), max(rts_arr)+offset_after_resp_samples])
        cropped_data_epoch[:] = np.nan
        cropped_trigger = []
        epochs_idx = []
        j = 0
        time0 = epochs.time_as_index(0)[0]
        for i in range(len(data_epoch)):
            if rts_arr[i] > 0:
                cropped_trigger.append(triggers[i])
            #Crops the epochs to time 0 (stim onset) up to RT
                cropped_data_epoch[j,:,:rts_arr[i]+offset_after_resp_samples] = \
                (data_epoch[i,:,time0:time0+rts_arr[i]+offset_after_resp_samples])
                epochs_idx.append(valid_epoch_index[i])#Keeps trial number
                j += 1
        x = 0
        while np.isnan(cropped_data_epoch[-1]).all():#Weird bug I guess it is perhps due to too long last epoch? update: cannot reproduce
            cropped_data_epoch = cropped_data_epoch[:-1]
            x += 1
        if x > 0:
            print(f'RTs > 0 longer than expected ({x})')
        print(f'{len(cropped_data_epoch)} trials were retained for participant {participant}')
        if verbose:
            print(f'End sampling frequency is {sfreq} Hz')

        epoch_data.append(hmp_data_format(cropped_data_epoch, epochs.info['sfreq'], None, offset_after_resp_samples, epochs=[int(x) for x in epochs_idx], channels = epochs.ch_names, metadata = metadata_i))

        y += 1
    epoch_data = xr.concat(epoch_data, dim = xr.DataArray(subj_idx, dims='participant'),
                          fill_value={'event':'', 'data':np.nan})
    return epoch_data

def _pick_channels(pick_channels,data,stim=True):
    if isinstance(pick_channels, list):
        try:
            data = data.pick(pick_channels)
        except:
            raise ValueError('incorrect channel pick specified')
    elif pick_channels == 'eeg' or pick_channels == 'meg':
            data = data.pick(pick_channels)
    else:
         raise ValueError('incorrect channel pick specified')
    return data

def parsing_epoched_eeg(data, rts, conditions, sfreq, start_time=0, offset_after_resp=0.1):
    '''
    Function to parse epochs and crop them to start_time (usually stimulus onset so 0) up to the reaction time of the trial.
    The returned object is a xarray Dataset allowing further processing using built-in methods

    Importantly 
    1) if you are considering some lower or upper limit on the RTs you should replace values outside of these ranges
    by np.nan (e.g. rts[rts < 200] = np.nan) or 0
    2) RTs and conditions need to be ordered in the same way as the epochs
    
    
    Parameters
    ----------
    data: pandas.dataframe
        pandas dataframe with columns time (in milliseconds), epoch number and one column for each channel 
        (column name will be taken as channel names)
    rts: list or 1d array
        list of reaction times in milliseconds for each epoch 
    epoch_index: list or 1d array
        number of the index (important for eventual dropped trials during the prepro154,cessing
    channel_index: list or 1d array
        list of name of the channels
    condition: list or 1d array
        list of condition associated with each epoch
    sfreq: float
        sampling frequency of the data
    start_time: float
        time defining the onset of a trial (default is 0 as trial usually defined from event/stimulus onset) in milliseconds
    offset_after_resp: float
        eventual time added after the response (e.g. if we expect later components) in milliseconds
    '''
    tstep = 1000/sfreq#time step
    offset_after_resp_samples = int(offset_after_resp/tstep)
    epoch_size = len(data.time.unique())*tstep
    if not isinstance(rts, np.ndarray):
        try:#pandas or xarray
            rts = rts.values
        except:
            raise ValueError('RTs should either be a numpy array or a pandas serie')
    if not isinstance(conditions, np.ndarray):
        try:#pandas or xarray
            conditions = conditions.values
        except:
            raise ValueError('Conditions should either be a numpy array or a pandas serie')
    if any(t > epoch_size for t in rts):
        print('Zeroing out RTs longer than epoch size, you might want to zero out rts that are outside of the range of interest')
        rts[rts > epoch_size] = 0
    rts[np.isnan(rts)] = 0
    rts = np.array([int(x) for x in rts/tstep])
    data = data[data.time >= start_time]#remove all baseline values
    epochs = data.epoch.unique()
    rts = rts[epochs]
    conditions = conditions[epochs]
    times = data.time.unique()
    data = data.drop(columns=['time', 'epoch'])
    #channel names/columns are assumed to be remaining columns affter removing time and epoch columns
    channel_columns = [x for x in data.columns]
    data = data.values.flatten()
    data = data.reshape((len(epochs), len(times),len(channel_columns)))
    data = np.swapaxes(data,1,2)

    nan_con = np.array([i for i,x in enumerate(conditions) if isinstance(x, float) and np.isnan(x)])
    if len(nan_con) > 0:
        print(f'NaN present in condition array, removing associated epoch and RT ({nan_con})')
        data = np.delete(data, nan_con, axis=0)
        conditions = np.delete(conditions, nan_con, axis=0)
        rts = np.delete(rts, nan_con, axis=0)
        epochs =  np.delete(epochs, nan_con, axis=0)
    epoch = 0
    for rt in rts:
        if rt == 0:
            data[epoch,:,:] = np.nan
            conditions[epoch] = ''
            #rts[epoch] = None#np.nan#np.delete(rts, epoch, axis=0)
            #epochs[epoch] = np.nan#np.delete(epochs, epoch, axis=0)
        epoch += 1
    cropped_conditions = []
    epoch_idx = []
    cropped_data_epoch = np.empty([len(epochs), len(channel_columns), max(rts)+offset_after_resp_samples])
    cropped_data_epoch[:] = np.nan
    j = 0
    for epoch in np.arange(len(data)):
        #Crops the epochs up to RT
        cropped_data_epoch[j,:,:rts[epoch]+offset_after_resp_samples] = \
        (data[epoch,:,:rts[epoch]+offset_after_resp_samples])
        j += 1
    print(f'Totaling {len(cropped_data_epoch)} valid trials')

    data_xr = hmp_data_format(cropped_data_epoch, sfreq, conditions, offset_after_resp_samples, epochs=epochs, channels = channel_columns)
    return data_xr

def hmp_data_format(data, sfreq, events=None, offset=0, participants=[], epochs=None, channels=None, metadata=None):

    '''
    Converting 3D matrix with dimensions (participant) * trials * channels * sample into xarray Dataset
    
    Parameters
    ----------
    data : ndarray
        4/3D matrix with dimensions (participant) * trials * channels * sample  
    events : ndarray
        np.array with 3 columns -> [samples of the event, initial value of the channel, event code]. To use if the
        automated event detection method of MNE is not appropriate 
    sfreq : float
        Sampling frequency of the data
    participants : list
        List of participant index
    epochs : list
        List of epochs index
    channels : list
        List of channel index
    '''
    if len(np.shape(data)) == 4:#means group
        n_subj, n_epochs, n_channels, n_samples = np.shape(data)
    elif len(np.shape(data)) == 3:
        n_epochs, n_channels, n_samples = np.shape(data)
        n_subj = 1
    else:
        raise ValueError(f'Unknown data format with dimensions {np.shape(data)}')
    if channels is None:
        channels = np.arange(n_channels)
    if epochs is None:
         epochs = np.arange(n_epochs)
    if n_subj < 2:
        data = xr.Dataset(
                {
                    "data": (["epochs", "channels", "samples"],data),
                },
                coords={
                    "epochs" :epochs,
                    "channels":  channels,
                    "samples": np.arange(n_samples)
                },
                attrs={'sfreq':sfreq,'offset':offset}
                )
    else:
        data = xr.Dataset(
                {
                    "data": (['participant',"epochs", "channels", "samples"],data),
                },
                coords={
                    'participant':participants,
                    "epochs" :epochs,
                    "channels":  channels,
                    "samples": np.arange(n_samples)
                },
                attrs={'sfreq':sfreq,'offset':offset}
                )
    if metadata is not None:
        metadata = metadata.loc[epochs]
        metadata = metadata.to_xarray()
        metadata = metadata.rename_dims({'index':'epochs'})
        metadata = metadata.rename_vars({'index':'epochs'})
        data = data.merge(metadata)
        data = data.set_coords(list(metadata.data_vars))
    if events is not None:
        data['events'] = xr.DataArray(
            events,
            dims=("participant", "epochs"),
            coords={"participant": participants, "epochs": epochs})
        data = data.set_coords('events')
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


def stack_data(data, subjects_variable='participant', channel_variable='component', single=False):
    '''
    Stacks the data going from format [participant * epochs * samples * channels] to [samples * channels]
    with sample indexes starts and ends to delimitate the epochs.
    
    
    Parameters
    ----------
    data : xarray
        unstacked xarray data from transform_data() or anyother source yielding an xarray with dimensions 
        [participant * epochs * samples * channels] 
    subjects_variable : str
        name of the dimension for subjects ID
    single : bool 
        Whether participant is unique (True) or a group of participant (False)
    
    Returns
    -------
    data : xarray.Dataset
        xarray dataset [samples * channels]
    '''    
    if isinstance(data, (xr.DataArray,xr.Dataset)) and 'component' not in data.dims:
        data = data.rename_dims({'channels':'component'})
        print(data.participant)
    if "participant" not in data.dims:
        data = data.expand_dims("participant")
    data = data.stack(all_samples=['participant','epochs',"samples"]).dropna(dim="all_samples")
    return data #xr.Dataset({'data':data, 'durations':durations})
    #return data, durations

def transform_data(data, subjects_variable="participant", apply_standard=True,  apply_zscore=True, method='pca', n_comp=None, return_weights=False):
    '''
    Adapts EEG epoched data (in xarray format) to the expected data format for hmps. 
    First this code can apply standardization of individual variances (if apply_standard=True).
    Second, a spatial PCA on the average variance-covariance matrix is performed (if method='pca', more methods in development)
    Third,stacks the data going from format [participant * epochs * samples * channels] to [samples * channels]
    Last, performs z-scoring on each epoch and for each principal component (PC)
    
    
    Parameters
    ----------
    data : xarray
        unstacked xarray data from transform_data() or anyother source yielding an xarray with dimensions 
        [participant * epochs * samples * channels] 
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

    Returns
    -------
    data : xarray.Dataset
        xarray dataset [n_samples * n_comp] data expressed in the PC space, ready for hsMM fit
    pca_data : xarray.Dataset
        loadings of the PCA, used to retrieve channel space
    pca.explained_variance_ : ndarray
        explained variance for each component
    means : xarray.DataArray
        means of the channels before PCA/zscore
    '''
    if isinstance(data, xr.DataArray):
        raise ValueError('Expected a xarray Dataset with data and event as DataArrays, check the data format')
    if apply_standard:
        mean_std = data.groupby(subjects_variable).std(dim=...).data.mean()
        data = data.assign(mean_std=mean_std.data)
        data = data.groupby(subjects_variable).map(standardize)
    if method == 'pca':
        from sklearn.decomposition import PCA
        var_cov_matrices = []
        if isinstance(data, xr.Dataset):
            data = data.data
        for i,trial_dat in data.stack(trial=("participant", "epochs")).drop_duplicates('trial').groupby('trial'):
            var_cov_matrices.append(vcov_mat(trial_dat)) #Would be nice not to have a for loop but groupby.map seem to fal
        var_cov_matrix = np.mean(var_cov_matrices,axis=0)
        # Performing spatial PCA on the average var-cov matrix
        if n_comp == None:
            import matplotlib.pyplot as plt
            n_comp = np.shape(var_cov_matrix)[0]-1
            fig, ax = plt.subplots(1,2, figsize=(.2*n_comp, 4))
            pca = PCA(n_components=n_comp, svd_solver='full')#selecting Principale components (PC)
            pca_data = pca.fit_transform(var_cov_matrix.T)
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
            n_comp = int(input(f'How many PCs (90 and 99% explained variance at component n{np.where(np.cumsum(var/np.sum(var)) >= .90)[0][0]+1} and n{np.where(np.cumsum(var/np.sum(var)) >= .99)[0][0]+1})?'))

        pca = PCA(n_components=n_comp, svd_solver='full')#selecting Principale components (PC)

        pca_data = pca.fit_transform(var_cov_matrix)/pca.explained_variance_ # divided by explained var for compatibility with matlab's PCA
        
        #Rebuilding pca PCs as xarray to ease computation
        coords = dict(channels=("channels", data.coords["channels"].values),
                     component=("component", np.arange(n_comp)))
        pca_data = xr.DataArray(pca_data, dims=("channels","component"), coords=coords)
        means = data.groupby('channels').mean(...)
        data = data @ pca_data

    elif method is None:
        data = data.rename({'channels':'component'})
        data['component'] = np.arange(len(data.component ))
    if apply_zscore:
        data = data.stack(trial=[subjects_variable,'epochs','component']).groupby('trial').map(zscore).unstack()
    if stack_data:
        data = stack_data(data)
    if return_weights:
        return data, pca_data, pca.explained_variance_, means
    else:
        return data
    

def LOOCV(data, subject, n_events, initial_fit, sfreq, event_width=50):
    '''
    Performs Leave-one-out cross validation, removes one participant from data, estimate n_events HMP parameters, 
    compute the likelihood of the data from the left out participant with the estimated parameters. The model is fit
    using initial fit as starting points for magnitudes and parameters
    
    
    Parameters
    ----------
    data : xarray.Dataset
        xarray data from transform_data() 
    subject : str
        name of the subject to remove
    n_events : int 
        How many events in the model
    initial_fit : xarray.Dataset
        Fit of the model with the same number of events and all participants
    sfreq : float
        Sampling frequency of the data
    event_width : float
        length of the events in milliseconds
    
    Returns
    -------
    likelihood : float
        likelihood computed for the left-out participant
    subject : str
        name of the subject to remove
    '''
    # wc 
    from hsmm_mvpy.models import hmp
    #Looping over possible number of events
    subjects_idx = data.participant.values
    likelihoods_loo = []
    #Extracting data without left out subject
    stacked_loo = stack_data(data.sel(participant= subjects_idx[subjects_idx!=subject],drop=False))
    #Fitting the HMP using previous estimated parameters as initial parameters
    model_loo = hmp(stacked_loo, sfreq=sfreq, event_width=event_width)
    fit = model_loo.fit_single(n_events, initial_fit.magnitudes.dropna('event').values, initial_fit.parameters, 1, verbose=False)
    #Evaluating likelihood for left out subject
    #Extracting data of left out subject
    stacked_left_out = stack_data(data.sel(participant=subject, drop=False))
    model_left_out = hmp(stacked_left_out, sfreq=sfreq, event_width=event_width)
    likelihood = model_left_out.estim_probs(fit.magnitudes.dropna('event').values, fit.parameters, n_events,True)
    return likelihood, subject

def loocv_estimation(data, subject, sfreq, event_width):
    '''
    Performs Leave-one-out cross validation, removes one participant from data, estimate n_events HMP parameters, 
    compute the likelihood of the data from the left out participant with the estimated parameters. The model is fit
    using initial fit as starting points for magnitudes and parameters
    
    
    Parameters
    ----------
    data : xarray
        xarray data from transform_data() 
    subject : str
        name of the subject to remove
    sfreq : float
        Sampling frequency of the data
    event_width : float
        length of the events in milliseconds
    
    Returns
    -------
    likelihood : float
        likelihood computed for the left-out participant
    subject : str
        name of the subject to remove
    '''    
    print(f'Leaving out participant #{subject}')
    from hsmm_mvpy.models import hmp
    #Looping over possible number of events
    subjects_idx = data.participant.values
    likelihoods_loo = []
    #Extracting data without left out subject
    stacked_loo = stack_data(data.sel(participant= subjects_idx[subjects_idx!=subject],drop=False))
    #Fitting the HMP using previous estimated parameters as initial parameters
    model_loo = hmp(stacked_loo, sfreq=sfreq, event_width=event_width, cpus=1)
    parameters, magnitudes, likelihoods = model_loo.sliding_event(verbose=False)
    estimates = model_loo.iterative_fit(likelihoods=likelihoods, parameters=parameters, magnitudes=magnitudes)
    #Evaluating likelihood for left out subject
    #Extracting data of left out subject
    stacked_left_out = stack_data(data.sel(participant=subject, drop=False))
    model_left_out = hmp(stacked_left_out, sfreq=sfreq, event_width=event_width, cpus=1)
    n_events = int(estimates.dropna('n_events',how='all').n_events.max())
    for n_event in range(1,n_events+1):
        likelihoods_loo.append( model_left_out.calc_EEG_50h(estimates.sel(n_events=n_event).magnitudes.dropna('event').values, estimates.sel(n_events=n_event).parameters.dropna('stage').values, n_event, True))
    return likelihoods_loo, subject

def loocv(stacked_data,sfreq, max_event, cpus=1, event_width=50):
    '''
    Performs Leave-one-out cross validation, removes one participant from data, estimate n_events HMP parameters, 
    compute the likelihood of the data from the left out participant with the estimated parameters. The model is fit
    using initial fit as starting points for magnitudes and parameters
    
    
    Parameters
    ----------
    data : xarray.Dataset
        xarray data from transform_data() 
    initial_fit : xarray.Dataset
        Fit of the model with the same number of events and all participants
    sfreq : float
        Sampling frequency of the data
    event_width : float
        length of the events in milliseconds
    
    Returns
    -------
    loocv
    '''
    unstacked_data = stacked_data.unstack()
    #Looping over possible number of events
    participants = unstacked_data.participant.data
    likelihoods_loo = []
    loocv = []
    if cpus>1:
        import multiprocessing
        with multiprocessing.Pool(processes=cpus) as pool:
            loo = pool.starmap(loocv_estimation, 
                zip(itertools.repeat(unstacked_data), participants, itertools.repeat(sfreq), itertools.repeat(event_width)))
        loocv.append(loo)
    else:
        loo = []
        for participant in participants:
            loo.append(loocv_estimation(unstacked_data, participant,sfreq, event_width))
        loocv.append(loo)
    loocv_arr = np.tile(np.nan, (max_event, len(participants)))
    par_arr = np.repeat(np.nan, len(participants))
    for idx, values in enumerate(loocv[0]):
        par_arr[idx] = np.array(values[-1])
        values = np.array(values[:-1][0])
        loocv_arr[:len(values), idx] = values 
    loocv = xr.DataArray(loocv_arr, coords={"n_event":np.arange(1,max_event+1),
                                                           "participants":par_arr}, name="loo_likelihood")
    return loocv


def loocv_mp(init, stacked_data, bests, func=LOOCV, cpus=2, verbose=True):
    '''
    multiprocessing wrapper for LOOCV()
    
    Parameters
    ----------
    init : hmp.model
        initialized hmp model
    data : xarray.Dataset
        xarray data from transform_data() , can also be a subset, e.g. based on conditions
    bests : xarray.Dataset
        Fit from all possible n event solution
    
    Returns
    -------
    loocv
    '''
    # warn('This method is deprecated and will be removed in future version, use loocv() instead', DeprecationWarning, stacklevel=2) 
    unstacked_data = stacked_data.unstack()
    import multiprocessing
    import itertools
    participants = unstacked_data.participant.data
    likelihoods_loo = []
    loocv = []
    for n_events in bests.n_events.values:
        if verbose:
            print(f'LOOCV for model with {n_events} event(s)')
        with multiprocessing.Pool(processes=cpus) as pool:
            loo = pool.starmap(func, 
                zip(itertools.repeat(unstacked_data), participants, itertools.repeat(n_events), 
                    itertools.repeat(bests.sel(n_events=n_events)), itertools.repeat(init.sfreq)))
        loocv.append(loo)

    loocv = xr.DataArray(np.array(loocv)[:,:,0].astype(np.float64), coords={"n_event":np.arange(1,bests.n_events.max().values+1)[::-1],
                                                           "participants":np.array(loocv)[0,:,1]}, name="loo_likelihood")
    return loocv

def reconstruct(magnitudes, PCs, eigen, means):
    '''
    Reconstruct channel activity from PCA
    
    Parameters
    ----------
    magnitudes :  
        2D or 3D ndarray with [n_components * n_events] can also contain several estimations [estimation * n_components * n_events]
    PCs : 
        2D ndarray with PCA loadings [channels x n_components]
    eigen : 
        PCA eigenvalues of the covariance matrix of data [n_components]
    means : 
        Grand mean [channels]
        
    Returns
    -------
    channels : ndarray
        a 2D ndarray with [channels * events]
    '''
    if len(np.shape(magnitudes))==2:
        magnitudes = np.array([magnitudes])
    n_iter, n_comp, n_events = np.shape(magnitudes)
    list_channels = []
    for iteration in np.arange(n_iter): 
        #channels = np.array(magnitudes[iteration].T * 
        channels =  np.array(magnitudes[iteration, ].T * np.tile(np.sqrt(eigen[:n_comp]).T, (n_events,1))) @ np.array(PCs[:,:n_comp]).T
        list_channels.append(channels + np.tile(means,(n_events,1)))#add means for each channel
    return np.array(list_channels)

def stage_durations(times):
    '''
    Returns the stage durations from the event onset times by substracting each stage to the previous
    
    Parameters
    ----------
    times : ndarray
        2D ndarray with [n_trials * n_events]
    
    Returns
    -------
    times : ndarray
        2D ndarray with [n_trials * n_events]
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

def event_times(data, times, channel, stage):
    '''
    Event times parses the single trial EEG signal of a given channel in a given stage, from event onset to the next one. If requesting the last
    stage it is defined as the onset of the last event until the response of the participants.

    Parameters
    ----------
    data : xr.Dataset
        HMP EEG data (untransformed but with trial and participant stacked)
    times : xr.DataArray
        Onset times as computed using onset_times()
    channel : str
        channel to pick for the parsing of the signal
    stage : float | ndarray
        Which stage to parse the signal into

    Returns
    -------
    brp_data : ndarray
        Matrix with trial_x_participant * samples with sample dimension given by the maximum stage duration
    '''

    brp_data = np.tile(np.nan, (len(data.trial_x_participant), int(round(max(times.sel(event=stage+1).data- times.sel(event=stage).data)))+1))
    i=0
    for trial, trial_dat in data.groupby('trial_x_participant'):
        trial_time = slice(times.sel(event=stage, trial_x_participant=trial), \
                                                 times.sel(event=stage+1, trial_x_participant=trial))
        trial_elec = trial_dat.sel(channels = channel, samples=trial_time).squeeze()
        try:
            brp_data[i, :len(trial_elec)] = trial_elec
        except:
            brp_data[i, :1] = trial_elec
            
        i += 1

    return brp_data    
    
def condition_selection(hmp_data, eeg_data, condition_string, variable='event'):
    unstacked = hmp_data.unstack().where(eeg_data[variable].str.contains(condition_string),drop=True)
    stacked = stack_data(unstacked)
    return stacked

def load_data(path):
    return xr.load_dataset(path)

    
def participant_selection(hmp_data, eeg_data, participant):
    unstacked = hmp_data.unstack().sel(participant = participant)
    stacked = stack_data(unstacked)
    return stacked

def bootstrapping(init, hmp_data, general_run, positions, eeg_data, iterations, threshold=1, verbose=True, plots=True, cpus=1):
    warn('This method is inaccurate and will be removed in future version, see the bootstraping function in the resample module instead', DeprecationWarning, stacklevel=2)
    from hsmm_mvpy.models import hmp
    from hsmm_mvpy.visu import plot_topo_timecourse
    try:
        import xskillscore as xs#Todo remove from dependency list
    except:
        raise ValueError('xskillscore should be installed to run this (deprecated) function')
    fitted_mags = general_run.magnitudes.values[np.unique(np.where(np.isfinite(general_run.magnitudes))[0]),:]#remove NAs
    mags_boot_mat = []#np.tile(np.nan, (iterations, init.compute_max_events(), init.n_dims))
    pars_boot_mat = []#np.tile(np.nan, (iterations, init.compute_max_events()+1, 2))

    for i in range(iterations):
        bootstapped = xs.resample_iterations(hmp_data.unstack(), iterations=1, dim='epochs')
        hmp_data_boot = stack_data(bootstapped.squeeze())
        init_boot = hmp(hmp_data_boot, sfreq=eeg_data.sfreq, event_width=init.event_width, cpus=init.cpus)
        estimates_boot = init_boot.fit(verbose=verbose, threshold=threshold)
        mags_boot_mat.append(estimates_boot.magnitudes)
        pars_boot_mat.append(estimates_boot.parameters)
        if plots:
            plot_topo_timecourse(eeg_data, estimates_boot, positions, init_boot)

    all_pars_aligned = np.tile(np.nan, (iterations, np.max([len(x) for x in pars_boot_mat]), 2))
    all_mags_aligned = np.tile(np.nan, (iterations, np.max([len(x) for x in mags_boot_mat]), init.n_dims))
    for iteration, _i in enumerate(zip(pars_boot_mat, mags_boot_mat)):
        all_pars_aligned[iteration, :len(_i[0]), :] = _i[0]
        all_mags_aligned[iteration, :len(_i[1]), :] = _i[1]

    booted = xr.Dataset({'parameters': (('iteration', 'stage','parameter'), 
                                 all_pars_aligned),
                        'magnitudes': (('iteration', 'event','component'), 
                                 all_mags_aligned)})
    return booted
