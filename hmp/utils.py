'''

'''

import numpy as np
from scipy.special import gamma as gamma_func
from scipy.stats import lognorm
import xarray as xr
import multiprocessing as mp
import itertools
import pandas as pd
from pandas import MultiIndex
import warnings
from warnings import warn, filterwarnings
from hmp import mcca
import json
import mne
import os

filterwarnings('ignore', 'Degrees of freedom <= 0 for slice.', )#weird warning, likely due to nan in xarray, not important but better fix it later 
filterwarnings('ignore', 'Mean of empty slice')#When trying to center all-nans trials

def gamma_scale_to_mean(scale, shape): 
    return scale*shape
def gamma_mean_to_scale(mean, shape): 
    return mean/shape

def logn_scale_to_mean(scale, shape): 
    return np.exp(scale+shape**2/2)
def logn_mean_to_scale(mean, shape):
    return np.exp(np.log(mean)-(shape**2/2))

def wald_scale_to_mean(scale, shape): 
    return scale*shape
def wald_mean_to_scale(mean, shape): 
    return mean/shape

def weibull_scale_to_mean(scale, shape):
    return scale*gamma_func(1+1/shape)
def weibull_mean_to_scale(mean, shape): 
    return mean/gamma_func(1+1/shape)


def read_mne_data(pfiles, event_id=None, resp_id=None, epoched=False, sfreq=None, 
                 subj_idx=None, metadata=None, events_provided=None, rt_col='rt', rts=None,
                 verbose=True, tmin=-.2, tmax=5, offset_after_resp = 0, 
                 high_pass=None, low_pass = None, pick_channels = 'eeg', baseline=(None, 0),
                 upper_limit_RT=np.inf, lower_limit_RT=0, reject_threshold=None, scale=1, reference=None, ignore_rt=False):
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
        0.2) if no events is provided, detect events in stimulus channel and keep events with id in event_id and resp_id.
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
        Rejection threshold to apply after cropping the epoch to the end of the sequence (e.g. RT), expressed in the unit of the data
    scale: float
        Scale to apply to the RT data (e.g. 1000 if ms)
    reference:
        What reference to use (see MNE documentation), if None, keep the existing one
    ignore_rt: bool
        Use RT to parse the epochs (False, Default) or ignore the RT and parse up to tmaxx in epochs (True)
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
    ev_i = 0 #syncing up indexing between event and raw files
    for participant in pfiles:

        print(f"Processing participant {participant}'s {dict_datatype[epoched]} {pick_channels}")

        # loading data
        if epoched == False:# performs epoching on raw data
            if '.fif' in participant:
                data = mne.io.read_raw_fif(participant, preload=True, verbose=verbose)
            elif '.bdf' in participant:
                data = mne.io.read_raw_bdf(participant, preload=True, verbose=verbose)
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
                    events = mne.events_from_annotations(data, verbose=verbose)[0]
                if events[0,1] > 0:#bug from some stim channel, should be 0 otherwise indicates offset in the trggers
                    print(f'Correcting event values as trigger channel has offset {np.unique(events[:,1])}')
                    events[:,2] = events[:,2]-events[:,1]#correction on event value             
                events_values = np.concatenate([np.array([x for x in event_id.values()]), np.array([x for x in resp_id.values()])])
                events = np.array([list(x) for x in events if x[2] in events_values])#only keeps events with stim or response
                events_stim = np.array([list(x) for x in events if x[2] in event_id.values()])#only stim
            else:
                if len(events_provided[0]) == 3:
                    events_provided = events_provided[np.newaxis]
                    events = events_provided[y]
                else:#assumes stacked event files
                    events = events_provided[ev_i]
                    ev_i += 1
            if reference is not None:
                data = data.set_eeg_reference(reference)
            data = _pick_channels(pick_channels,data, stim=True)
            data.load_data()

            if sfreq < data.info['sfreq']:#Downsampling
                print(f'Downsampling to {sfreq} Hz')
                decim = np.round( data.info['sfreq'] / sfreq).astype(int)
                obtained_sfreq = data.info['sfreq'] / decim
                low_pass = obtained_sfreq / 3.1
            else: 
                decim = 1
                if sfreq > data.info['sfreq']+1:
                    warn(f'Requested higher frequency {sfreq} than found in the EEG data, no resampling is performed')
            if high_pass is not None or low_pass is not None:
                data.filter(high_pass, low_pass, fir_design='firwin', verbose=verbose)
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
                    baseline=baseline, preload=True, picks=pick_channels, decim=decim,
                    verbose=verbose, detrend=None, on_missing = 'warn', event_repeated='drop',
                    metadata=metadata_i, reject_by_annotation=True)
            epochs.metadata.rename({'response':'rt'}, axis=1, inplace=True)
            metadata_i = epochs.metadata
        else:
            if '.fif' in participant:
                epochs = mne.read_epochs(participant, preload=True, verbose=verbose)
                if high_pass is not None or low_pass is not None:
                    epochs.filter(high_pass, low_pass, fir_design='firwin', verbose=verbose)
                if sfreq is None: 
                    sfreq = epochs.info['sfreq']
                elif sfreq  < epochs.info['sfreq']:
                    if verbose:
                        print(f'Resampling data at {sfreq}')
                    epochs = epochs.resample(sfreq)
            else:
                raise ValueError('Incorrect file format')
            if reference is not None:
                epochs = epochs.set_eeg_reference(reference)
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
        if upper_limit_RT == np.inf:
            upper_limit_RT = epochs.tmax-offset_after_resp+1*(1/sfreq)
        if ignore_rt:
            metadata_i[rt_col] = epochs.tmax
        offset_after_resp_samples = np.rint(offset_after_resp*sfreq).astype(int)
        valid_epoch_index = [x for x,y in enumerate(epochs.drop_log) if len(y) == 0]
        try:#Differences among MNE's versions
            data_epoch = epochs.get_data(copy=False)#preserves index
        except:
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
        if reject_threshold is None:
            reject_threshold = np.inf
        rej = 0
        time0 = epochs.time_as_index(0)[0]
        for i in range(len(data_epoch)):
            if rts_arr[i] > 0:
                #Crops the epochs to time 0 (stim onset) up to RT
                if (np.abs(data_epoch[i,:,time0:time0+rts_arr[i]+offset_after_resp_samples]) < reject_threshold).all():
                    cropped_data_epoch[j,:,:rts_arr[i]+offset_after_resp_samples] = \
                    (data_epoch[i,:,time0:time0+rts_arr[i]+offset_after_resp_samples])
                    epochs_idx.append(valid_epoch_index[i])#Keeps trial number
                    cropped_trigger.append(triggers[i])
                    j += 1
                else:
                    rej += 1
                    rts_arr[i] = 0 
        while np.isnan(cropped_data_epoch[-1]).all():#Remove excluded epochs based on rejection
            cropped_data_epoch = cropped_data_epoch[:-1]
        if ~np.isinf(reject_threshold):
            print(f'{rej} trial rejected based on threshold of {reject_threshold}')
        print(f'{len(cropped_data_epoch)} trials were retained for participant {participant}')
        if verbose:
            print(f'End sampling frequency is {sfreq} Hz')

        epoch_data.append(hmp_data_format(cropped_data_epoch, epochs.info['sfreq'], None, offset_after_resp_samples, epochs=[int(x) for x in epochs_idx], channels = epochs.ch_names, metadata = metadata_i))

        y += 1
    epoch_data = xr.concat(epoch_data, dim = xr.DataArray(subj_idx, dims='participant'),
                          fill_value={'event':'', 'data':np.nan})
    n_trials = (~np.isnan(epoch_data.data[:,:, :, 0].data)).sum(axis=1)[:,0].sum()#Compute number of trials based on trials where first sample is nan
    epoch_data = epoch_data.assign_attrs(lowpass=epochs.info['lowpass'], highpass=epochs.info['highpass'],
                                         lower_limit_RT=lower_limit_RT, upper_limit_RT=upper_limit_RT, 
                                         reject_threshold=reject_threshold, n_trials=n_trials,
                                         

)
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

def _standardize(x):
    '''
    Scaling variances to mean variance of the group
    '''
    return ((x.data / x.data.std(dim=...))*x.mean_std)

def _center(data):
    '''
    center the data
    '''
    mean_last_dim = np.nanmean(data.values, axis=-1)
    mean_last_dim_expanded = np.expand_dims(mean_last_dim, axis=-1)
    centred = data.values - mean_last_dim_expanded
    data.values = centred

    return data

def zscore_xarray(data):
    '''
    zscore of the data in an xarray, avoiding any nans
    '''
    if isinstance(data, xr.Dataset):#if no PCA
        data = data.data
    non_nan_mask = ~np.isnan(data.values)
    if non_nan_mask.any(): #if not everything is nan, calc zscore
        data.values[non_nan_mask] = (data.values[non_nan_mask] - data.values[non_nan_mask].mean()) / data.values[non_nan_mask].std()
    return data

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
    if "participant" not in data.dims:
        data = data.expand_dims("participant")
    data = data.stack(all_samples=['participant','epochs',"samples"]).dropna(dim="all_samples")
    return data

def _filtering(data, filter, sfreq):
    print("NOTE: filtering at this step is suboptimal, filter before epoching if at all possible, see")
    print("also https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html")
    from mne.filter import filter_data

    lfreq, hfreq = filter
    n_participant, n_epochs, _, _ = data.data.values.shape
    for pp in range(n_participant):
        for trial in range(n_epochs):

            dat = data.data.values[pp, trial, :, :]

            if not np.isnan(dat).all():
                dat = dat[:,~np.isnan(dat[0,:])] #remove nans

                #pad by reflecting the whole trial twice
                trial_len = dat.shape[1] * 2
                dat = np.pad(dat, ((0,0),(trial_len,trial_len)), mode='reflect')

                #filter
                dat = filter_data(dat, sfreq, lfreq, hfreq, verbose=False)

                #remove padding
                dat = dat[:,trial_len:-trial_len]
                data.data.values[pp, trial, :, :dat.shape[1]] = dat
        return data

def _pca(pca_ready_data, n_comp, channels):
    from sklearn.decomposition import PCA
    # pca_ready_data = pca_ready_data.transpose(...,'channels')
    if n_comp == None:
        import matplotlib.pyplot as plt
        n_comp = np.shape(pca_ready_data)[0]-1
        fig, ax = plt.subplots(1,2, figsize=(.2*n_comp, 4))
        pca = PCA(n_components=n_comp, svd_solver='full')#selecting Principale components (PC)
        pca.fit(pca_ready_data)

        ax[0].plot(np.arange(pca.n_components)+1, pca.explained_variance_ratio_,'.-')
        ax[0].set_ylabel('Normalized explained variance')
        ax[0].set_xlabel('Component')
        ax[1].plot(np.arange(pca.n_components)+1, np.cumsum(pca.explained_variance_ratio_),'.-')
        ax[1].set_ylabel('Cumulative normalized explained variance')
        ax[1].set_xlabel('Component')
        plt.tight_layout()
        plt.show()
        n_comp = int(input(f'How many PCs (95 and 99% explained variance at component n{np.where(np.cumsum(pca.explained_variance_ratio_) >= .95)[0][0]+1} and n{np.where(np.cumsum(pca.explained_variance_ratio_) >= .99)[0][0]+1}; components till n{np.where(pca.explained_variance_ratio_ >= .01)[0][-1]+1} explain at least 1%)?'))

    pca = PCA(n_components=n_comp, svd_solver='full')#selecting Principale components (PC)
    pca.fit(pca_ready_data)
    #Rebuilding pca PCs as xarray to ease computation
    coords = dict(channels=("channels", channels),
                 component=("component", np.arange(n_comp)))
    pca_weights = xr.DataArray(pca.components_.T, dims=("channels","component"), coords=coords)
    return pca_weights

def transform_data(epoch_data, participants_variable="participant", apply_standard=False, averaged=False, apply_zscore='trial', zscore_acrossPCs=False, method='pca', cov=True, centering=True, n_comp=None, n_ppcas=None, pca_weights=None, bandfilter=None, mcca_reg=0):
    '''
    Adapts EEG epoched data (in xarray format) to the expected data format for hmps. 
    First this code can apply standardization of individual variances (if apply_standard=True).
    Second, a spatial PCA on the average variance-covariance matrix is performed (if method='pca', more methods in development)
    Third,stacks the data going from format [participant * epochs * samples * channels] to [samples * channels]
    Last, performs z-scoring on each epoch and for each principal component (PC), or for each participant and PC,
    or across all data for each PC.
        
    Parameters
    ----------
    data : xarray
        unstacked xarray data from transform_data() or anyother source yielding an xarray with dimensions 
        [participant * epochs * samples * channels] 
    participants_variable : str
        name of the dimension for participants ID
    apply_standard : bool 
        Whether to apply standardization of variance between participants, recommended when they are few of them (e.g. < 10)
    averaged : bool
        Applying the pca on the averaged ERP (True) or single trial ERP (False, default). No effect if cov = True
    apply_zscore : str 
        Whether to apply z-scoring and on what data, either None, 'all', 'participant', 'trial', for zscoring across all data, by participant, or by trial, respectively. If set to true, evaluates to 'trial' for backward compatibility.
    method : str
        Method to apply, 'pca' or 'mcca'
    cov : bool
        Wether to apply the pca/mcca to the variance covariance (True, default) or the epoched data
    n_comp : int
        How many components to select from the PC space, if None plots the scree plot and a prompt requires user
        to specify how many PCs should be retained
    n_ppcas : int
        If method = 'mcca', controls the number of components retained for the by-participant PCAs
    pca_weigths : xarray
        Weights of a PCA to apply to the data (e.g. in the resample function)
    bandfilter: None | (lfreq, hfreq) 
        If none, no filtering is appliedn. If tuple, data is filtered between lfreq-hfreq.
        NOTE: filtering at this step is suboptimal, filter before epoching if at all possible, see
              also https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html
    mcca_reg: float
        regularization used for the mcca computation (see mcca.py)

    Returns
    -------
    data : xarray.Dataset
        xarray dataset [n_samples * n_comp] data expressed in the PC space, ready for HMP fit
    '''
    data = epoch_data.copy(deep=True)
    if isinstance(data, xr.DataArray):
        raise ValueError('Expected a xarray Dataset with data and event as DataArrays, check the data format')
    if not apply_zscore in ['all', 'participant', 'trial'] and not isinstance(apply_zscore,bool):
        raise ValueError('apply_zscore should be either a boolean or one of [\'all\', \'participant\', \'trial\']')
    assert np.sum(np.isnan(data.groupby('participant', squeeze=False).mean(['epochs','samples']).data.values)) == 0,\
        'at least one participant has an empty channel'
    if method == 'mcca' and data.sizes['participant'] == 1:
        raise ValueError('MCCA cannot be applied to only one participant')
    sfreq = data.sfreq
    if bandfilter:
        data = _filtering(data, bandfilter, sfreq)
    if apply_standard:
        if 'participant' not in data.dims or len(data.participant) == 1:
            warn('Requested standardization of between participant variance yet no participant dimension is found in the data or only one participant is present. No standardization is done, set apply_standard to False to avoid this warning.')
        else:
            mean_std = data.groupby(participants_variable, squeeze=False).std(dim=...).data.mean()
            data = data.assign(mean_std=mean_std.data)
            data = data.groupby(participants_variable, squeeze=False).map(_standardize)
    if isinstance(data, xr.Dataset):#needs to be a dataset if apply_standard is used
            data = data.data
    if centering or method=='mcca':
        data = _center(data)
    if apply_zscore == True:
        apply_zscore = 'trial' #defaults to trial
    data = data.transpose('participant','epochs','channels','samples')
    if method == 'pca':
        if pca_weights is None:
            if cov:
                indiv_data = np.zeros((data.sizes['participant'], data.sizes['channels'], data.sizes['channels']))
                for i in range(data.sizes['participant']):
                    x_i = np.squeeze(data.data[i])
                    indiv_data[i] = np.mean(\
                        [np.cov(
                            x_i[trial,:,~np.isnan(x_i[trial,0,:])].T)\
                            for trial in range(x_i.shape[0]) 
                            if ~np.isnan(x_i[trial,0,:]).all()],axis=0)
                pca_ready_data = np.mean(np.array(indiv_data),axis=0)
            else:#assumes ERPs
                if averaged:
                    erps = []
                    for part in data.participant:
                        erps.append(data.sel(participant=part).groupby('samples').mean('epochs').T)
                    pca_ready_data = np.nanmean(erps,axis=0)
                else: 
                    pca_ready_data = data.stack({'all':['participant','epochs','samples']}).dropna('all')
                    pca_ready_data = pca_ready_data.transpose('all','channels')
            # Performing spatial PCA on the average var-cov matrix
            pca_weights = _pca(pca_ready_data, n_comp, data.coords["channels"].values)
            data = data @ pca_weights
            data.attrs['pca_weights'] = pca_weights
    elif method == 'mcca':
        ori_coords = data.drop_vars('channels').coords
        if n_ppcas is None:
            n_ppcas = n_comp*3
        mcca_m = mcca.MCCA(n_components_pca=n_ppcas, n_components_mcca=n_comp, r=mcca_reg)
        if cov:
            fitted_data = data.transpose('participant','epochs','samples','channels').data
            ccs = mcca_m.obtain_mcca_cov(fitted_data)
        else:
            if averaged:
                fitted_data = data.mean('epochs').transpose('participant','samples','channels').data
            else:
                fitted_data = data.stack({'all':['epochs','samples']})\
                .transpose('participant','all','channels').data
            ccs = mcca_m.obtain_mcca(fitted_data)
        trans_ccs = np.tile(np.nan, (data.sizes['participant'], data.sizes['epochs'], data.sizes['samples'], ccs.shape[-1]))
        for i, part in enumerate(data.participant):
                trans_ccs[i] = mcca_m.transform_trials(data.sel(participant=part).transpose('epochs','samples', 'channels').data.copy())
        data = xr.DataArray(trans_ccs,
             dims = ["participant","epochs","samples","component"],
             coords=dict(
                 participant=data.participant,
                 epochs=data.epochs,
                 samples=data.samples,
                 component = np.arange(n_comp))#n_comp
            )
        data = data.assign_coords(ori_coords)
        data.attrs['mcca_weights'] = mcca_m.mcca_weights
        data.attrs['pca_weights'] = mcca_m.pca_weights
    elif method is None:
        data = data.rename({'channels':'component'})
        data['component'] = np.arange(len(data.component))
        data.attrs['pca_weights'] = np.identity(len(data.component))
    else:
        raise ValueError(f"method {method} is unknown, choose either 'pca', 'mcca' or None")

    if apply_zscore:
        ori_coords = data.coords
        match apply_zscore:
            case 'all':
                if zscore_acrossPCs:
                    data = zscore_xarray(data)
                else:
                    data = data.stack(comp=['component']).groupby('comp', squeeze=False).map(zscore_xarray).unstack()
            case 'participant':
                if zscore_acrossPCs:
                    data = data.groupby('participant').map(zscore_xarray)
                else:
                    data = data.stack(participant_comp=[participants_variable,'component']).groupby('participant_comp', squeeze=False).map(zscore_xarray).unstack()
            case 'trial':
                if zscore_acrossPCs:
                    data = data.stack(trial=[participants_variable,'epochs']).groupby('trial').map(zscore_xarray).unstack()
                else:
                    data = data.stack(trial=[participants_variable,'epochs','component']).groupby('trial', squeeze=False).map(zscore_xarray).unstack()
        data = data.transpose('participant','epochs','samples','component')
        data = data.assign_coords(ori_coords)

    data.attrs['pca_weights'] = pca_weights
    data.attrs['sfreq'] = sfreq
    data = stack_data(data)
    return data
    


def save(data, filename):
    '''
    Save fit
    '''
    data.unstack().to_netcdf(filename)
    print(f"{filename} saved")

def load(filename):
    '''
    Load fit or data
    '''
    with xr.open_dataset(filename) as data:
        data.load()
    if 'trials' in data:
        data = data.stack(trial_x_participant=["participant","trials"]).dropna(dim="trial_x_participant", how='all')
    if 'eventprobs' in data and all(key in data for key in ['trial_x_participant','samples','event']) :
        # Ensures correct order of dimensions for later index use
        if 'iteration' in data:
            data['eventprobs'] = data.eventprobs.transpose('iteration','trial_x_participant','samples','event')
        else:
            data['eventprobs'] = data.eventprobs.transpose('trial_x_participant','samples','event')
    return data

def save_eventprobs(eventprobs, filename):
    '''
    Saves eventprobs to filename csv file
    '''
    eventprobs = eventprobs.unstack()
    eventprobs.to_dataframe().to_csv(filename)
    print(f"Saved at {filename}")

def centered_activity(data, times, channels, event, n_samples=None, center=True, cut_after_event=0, baseline=0, cut_before_event=0, event_width=0, impute=None):
    '''
    Parses the single trial signal of a given channel in a given number of samples before and after an event.

    Parameters
    ----------
    data : xr.Dataset
        HMP data (untransformed but with trial and participant stacked)
    times : xr.DataArray
        Onset times as computed using onset_times()
    channels : list
        channels to pick for the parsing of the signal, must be a list even if only one
    event : int 
        Which event is used to parse the signal 
    n_samples : int
        How many samples to record after the event (default = maximum duration between event and the consecutive event)
    cut_after_event: int
        Which event after ```event``` to cut samples off, if 1 (Default) cut at the next event
    baseline: int
        How much samples should be kept before the event
    cut_before_event: int
        At which previous event to cut samples from, ```baseline``` if 0 (Default), no effect if baseline = 0
    event_width: int
        Duration of the fitted events, used when cut_before_event is True

    Returns
    -------
    centered_data : xr.Dataset
        Xarray dataset with electrode value (data) and trial event time (time) and with trial_x_participant * samples dimension
    '''
    if event == 0:#no samples before stim onset
        baseline = 0
    elif event == 1:#no event at stim onset
        event_width = 0
    if cut_before_event == 0:#avoids searching before stim onset
        cut_before_event = event
    if n_samples is None:
        if cut_after_event is None:
            raise ValueError('One of ```n_samples``` or ```cut_after_event``` has to be filled to use an upper limit')
        n_samples = max(times.sel(event=event+cut_after_event).data- 
                                   times.sel(event=event).data)+1
    if impute is None:
        impute = np.nan
    if center:
        centered_data = np.tile(impute, (len(data.trial_x_participant), len(channels),
            int(round(n_samples-baseline+1))))
    else:
        centered_data = np.tile(impute, (len(data.trial_x_participant), len(channels),
            len(data.samples)))
        
    i = 0
    trial_times = np.zeros(len(data.trial_x_participant))*np.nan
    valid_indices = list(times.groupby('trial_x_participant', squeeze=False).groups.keys())
    for trial, trial_dat in data.groupby('trial_x_participant', squeeze=False):
        if trial in valid_indices:
            if cut_before_event>0:
                #Lower lim is baseline or the last sample of the previous event
                lower_lim = np.max([
                    -np.max([times.sel(event=event, trial_x_participant=trial)-
                            times.sel(event=event-cut_before_event, trial_x_participant=trial)-
                        event_width//2,0]), baseline])
            else:
                lower_lim = 0
            if cut_after_event>0:
                upper_lim = np.max([np.min([times.sel(event=event+cut_after_event, trial_x_participant=trial) - times.sel(event=event, trial_x_participant=trial)- event_width//2, \
                                            n_samples]), 0])
            else:
                upper_lim = n_samples
            
            # Determine samples in the signal to store
            start_idx = int(times.sel(event=event, trial_x_participant=trial) + lower_lim)
            end_idx = int(times.sel(event=event, trial_x_participant=trial) + upper_lim)
            trial_time = slice(start_idx, end_idx)
            trial_time_idx =  slice(start_idx, end_idx+1)
            trial_elec = trial_dat.sel(channels = channels, samples=trial_time).squeeze('trial_x_participant')
            # If center, adjust to always center on the same sample if lower_lim < baseline
            baseline_adjusted_start = int(abs(baseline - lower_lim))
            baseline_adjusted_end = baseline_adjusted_start + trial_elec.shape[-1]
            trial_time_arr = slice(baseline_adjusted_start, baseline_adjusted_end)

            if center:
                centered_data[i, :,  trial_time_arr] = trial_elec 
            else:
                centered_data[i, :, trial_time_idx] = trial_elec
            trial_times[i] = times.sel(event=event, trial_x_participant=trial)
            i += 1
    
    part, trial = data.coords['participant'].values, data.coords['epochs'].values
    trial_x_part = xr.Coordinates.from_pandas_multiindex(MultiIndex.from_arrays([part,trial],\
              names=('participant','trials')),'trial_x_participant')
    centered_data = xr.Dataset({'data': (('trial_x_participant','channel','samples'), centered_data),
                          'times': (('trial_x_participant'), trial_times)},
                            {'channel':channels,
                            'samples':np.arange(centered_data.shape[-1])+baseline},
                          
                        attrs = {
                                 'event':event})

    return centered_data.assign_coords(trial_x_part)
    
def condition_selection(hmp_data, condition_string, variable='event', method='equal'):
    '''
    condition_selection select a subset from hmp_data. It selects epochs for which
    'condition_string' is in 'variable' based on 'method'.

    Parameters
    ----------
    hmp_data : xr.Dataset
        transformed EEG data for hmp, from utils.transform_data
    condition_string : str | num
        condition indicator for selection
    variable : str
        variable present in hmp_data that is used for condition selection
    method : str
        'equal' selects equal trials, 'contains' selects trial in which conditions_string
        appears in variable

    Returns
    -------
    dat : xr.Dataset
        Subset of hmp_data.
        
    '''
    unstacked = hmp_data.unstack()
    unstacked[variable] = unstacked[variable].fillna("")
    if method == 'equal':
        unstacked = unstacked.where(unstacked[variable] == condition_string, drop=True)
        stacked = stack_data(unstacked)
    elif method == 'contains':
        unstacked = unstacked.where(unstacked[variable].str.contains(condition_string),drop=True)
        stacked = stack_data(unstacked)
    else:
        print('unknown method, returning original data')
        stacked = hmp_data
    return stacked

def condition_selection_epoch(epoch_data, condition_string, variable='event', method='equal'):

    if len(epoch_data.dims) == 4:
        stacked_epoch_data = epoch_data.stack(trial_x_participant=('participant','epochs')).dropna('trial_x_participant',how='all')

    if method == 'equal':
        stacked_epoch_data = stacked_epoch_data.where(stacked_epoch_data[variable] == condition_string, drop=True)
    elif method == 'contains':
        stacked_epoch_data = stacked_epoch_data.where(stacked_epoch_data[variable].str.contains(condition_string),drop=True)
    return stacked_epoch_data.unstack()
    
def participant_selection(hmp_data, participant):
    unstacked = hmp_data.unstack().sel(participant = participant)
    stacked = stack_data(unstacked)
    return stacked

def filter_non_converged(estimates):
    for iteration in estimates.iteration.values:
        if np.diff(estimates.sel(iteration=iteration).traces.dropna('em_iteration')[-2:]) < -1e-10:
            estimates = estimates.drop_sel({'iteration':iteration})
    estimates["iteration"] = range(len(estimates.iteration))
    return estimates


def epoch_between_events(raw, events, event_id_from, event_id_to, baseline=None, picks=None, reject=None, tmin=0, tmax=0, decim=1, reject_by_annotation=True, proj='delayed', metadata=None, resample_freq = None, verbose=None):
    '''
    Epoch data between a 'from event' and a 'to event', typically between stimulus and response.
    Empty samples due to different epoch lengths are filled with nans. There must be the same 
    number of from and to events.

    NOTE: reject and resample of resulting Epochs object do not work properly!

    Parameters
    ----------
    raw : mne.io.Raw object
        An instance of Raw.
    events : array of int, shape (n_events, 3)
        The array of events. The first column contains the event time in samples, with first_samp included. 
        The third column contains the event id.
    event_id_from : dict
        The id of the 'from events' to consider. The keys can later be used to access associated events. Example: dict(auditory=1, visual=3). 
    event_id_to : dict
        The id of the 'to events' to consider. The keys can later be used to access associated events. Example: dict(correct=11, incorrect=13).
    baseline : None | tuple of length 2
        The time interval to consider as “baseline” with respect to the 'from events' when applying baseline correction. If None, do not apply baseline correction. If a tuple (a, b), the interval is between a and b (in seconds), including the endpoints. If a is None, the beginning of the data is used; and if b is None, it is set to the end of the interval. If (None, None), the entire time interval is used.
    picks : str | array_like | slice | None
        Channels to include. Slices and lists of integers will be interpreted as channel indices. In lists, channel type strings (e.g., ['meg', 'eeg']) will pick channels of those types, channel name strings (e.g., ['MEG0111', 'MEG2623'] will pick the given channels. Can also be the string values “all” to pick all channels, or “data” to pick data channels. None (default) will pick all channels. Note that channels in info['bads'] will be included if their names or indices are explicitly provided.
    reject : dict | None
        Reject epochs based on maximum peak-to-peak signal amplitude (PTP), i.e. the absolute difference between the lowest and the highest signal value. In each individual epoch, the PTP is calculated for every channel. If the PTP of any one channel exceeds the rejection threshold, the respective epoch will be dropped. The dictionary keys correspond to the different channel types; valid keys can be any channel type present in the object. Example:
        reject = dict(grad=4000e-13,  # unit: T / m (gradiometers)
              mag=4e-12,      # unit: T (magnetometers)
              eeg=40e-6,      # unit: V (EEG channels)
              eog=250e-6      # unit: V (EOG channels)
              )
    tmin, tmax : float
        Start and end time of the epochs in seconds, relative to respectively the from and the end events. The closest or matching samples corresponding to the start and end time are included. Defaults to 0.
    decim : int
        Factor by which to subsample the data.
    reject_by_annotation : bool
        Whether to reject based on annotations. If True (default), epochs overlapping with segments whose description begins with 'bad' are rejected. If False, no rejection based on annotations is performed.
    proj : bool | ‘delayed’
        Apply SSP projection vectors. If proj is ‘delayed’ and reject is not None the single epochs will be projected before the rejection decision, but used in unprojected state if they are kept. This way deciding which projection vectors are good can be postponed to the evoked stage without resulting in lower epoch counts and without producing results different from early SSP application given comparable parameters. Note that in this case baselining, detrending and temporal decimation will be postponed. If proj is False no projections will be applied which is the recommended value if SSPs are not used for cleaning the data.    
    metadata : instance of pandas.DataFrame | None
        A pandas.DataFrame specifying metadata about each epoch. If given, len(metadata) must equal len(events). The DataFrame may only contain values of type (str | int | float | bool). If metadata is given, then pandas-style queries may be used to select subsets of data, see mne.Epochs.__getitem__(). When a subset of the epochs is created in this (or any other supported) manner, the metadata object is subsetted accordingly, and the row indices will be modified to match epochs.selection.
    resample_freq: float | None
        Resample data to resample_freq.
     verbose : bool | str | int | None
        Control verbosity of the logging output. If None, use the default verbosity level. See the logging documentation and mne.verbose() for details. Should only be passed as a keyword argument.   

    Returns
    -------
    epochs : mne.Epochs
        mne.Epochs object with epoched data, missing samples are np.nan
    '''

    #set params for raw.get_data()
    reject_by_annotation = 'NaN' if reject_by_annotation else None

    #get data and fill matrix
    from_id = list(event_id_from.values())
    to_id = list(event_id_to.values())

    n_epochs = np.count_nonzero(np.isin(events[:,2],from_id))
    n_channels = len(raw.get_channel_types(picks=picks))

    epochs_start_stop = np.zeros((n_epochs,4), dtype='int64') #start and stop samples, length, and event from id 

    #find start ids in events and following to id, store
    epochs_cnt = 0
    for i, ev in enumerate(events):
        if ev[2] in from_id:
            end_ev = i
            while events[end_ev,2] not in to_id:
                end_ev += 1
                if end_ev == len(events):
                    raise ValueError(
                        "Every from_id must have a following to_id"
                    )
            epochs_start_stop[epochs_cnt] = [ev[0], events[end_ev,0], 0, ev[2]]
            epochs_cnt += 1

    #apply tmin, tmax; calc len
    epochs_start_stop[:,0] = epochs_start_stop[:,0] + round(tmin * raw.info["sfreq"])
    epochs_start_stop[:,1] = epochs_start_stop[:,1] + round(tmax * raw.info["sfreq"])
    epochs_start_stop[:,2] = epochs_start_stop[:,1] - epochs_start_stop[:,0]
    if any(epochs_start_stop[:,2] < 0):
        raise ValueError("Negative epoch length due to tmin/tmax application.")
    
    #make data matrix, epochs x channels x timepoint (e.g. 228, 25, 500)
    nr_samples = max(epochs_start_stop[:,2])
    dat = np.empty((n_epochs, n_channels, nr_samples))
    dat[:] = np.nan
    drop_log = [()] * n_epochs

    #fill matrix
    for i, ep in enumerate(epochs_start_stop):

        #get data from raw, taking annotations into account
        dat[i,:,:ep[2]] = raw.get_data(picks=picks, start = ep[0], stop = ep[1], reject_by_annotation=reject_by_annotation, verbose=verbose)

        #check reject:
        #missing data due to annotations:
        if np.isnan(dat[i,:,:ep[2]]).any():
            drop_log[i] = drop_log[i] + ('BAD_artifact',)

        #reject based on range
        if reject != None:
            if not np.isnan(dat[i,:,:ep[2]]).all():
                for k,v in reject.items():
                    ch_types = raw.get_channel_types(picks=picks)
                    for ch in np.where(np.array(ch_types) == k)[0]:
                        if np.nanmax((dat[i,ch,:ep[2]])) - np.nanmin((dat[i,ch,:ep[2]])) > v:
                            drop_log[i] = drop_log[i] + (raw.ch_names[ch],)
    
    #remove drops  
    dat = np.delete(dat, [True if tmp != () else False for tmp in drop_log], 0)
    events = mne.pick_events(events, include=from_id)
    events_drop = np.delete(events, [True if tmp != () else False for tmp in drop_log], 0)

    #resample
    #based on Epochs.resample, but taking care of NaNs
    if resample_freq != None:
        
        sfreq = float(resample_freq)
        o_sfreq = raw.info["sfreq"]

        nr_samples = int(round(dat.shape[2] * sfreq/o_sfreq)) #perhaps + 1 to make sure
        dat_resampled = np.empty((dat.shape[0], dat.shape[1], nr_samples))
        dat_resampled[:] = np.nan

        #resample by epoch because NaNs need to be removed
        for i, ep in enumerate(dat):
            #filter non-nan samples
            ep_resamp = mne.filter.resample(ep[:, ~np.isnan(ep[0,:])], sfreq, o_sfreq, npad="auto", window="boxcar", pad="edge")
            dat_resampled[i,:,:ep_resamp.shape[1]] = ep_resamp

        #check lowpass
        lowpass = raw.info.get("lowpass")
        lowpass = np.inf if lowpass is None else lowpass
        with raw.info._unlock():
            raw.info["lowpass"] = min(lowpass, sfreq / 2.0)
            raw.info["sfreq"] = float(sfreq)
        
        dat = dat_resampled

    #create epochs object
    # NOTE using reject here does not work, it only rejects (if necessary) the first episode.
    # This seems due to the NaNs.
    epochs = mne.EpochsArray(dat, raw.info, events = events_drop, tmin=tmin, event_id=event_id_from, baseline=baseline, proj=proj, metadata=metadata, drop_log=tuple(drop_log), verbose=verbose)

    #decimate if asked for
    if decim > 1:
        epochs = epochs.decimate(decim, offset=0, verbose=verbose)

    return epochs
