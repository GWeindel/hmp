'''

'''

import numpy as np
from scipy.special import gamma as gamma_func
from scipy.stats import lognorm
import xarray as xr
import multiprocessing as mp
import itertools
import pandas as pd
import warnings
from warnings import warn, filterwarnings
from seaborn.algorithms import bootstrap
import json
import mne
import os

filterwarnings('ignore', 'Degrees of freedom <= 0 for slice.', )#weird warning, likely due to nan in xarray, not important but better fix it later 
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

def maxb_scale_to_mean(scale, shape):
    return 2*scale*np.sqrt(2/np.pi)
def maxb_mean_to_scale(mean, shape): 
    return mean/2/np.sqrt(2/np.pi)

def ray_scale_to_mean(scale, shape):
    return scale*np.sqrt(np.pi/2)
def ray_mean_to_scale(mean, shape): 
    return mean/np.sqrt(np.pi/2)

def halfn_scale_to_mean(scale, shape):
    return (scale*np.sqrt(2))/np.sqrt(np.pi)
def halfn_mean_to_scale(mean, shape): 
    return mean/np.sqrt(2/np.pi)

def fisk_scale_to_mean(scale, shape):
    return  (scale*np.pi/shape)/np.sin(np.pi/shape)
def fisk_mean_to_scale(mean, shape): 
    return  shape*(mean*np.sin(np.pi/shape))/np.pi

def uniform_scale_to_mean(scale, shape): 
    return (scale-shape)/2
def uniform_mean_to_scale(mean, shape): 
    return 2*(mean-shape)

def read_mne_data(pfiles, event_id=None, resp_id=None, epoched=False, sfreq=None, 
                 subj_idx=None, metadata=None, events_provided=None, rt_col='rt', rts=None,
                 verbose=True, tmin=-.2, tmax=5, offset_after_resp = 0, 
                 high_pass=None, low_pass = None, pick_channels = 'eeg', baseline=(None, 0),
                 upper_limit_RT=np.inf, lower_limit_RT=0, reject_threshold=None, scale=1, reference=None):
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
        Rejection threshold to apply when creating epochs, expressed in microvolt
    scale: float
        Scale to apply to the RT data (e.g. 1000 if ms)
    reference:
        What reference to use (see MNE documentation), if None, keep the existing one
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
                    verbose=verbose, detrend=1, on_missing = 'warn', event_repeated='drop',
                    metadata=metadata_i, reject_by_annotation=True, reject=reject_threshold)
            epochs.metadata.rename({'response':'rt'}, axis=1, inplace=True)
            metadata_i = epochs.metadata
        else:
            if '.fif' in participant:
                epochs = mne.read_epochs(participant, preload=True, verbose=verbose)
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
    epoch_data = epoch_data.assign_attrs(lowpass=epochs.info['lowpass'], highpass=epochs.info['highpass'],
                                         lower_limit_RT=lower_limit_RT,  upper_limit_RT=upper_limit_RT, 

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

def _center(data):
    '''
    zscore of the data
    '''
    return data - data.mean()

def zscore(data):
    '''
    zscore of the data
    '''
    return (data - data.mean()) / data.std()

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

def compute_ci(times):
    '''
    Compute confidence intervals
    '''
    return np.abs(np.squeeze([np.nanpercentile(bootstrap(times), q=[2.5,97.5])]) - np.mean(times))


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

def transform_data(data, participants_variable="participant", apply_standard=True,  apply_zscore='participant', method='pca', centering=False, n_comp=None, pca_weights=None, filter=None):
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
        Whether to apply standardization
    apply_zscore : str 
        Whether to apply z-scoring and on what data, either None, 'all', 'participant', 'trial', for zscoring across all data, by participant, or by trial, respectively. If set to true, evaluates to 'trial' for backward compatibility.
    method : str
        Method to apply, for now limited to 'pca'
    n_comp : int
        How many components to select from the PC space, if None plots the scree plot and a prompt requires user
        to specify how many PCs should be retained
    pca_weigths : xarray
        Weights of a PCA to apply to the data (e.g. in the resample function)
    filter: None | (lfreq, hfreq) 
        If none, no filtering is appliedn. If tuple, data is filtered between lfreq-hfreq.
        NOTE: filtering at this step is suboptimal, filter before epoching if at all possible, see
              also https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html

    Returns
    -------
    data : xarray.Dataset
        xarray dataset [n_samples * n_comp] data expressed in the PC space, ready for HMP fit
    pca_weigths : xarray.Dataset
        loadings of the PCA, used to retrieve channel space
    pca.explained_variance_ : ndarray
        explained variance for each component
    means : xarray.DataArray
        means of the channels before PCA/zscore
    '''
    if isinstance(data, xr.DataArray):
        raise ValueError('Expected a xarray Dataset with data and event as DataArrays, check the data format')
    if not apply_zscore in ['all', 'participant', 'trial'] and not isinstance(apply_zscore,bool):
        raise ValueError('apply_zscore should be either a boolean or one of [\'all\', \'participant\', \'trial\']')
    assert np.sum(np.isnan(data.groupby('participant', squeeze=False).mean(['epochs','samples']).data.values)) == 0,\
        'at least one participant has an empty channel'
    sfreq = data.sfreq

    if filter:
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

    if apply_zscore == True:
        apply_zscore = 'trial' #defaults to trial
    if apply_standard:
        if 'participant' not in data.dims or len(data.participant) == 1:
            warn('Requested standardization of between participant variance yet no participant dimension is found in the data or only one participant is present. No standardization is done, set apply_standard to False to avoid this warning.')
        else:
            mean_std = data.groupby(participants_variable, squeeze=False).std(dim=...).data.mean()
            data = data.assign(mean_std=mean_std.data)
            data = data.groupby(participants_variable, squeeze=False).map(standardize)


    if method == 'pca':
        if isinstance(data, xr.Dataset):
            data = data.data
        if pca_weights is None:
            from sklearn.decomposition import PCA
            var_cov_matrices = []
            for i,trial_dat in data.stack(trial=("participant", "epochs")).drop_duplicates('trial').groupby('trial', squeeze=False):
                var_cov_matrices.append(vcov_mat(trial_dat)) #Would be nice not to have a for loop but groupby.map seem to fal
            var_cov_matrix = np.mean(var_cov_matrices,axis=0)
            # Performing spatial PCA on the average var-cov matrix
            if n_comp == None:
                import matplotlib.pyplot as plt
                n_comp = np.shape(var_cov_matrix)[0]-1
                fig, ax = plt.subplots(1,2, figsize=(.2*n_comp, 4))
                pca = PCA(n_components=n_comp, svd_solver='full')#selecting Principale components (PC)
                pca.fit(var_cov_matrix)

                ax[0].plot(np.arange(pca.n_components)+1, pca.explained_variance_ratio_,'.-')
                ax[0].set_ylabel('Normalized explained variance')
                ax[0].set_xlabel('Component')
                ax[1].plot(np.arange(pca.n_components)+1, np.cumsum(pca.explained_variance_ratio_),'.-')
                ax[1].set_ylabel('Cumulative normalized explained variance')
                ax[1].set_xlabel('Component')
                plt.tight_layout()
                plt.show()
                n_comp = int(input(f'How many PCs (90 and 99% explained variance at component n{np.where(np.cumsum(pca.explained_variance_ratio_) >= .90)[0][0]+1} and n{np.where(np.cumsum(pca.explained_variance_ratio_) >= .99)[0][0]+1})?'))
            pca = PCA(n_components=n_comp, svd_solver='full')#selecting Principale components (PC)
            pca.fit(var_cov_matrix)
            #Rebuilding pca PCs as xarray to ease computation
            coords = dict(channels=("channels", data.coords["channels"].values),
                         component=("component", np.arange(n_comp)))
            pca_weights = xr.DataArray(pca.components_.T, dims=("channels","component"), coords=coords)
        data = data @ pca_weights
    elif method is None:
        data = data.rename({'channels':'component'})
        data['component'] = np.arange(len(data.component))
        pca_weigths = np.identity(len(data.component))
    # zscore either across all data, by participant (preferred), or by trial
    if centering:
        ori_coords = data.coords
        data = data.stack(trial=[participants_variable,'epochs','component']).groupby('trial', squeeze=False).map(_center).unstack()
        data = data.transpose('participant','epochs','samples','component')
        data = data.assign_coords(ori_coords)
    if apply_zscore:
        ori_coords = data.coords
        match apply_zscore:
            case 'all':
                data = data.stack(comp=['component']).groupby('comp', squeeze=False).map(zscore_xarray).unstack()
            case 'participant':
                data = data.stack(participant_comp=[participants_variable,'component']).groupby('participant_comp', squeeze=False).map(zscore_xarray).unstack()
            case 'trial':
                data = data.stack(trial=[participants_variable,'epochs','component']).groupby('trial', squeeze=False).map(zscore_xarray).unstack()
        data = data.transpose('participant','epochs','samples','component')
        data = data.assign_coords(ori_coords)
    data.attrs['pca_weights'] = pca_weights
    data.attrs['sfreq'] = sfreq
    data = stack_data(data)
    return data
    

def loocv_calcs(data, init, participant, initial_fit, cpus=None, verbose=False):
    '''
    Fits model based on init settings and initial_fit parameters to data of 
    n - 1 (participant) participants, and calculates likelihood on the left-out
    participant.
        
    Parameters
    ----------
    data : xarray.Dataset
        xarray data from transform_data() 
    init : hmp object 
        original hmp object used for the fit, all settings are copied to the left out models
    participant : str
        name of the participant to leave out and estimate likelihood on
    initial_fit : xarray.Dataset
        Fit of the model with the same number of events and all participants
    cpus : int
        Number of cpus to use to fit the models. 
    verbose : bool

    Returns
    -------
    likelihood : float
        likelihood computed for the left-out participant
    '''

    from hmp.models import hmp

    if verbose:
            print(f'\t\tCalculating fit for participant {participant}')
    if cpus is None:
        cpus = init.cpus

    participants_idx = data.participant.values

    #Extracting data with and without left out participant
    data_without_pp = stack_data(data.sel(participant = participants_idx[participants_idx != participant], drop=False))
    data_pp = stack_data(data.sel(participant=participant, drop=False))

    #Building models 
    model_without_pp = hmp(data_without_pp, sfreq=init.sfreq, event_width=init.event_width, cpus=cpus, shape=init.shape, template=init.template, location=init.location, distribution=init.distribution, em_method=init.em_method)
    model_pp = hmp(data_pp, sfreq=init.sfreq, event_width=init.event_width, cpus=cpus, shape=init.shape, template=init.template, location=init.location, distribution=init.distribution, em_method=init.em_method)

    #fit the HMP using previously estimated parameters as initial parameters, and estimate likelihood
    if 'condition' in initial_fit.dims:
        #fit model
        fit_without_pp = model_without_pp.fit_single_conds(initial_fit.magnitudes.values, initial_fit.parameters.values, mags_map=initial_fit.mags_map, pars_map=initial_fit.pars_map, conds=initial_fit.conds_dict, verbose=False)
        #calc lkh
        conds_pp = initial_fit.sel(participant=participant)['cond'].values
        likelihood = model_pp.estim_probs_conds(fit_without_pp.magnitudes.values, fit_without_pp.parameters.values, initial_fit.mags_map, initial_fit.pars_map, conds_pp, lkh_only=True)
    else:
        #fit model
        n_eve = np.max(initial_fit.event.values)+1
        fit_without_pp = model_without_pp.fit_single(n_eve, initial_fit.magnitudes.dropna('event').values, initial_fit.parameters.dropna('stage').values, verbose=False)
        #calc lkh
        likelihood = model_pp.estim_probs(fit_without_pp.magnitudes.dropna('event').values, fit_without_pp.parameters.dropna('stage').values, n_eve, None, True)

    return likelihood


def loocv(init, data, estimate, cpus=1, verbose=True, print_warning=True):
    '''
    Performs leave-one-out cross validation. For provided estimate(s), it will perform loocv by 
    leaving out one participant, estimating a fit, and computing the likelihood of the data from 
    the left out participant with the estimated parameters. This is repeated for all participants.

    Initial parameters for the models are based on estimate(s), hmp model settings on init.

    Estimate(s) can be provides as:
    - a single model estimate (from fit_single(..))
    - a set of fits from backward estimation (from backward_estimation(..))
    - a model fit with different conditions (from fit_single_conds(...))
    - a list of one or more of the above
    Note that all need to share the same data and participants.
    
    IMPORTANT:  This loocv procedure is incorrect in the sense that an initial estimate is used
                to inform both the fit of the left-out participant and the other participants.
                This means that they are not fully independent, unless the initial estimate is
                based on the literature or another task. However, it does give a very good initial
                idea of the correct loocv procedure, and is relatively quick.

                To do this correctly, use loocv_backward, loocv_fit, or the general loocv_fun, 
                which all three also calculate the initial estimate for every fold by applying
                backward estimation, the fit function, or your own function, respectively.

    Parameters
    ----------
    init : hmp model
        initialized hmp model
    data : xarray.Dataset
        xarray data from transform_data() 
    estimate : hmp model estimate or list of estimates
        See above.
    cpus : int
        Nr of cpus to use. If 1, LOOCV is performed on a single CPU. Otherwise
        on the provided int or setting in init.
        We recommend using 1 CPU at this level on a laptop or normal PC. Only use multiple
        CPUs if you have *a lot* of memory available.
    verbose : bool
    print_warning : bool
        whether to plot the loocv 'incorrectness' warning
        
    Returns
    -------
    list of likelihood objects for provided model estimates
    '''

    if verbose:
        if print_warning:
            print()
            print("IMPORTANT:  This loocv procedure is incorrect in the sense that an initial estimate")
            print("is used to inform both the fit of the left-out participant and the other participants.")
            print("This means that they are not fully independent, unless the initial estimate is")
            print("based on the literature or another task. However, it does give a very good initial")
            print("idea of the correct loocv procedure and is relatively quick.")

            print("\nTo do loocv correctly, use loocv_backward or the general loocv_func,")
            print("which calculate the initial estimate for every fold by applying")
            print("backward estimation or your own function, respectively.")
            print()

    if cpus is None:
        cpus = init.cpus

    if cpus != 1:
        print('We recommend using cpus==1 unless you have *a lot* of memory and cpus available.')

    data = data.unstack()
    participants_idx = data.participant.values

    if not isinstance(estimate, list):
        models = [estimate]
    else:
        models = estimate

    n_models = len(models)
    if verbose:
        print(f'LOOCV started for {n_models} model(s)')

    #no mp here, but at participant level
    likelihoods = []
    for model in models:

        #option 1 and 2: single model and single model with conditions
        if not 'n_events' in model.dims:
            if verbose:
                if 'condition' in model.dims:
                    print(f'\tLOOCV for condition-based model with {np.max(model.event).values+1} event(s)')
                else:
                    print(f'\tLOOCV for single model with {np.max(model.event).values+1} event(s)')

            loocv = []
            if cpus == 1: #not mp            
                for participant in participants_idx:
                    loocv.append(loocv_calcs(data, init, participant, model, verbose=verbose))
            else: #mp
                with mp.Pool(processes=cpus) as pool:
                    loocv = pool.starmap(loocv_calcs,
                                        zip(itertools.repeat(data), itertools.repeat(init),participants_idx,
                                            itertools.repeat(model),itertools.repeat(1),itertools.repeat(verbose)))
            
            likelihoods.append(xr.DataArray(np.array(loocv).astype(np.float64), dims='participant',
                    coords = {"participant": participants_idx}, 
                    name = "loo_likelihood"))
           
        #option 3: backward
        if 'n_events' in model.dims:
            if verbose:
                print(f'\tLOOCV for backward estimation models with {model.n_events.values} event(s)')
            
            loocv_back = []
            for n_eve in model.n_events.values:
                if verbose:
                    print(f'\t  Estimating backward estimation model with {n_eve} event(s)')
                loocv = []
                if cpus == 1: #not mp            
                    for participant in participants_idx:
                        loocv.append(loocv_calcs(data, init, participant, model.sel(n_events=n_eve).dropna('event'), verbose=verbose))
                else: #mp
                    with mp.Pool(processes=cpus) as pool:
                        loocv = pool.starmap(loocv_calcs,
                                            zip(itertools.repeat(data), itertools.repeat(init),participants_idx,
                                                itertools.repeat(model.sel(n_events=n_eve).dropna('event')),itertools.repeat(1),itertools.repeat(verbose)))

                loocv_back.append(xr.DataArray(np.expand_dims(np.array(loocv).astype(np.float64),axis=0), 
                                        dims=('n_event', 'participant'),
                                        coords = {"n_event": np.array([n_eve]),
                                                   "participant": participants_idx}, 
                                        name = "loo_likelihood"))
                
            likelihoods.append(xr.concat(loocv_back, dim = 'n_event'))
            
    if n_models == 1:
       likelihoods = likelihoods[0]

    return likelihoods


def loocv_mp(init, stacked_data, bests, func=loocv_calcs, cpus=2, verbose=True):
    '''
    Deprecated, use loocv instead.
    '''
    warn('This method is deprecated, use loocv() instead', DeprecationWarning, stacklevel=2) 
    
    return loocv(init, stacked_data, bests, cpus=cpus, verbose=verbose)


def example_fit_single_func(hmp_model, n_events, magnitudes=None, parameters=None, verbose=False):
    '''
    Example of simple function that can be used with loocv_func.
    This fits a model with n_events and potentially provided mags and params.

    Can be called, for example, as :
        loocv_func(hmp_model, hmp_data, example_fit_single_func, func_args=[2])
    '''
    return hmp_model.fit_single(n_events, magnitudes=magnitudes, parameters=parameters, verbose=verbose) 

def example_complex_fit_func(hmp_model, max_events=None, n_events=1, mags_map=None, pars_map=None, conds=None, verbose=False):
    '''
    Example of a complex function that can be used with loocv_func.
    This function first performs backwards estimation up to max_events,
    and follows this with a condition-based model of n_events, informed
    by the selected backward model and the provided maps. It returns
    both models, so for both the likelihood will be estimated.

    Can be called, for example, as :

        pars_map = np.array([[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 2, 0],
                     [0, 0, 0, 0, 3, 0],
                     [0, 0, 0, 0, 4, 0]])
        conds = {'rep': np.arange(5)+1}
        loocv_func(hmp_model, hmp_data, example_complex_fit_func, func_args=[7, 5, None, pars_map,conds])
    '''
   
    #fit backward model up to max_events
    backward_model = hmp_model.backward_estimation(max_events)

    #select n_events model
    n_event_model = backward_model.sel(n_events=n_events).dropna('event')
    mags = n_event_model.magnitudes.dropna('event').data
    pars = n_event_model.parameters.dropna('stage').data

    #fit condition model
    cond_model = hmp_model.fit_single_conds(magnitudes=mags, parameters=pars, mags_map=mags_map, pars_map=pars_map, conds=conds, verbose=verbose)
    
    return [backward_model, cond_model]


def loocv_estimate_func(data, init, participant, func_estimate, func_args=None, cpus=None, verbose=False):
    '''
    Applies func_estimate with func_args to data of n - 1 (participant) participants.
    func_estimate should return an estimated hmp model; either a single model, 
    a condition model, or a backward estimation model. This model is then used 
    to calculate the fit on the left out participant with loocv_likelihood.

    Parameters
    ----------
    data : xarray.Dataset
        xarray data from transform_data() 
    init : hmp object 
        original hmp object used for the fit, all settings are copied to the left out models
    participant : str
        name of the participant to leave out
    func_estimate : function that returns a hmp model estimate
        this can be backward_estimation, fit, or your own function.
        It should take an initialized hmp model as its first argument,
        other arguments are passed on from func_args.
        See also loocv_func(..)
    func_args : list
        List of arguments that need to be passed on to func_estimate.
        See also loocv_func(..)
    cpus : int
        number of cpus to use
    verbose : bool

    Returns
    -------
    hmp model
        estimated hmp_model with func_estimate on n-1 participants
    '''

    from hmp.models import hmp

    if verbose:
            print(f'\tEstimating model for all participants except {participant}')
    if cpus is None:
        cpus = init.cpus

    participants_idx = data.participant.values

    #Extract data without left out participant
    data_without_pp = stack_data(data.sel(participant = participants_idx[participants_idx != participant], drop=False))

    #Building model
    model_without_pp = hmp(data_without_pp, sfreq=init.sfreq, event_width=init.event_width, cpus=cpus, shape=init.shape, template=init.template, location=init.location, distribution=init.distribution, em_method=init.em_method)

    #Apply function and return
    estimates = func_estimate(model_without_pp, *func_args)
    if isinstance(estimates,list):
        for i in range(len(estimates)):
            estimates[i] = estimates[i].drop_vars(['eventprobs'])
    else:
        estimates = estimates.drop_vars(['eventprobs'])

    return estimates


def loocv_likelihood(data, init, participant, estimate, cpus=None, verbose=False):
    '''
    Calculate likelihood of fit on participant participant using parameters from estimate,
    either using single model or condition based model.
        
    Parameters
    ----------
    data : xarray.Dataset
        xarray data from transform_data() 
    init : hmp object 
        original hmp object used for the fit, all settings are copied to the left out models
    participant : str
        name of the participant to leave out and estimate likelihood on
    estimate : xarray.Dataset
        estimate that has parameters to apply.
    cpus : int
        Number of cpus to use to fit the models. 
    verbose : bool

    Returns
    -------
    likelihood : float
        likelihood computed for the left-out participant
    '''

    from hmp.models import hmp

    if verbose:
            print(f'\tCalculating likelihood for participant {participant}')
    if cpus is None:
        cpus = init.cpus

    #Extracting data of left out participant
    data_pp = stack_data(data.sel(participant=participant, drop=False))

    #Building model 
    model_pp = hmp(data_pp, sfreq=init.sfreq, event_width=init.event_width, cpus=cpus, shape=init.shape, template=init.template, location=init.location, distribution=init.distribution, em_method=init.em_method)

    #estimate likelihood with previously estimated parameters
    if 'condition' in estimate.dims:

        from itertools import product    

        #create conds for this participant based on estimate.conds_dict and model_pp 
        #description of condition for this participant, which is not available
        conds = estimate.conds_dict
        cond_names = []
        cond_levels = []
        cond_trials = []
        for cond in conds:
            cond_names.append(list(cond.keys())[0])
            cond_levels.append(cond[cond_names[-1]])
            cond_trials.append(model_pp.trial_coords[cond_names[-1]].data.copy())

        cond_levels = list(product(*cond_levels))
        cond_levels = np.array(cond_levels, dtype=object) #otherwise comparison below can fail

        #build condition array with digit indicating the combined levels
        cond_trials = np.vstack(cond_trials).T
        conds = np.zeros((cond_trials.shape[0])) * np.nan
        for i, level in enumerate(cond_levels):
            conds[np.where((cond_trials == level).all(axis=1))] = i
        conds=np.int8(conds)

        likelihood = model_pp.estim_probs_conds(estimate.magnitudes.values, estimate.parameters.values, estimate.mags_map, estimate.pars_map, conds, lkh_only=True)
    else:
        n_eve = np.max(estimate.event.dropna('event').values)+1
        likelihood = model_pp.estim_probs(estimate.magnitudes.dropna('event').values, estimate.parameters.dropna('stage').values, n_eve, None, True)

    return likelihood
        

def loocv_func(init, data, func_estimate, func_args=None, cpus=1, verbose=True):
    '''
    Performs leave-one-out cross validation using func_estimate to calculate the initial fit.
    It will perform loocv by leaving out one participant, applying 'func_estimate' to the 
    data to estimate a fit, and computing the likelihood of the data from the left out
    participant with the estimated parameters. This is repeated for all participants. Hmp 
    model settings are based on init.

    func_estimate is also allowed to return a list of estimates; for all provided estimates
    the likelihood of the left out participant will be calculated.

    For example of func_estimate, see these function above:
    example_fit_single_func(..)
    example_complex_single_func(..)

    They can be called, for example, as
        loocv_func(hmp_model, hmp_data, example_fit_single_func, func_args=[1])
    
    Note that func_args is not named, so all arguments up to the one you want to use
    of func_estimate need to be provided.
        
    Parameters
    ----------
    init : hmp model
        initialized hmp model
    data : xarray.Dataset
        xarray data from transform_data() 
    func_estimate : function that returns an hmp model estimate or a list
        of hmp model estimates. These can be the results of backward_estimation,
        fit_single, fit_single_conds, or your own function.
        It should take an initialized hmp model as its first argument,
        other arguments are passed on from func_args.
    func_args : list
        List of arguments that need to be passed on to func_estimate.
    cpus : int
        Nr of cpus to use. If 1, LOOCV is performed on a single CPU. Otherwise
        on the provided int or setting in init.
        We recommend using 1 CPU at this level on a laptop or normal PC. Only use multiple
        CPUs if you have *a lot* of memory available.
    verbose : bool
        
    Returns
    -------
    likelihood object containing likelihoods on left out participant
    estimates : list of all models without the left out participant
    '''

    if cpus is None:
        cpus = init.cpus

    if cpus != 1:
        print('We recommend using cpus==1 unless you have *a lot* of memory and cpus available.')

    data = data.unstack()
    participants_idx = data.participant.values

    #first get estimates on n-1 subjects for all folds
    if verbose:
        print(f'Calculating estimates with func {func_estimate} and args {func_args}.')

    estimates = []
    if cpus == 1: #not mp            
        for participant in participants_idx:
            estimates.append(loocv_estimate_func(data, init, participant, func_estimate, func_args=func_args, verbose=verbose))
    else: #mp
        with mp.Pool(processes=cpus) as pool:
            estimates = pool.starmap(loocv_estimate_func,
                        zip(itertools.repeat(data), itertools.repeat(init),participants_idx,
                            itertools.repeat(func_estimate),itertools.repeat(func_args),
                            itertools.repeat(1), itertools.repeat(verbose)))

    #if multiple estimates are repeated per subject, rearrange data
    if isinstance(estimates[0], list):
        all_estimates = []
        for est_idx in range(len(estimates[0])):
            all_estimates.append([estimate[est_idx] for estimate in estimates])
    else: #only one model estimate given
        all_estimates = [estimates]
    
    #second, calculate likelihood of left out subject for all folds
    print()

    all_likelihoods = []
    for estimates in all_estimates: 

        #option 1 and 2: single model and single model with conditions
        if not 'n_events' in estimates[0].dims:
            if verbose:
                if 'condition' in estimates[0].dims:
                    print(f'Calculating likelihood for condition-based model with {np.max(estimates[0].event).values+1} event(s)')
                else:
                    print(f'Calculating likelihood for single model with {np.max(estimates[0].event).values+1} event(s)')

            loocv = []
            if cpus == 1: #not mp            
                for pidx, participant in enumerate(participants_idx):
                    loocv.append(loocv_likelihood(data, init, participant, estimates[pidx], verbose=verbose))
            else: #mp
                with mp.Pool(processes=cpus) as pool:
                    loocv = pool.starmap(loocv_likelihood,
                                        zip(itertools.repeat(data), itertools.repeat(init),participants_idx,
                                            estimates, itertools.repeat(1),itertools.repeat(verbose)))
            
            likelihoods = xr.DataArray(np.array(loocv).astype(np.float64), dims='participant',
                    coords = {"participant": participants_idx}, 
                    name = "loo_likelihood")

        #option 3: backward
        if 'n_events' in estimates[0].dims:
            if verbose:
                print(f'Calculating likelihood for backward estimation models with {estimates[0].n_events.values} event(s)')
            
            loocv_back = []
            for n_eve in estimates[0].n_events.values:
                if verbose:
                    print(f'  Calculating likelihood for backward estimation model with {n_eve} event(s)')
                loocv = []
                if cpus == 1: #not mp            
                    for pidx, participant in enumerate(participants_idx):
                        loocv.append(loocv_likelihood(data, init, participant, estimates[pidx].sel(n_events=n_eve).dropna('event'), verbose=verbose))
                else: #mp
                    with mp.Pool(processes=cpus) as pool:
                        loocv = pool.starmap(loocv_likelihood,
                                            zip(itertools.repeat(data), itertools.repeat(init),participants_idx,
                                                [estimates[x].sel(n_events=n_eve).dropna('event') for x in range(len(participants_idx))],itertools.repeat(1),itertools.repeat(verbose)))

                loocv_back.append(xr.DataArray(np.expand_dims(np.array(loocv).astype(np.float64),axis=0), 
                                        dims=('n_event', 'participant'),
                                        coords = {"n_event": np.array([n_eve]),
                                                    "participant": participants_idx}, 
                                        name = "loo_likelihood"))
                
            likelihoods = xr.concat(loocv_back, dim = 'n_event')

        all_likelihoods.append(likelihoods)    

    if len(all_likelihoods) == 1:
        all_likelihoods = all_likelihoods[0]
        all_estimates = all_estimates[0]

    return all_likelihoods, all_estimates


def backward_func(hmp_model, max_events=None, min_events=0, max_starting_points=1, method="random", tolerance=1e-4, maximization=True, max_iteration=1e3):
    '''
    Helper function for loocv_backward. Calls backward_estimation on hmp_model with provided args.
    '''    
    return hmp_model.backward_estimation(max_events, min_events, None, max_starting_points, method, tolerance, maximization, max_iteration)


def loocv_backward(init, data, max_events=None, min_events=0, max_starting_points=1, method="random", tolerance=1e-4, maximization=True, max_iteration=1e3, cpus=1, verbose=True):
    '''
    Performs leave-one-out cross validation using backward_estimation to calculate the initial fit.
    It will perform loocv by leaving out one participant, applying 'backward_estimation' to the 
    data to estimate a fit, and computing the likelihood of the data from the left out
    participant with the estimated parameters. This is repeated for all participants.

    Hmp model settings are based on init.
    
    Parameters
    ----------
    init : hmp model
        initialized hmp model
    data : xarray.Dataset
        xarray data from transform_data() 
    max_events : int
        Maximum number of events to be estimated, by default the output of hmp.models.hmp.compute_max_events()
    min_events : int
        The minimum number of events to be estimated
    max_fit : xarray
        To avoid re-estimating the model with maximum number of events it can be provided 
        with this arguments, defaults to None
    max_starting_points: int
        how many random starting points iteration to try for the model estimating the maximal number of events
    method: str
        What starting points generation method to use, 'random'or 'grid' (grid is not yet fully supported)
    tolerance: float
        Tolerance applied to the expectation maximization in the EM() function
    maximization: bool
        If True (Default) perform the maximization phase in EM() otherwhise skip
    max_iteration: int
        Maximum number of iteration for the expectation maximization in the EM() function
    cpus : int
        Nr of cpus to use. If 1, LOOCV is performed on a single CPU. Otherwise
        on the provided int or setting in init.
        We recommend using 1 CPU at this level on a laptop or normal PC. Only use multiple
        CPUs if you have *a lot* of memory available.
    verbose : bool
        
    Returns
    -------
    likelihood object and fitten backward estimation models
    '''

    return loocv_func(init, data, backward_func, func_args=[max_events, min_events, max_starting_points, method, tolerance, maximization, max_iteration], cpus=cpus, verbose=verbose)


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

def event_times(data, times, channel, stage, baseline=0):
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

    brp_data = np.tile(np.nan, (len(data.trial_x_participant), int(round(baseline+max(times.sel(event=stage+1).data- times.sel(event=stage).data)))+1))
    i=0
    for trial, trial_dat in data.groupby('trial_x_participant', squeeze=False):
        trial_time = slice(times.sel(event=stage, trial_x_participant=trial)-baseline, \
                                                 times.sel(event=stage+1, trial_x_participant=trial))
        trial_elec = trial_dat.sel(channels = channel, samples=trial_time).squeeze()
        try:#If only one sample -> TypeError: len() of unsized object
            brp_data[i, :len(trial_elec)] = trial_elec
        except:
            brp_data[i, :1] = trial_elec
            
        i += 1

    return brp_data    
    
def condition_selection(hmp_data, epoch_data, condition_string, variable='event', method='equal'):
    '''
    condition_selection select a subset from hmp_data. It selects epochs for which
    'condition_string' is in 'variable' based on 'method'.

    Parameters
    ----------
    hmp_data : xr.Dataset
        transformed EEG data for hmp, from utils.transform_data
    epoch_data : deprecated
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


def load_data(path):
    return xr.load_dataset(path)

    
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


#Save eeg/meg data with separate event file, and event_ids
def save_raw_events(fname, data, events, event_dict=None, overwrite=False):
    '''
    Saves raw MNE EEG/MEG data with a separate event file and event_ids, as:
    fname.fif (data)
    fname-eve.fif (events)
    fname-eve.txt (event_ids)

    Parameters
    ----------
    fname : str
        file name, following convention should end in eeg.fif
    data : mne.io.Raw
        EEG/MEG data object
    events : array of int, shape (n_events, 3)
        The array of events. The first column contains the event time in samples, 
        with first_samp included. The third column contains the event id.
    event_dict : dict
        Dictionary of {str: int} mappings of event IDs.
    overwrite : bool
        If True (default False), overwrite the destination file if it exists.
    '''

    #save data
    data.save(fname, overwrite=overwrite)
    
    #save events, same fname + -eve
    ftype = fname[-4:]
    fname_events = fname[:-4] + '-eve' + ftype
    mne.write_events(fname_events, events, overwrite=overwrite)
    
    #save event_dict if given
    if event_dict:
        with open(fname_events[:-4] + '.txt', 'w') as fp:
            json.dump(event_dict, fp)


#Load eeg/meeg data with event file and event_ids if exist
def load_raw_events(fname,preload=False):
    '''
    Load eeg/meeg data with event file and event_ids if exist, saved with save_raw_events

    Parameters
    ----------
    fname : str
        file name, following convention should end in eeg.fif, event file is then assumed
        to end in eeg-eve.fif, event dict in eeg-eve.txt
    preload : bool
        Preload data into memory for data manipulation and faster indexing. If True, the 
        data will be preloaded into memory (fast, requires large amount of memory). 
        If preload is a string, preload is the file name of a memory-mapped file which 
        is used to store the data on the hard drive (slower, requires less memory).

    Returns
    -------
    raw : mne.io.Raw
        Raw object with EEG/MEG data
    events : array of int, shape (n_events, 3)
        The array of events. The first column contains the event time in samples, with 
        first_samp included. The third column contains the event id.
    event_id : dict
        Dictionary of {str: int} mappings of event IDs.
    '''
    #load data
    raw = mne.io.read_raw(fname, preload=preload)

    #load events
    ftype = fname[-4:]
    fname_events = fname[:-4] + '-eve' + ftype
    if os.path.exists(fname_events):
        events = mne.read_events(fname_events)
    else:
        warnings.warn('No event file, data still loaded.')
        events = None

    #load event_dict
    fname_event_dict = fname_events[:-4] + '.txt'
    if os.path.exists(fname_event_dict):
        with open(fname_event_dict, 'r') as fp:
            event_dict = json.load(fp)
    else:
        warnings.warn('No event dictionary found.')
        event_dict = None

    return raw, events, event_dict


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