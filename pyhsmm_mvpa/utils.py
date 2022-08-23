'''

'''

import numpy as np
import scipy.stats as stats
import xarray as xr
import multiprocessing as mp
import itertools
import warnings

warnings.filterwarnings('ignore', 'Degrees of freedom <= 0 for slice.', )#weird warning, likely due to nan in xarray, not important but better fix it later

def read_mne_EEG(pfiles, event_id, resp_id, sfreq, events=None,
                 tmin=-.2, tmax=5, offset_after_resp = .1, low_pass=.5, \
                 high_pass = 30, upper_limit_RT=5, lower_limit_RT=0.001, reject_threshold=None):
    ''' 
    Reads EEG data using MNE's integrated function. If no events is provided 
    
    Parameters
    ----------
    ...
    data : ndarray
        2D ndarray with n_samples * components

    Returns
    -------
    bumbs : ndarray
        a 2D ndarray with samples * PC components where cell values have
        been correlated with bump morphology
    '''
    import mne
    tstep = 1/sfreq
    epoch_data = [] 
    if isinstance(pfiles,str):#only one participant
        pfiles = [pfiles]
    for participant in pfiles:
        if '.fif' in participant:
            data = mne.io.read_raw_fif(participant, preload=False, verbose=False)
        elif '.bdf' in participant:
            data = mne.io.read_raw_bdf(participant, preload=False, verbose=False)
        else:
            raise ValueError(f'Unknown EEG file format for participant {participant}')
        data.load_data()
        data.filter(low_pass, high_pass, fir_design='firwin', verbose=False)#Filtering out frequency outside range .5 and 30Hz, as study by Anderson et al. (Berberyan used 40 Hz)
        # Loading events (in our case one event = one trial)
        if events is None:
            events = mne.find_events(data, verbose=False)
            if events[0,1] > 0:#bug from some stim channel, should be 0 otherwise indicates offset in the trggers
                events[:,2] = events[:,2]-events[:,1]#correction on event value
            events_values = np.concatenate([np.array([x for x in event_id.values()]), np.array([x for x in resp_id.values()])])
            events = np.array([list(x) for x in events if x[2] in events_values])#only keeps events with stim or response
        #else:
        #    if sfreq < data.info['sfreq']:
        #        raise ValueError('Cannot provide events and specify downsampling')

        if sfreq < data.info['sfreq']:
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
                        verbose=True,detrend=1, on_missing = 'warn',
                        metadata=metadata, reject_by_annotation=True, reject=reject_threshold)
        data_epoch = epochs.get_data()

        valid_epochs_idx = [x for x in np.arange(len(epochs.drop_log)) if epochs.drop_log[x] == ()]
        correct_stim_timing  = np.array([list(x) for x in events if x[2] in event_id.values()])[valid_epochs_idx,0]
        stim_events = np.array([x for x in np.arange(len(events)) if events[x,0] in correct_stim_timing])
        
        rts=[]#reaction times
        trigger = []
        for i in stim_events:
            if events[i+1,2] in resp_id.values():
                rts.append(events[i+1,0] - events[i,0] )
            elif events[i+1,2] not in resp_id.values(): #trials without resp
                rts.append(0)
            else:
                raise ValueError('Problem in event values')
        rts = np.array(rts)
        print(f'Applying reaction time trim to keep RTs between {lower_limit_RT} and {upper_limit_RT} seconds')
        rts[rts > sfreq*upper_limit_RT] = 0 #removes RT above x sec
        rts[rts < sfreq*lower_limit_RT] = 0 #removes RT below x sec, important as determines max bumps
        triggers = epochs.metadata["event_name"].reset_index(drop=True)
        cropped_data_epoch = np.empty([len(rts[rts> 0]), len(epochs.ch_names), max(rts)+offset_after_resp_samples])
        cropped_data_epoch[:] = np.nan
        cropped_trigger = []
        j = 0
        for i in np.arange(len(data_epoch)):
            if rts[i] > 0:
                cropped_trigger.append(triggers[i])
            #Crops the epochs to time 0 (stim onset) up to RT
                cropped_data_epoch[j,:,:rts[i]+offset_after_resp_samples] = \
                (data_epoch[i,:,epochs.time_as_index(0)[0]:\
                epochs.time_as_index(0)[0]+int(rts[i])+offset_after_resp_samples])
                j += 1
        x = 0
        while np.isnan(cropped_data_epoch[-1]).all():#Weird bug I guess it is perhps due to too long epoch?
            cropped_data_epoch = cropped_data_epoch[:-1]
            x += 1
        if x > 0:
            print(f'RTs > 0 longer than expected ({x})')

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

def transform_data(data, subjects_variable, apply_standard=True,  apply_zscore=True, method='pca', n_comp=None, single=False, return_weights=False):
    #Extract durations of epochs (equivalent to RTs) to partition the stacked data

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
        return data, pca_data, pca.explained_variance_, means
    else:
        return data

def stack_data(data, subjects_variable, single=False):
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
        data = data.stack(all_samples=[subjects_variable,'epochs',"samples"]).dropna(dim="all_samples")
    return xr.Dataset({'data':data, 'starts':starts, 'ends':ends})


def LOOCV(data, subject, n_bumps, iterative_fits, sfreq, bump_width=50):
    from pyhsmm_mvpa.models import hsmm
    #Looping over possible number of bumps
    subjects_idx = data.participant.values
    likelihoods_loo = []
    
    #Extracting data without left out subject
    stacked_loo = stack_data(data.sel(participant= subjects_idx[subjects_idx!=subject],drop=False),\
                           'participant')
    #Fitting the HsMM using previous estimated parameters as initial parameters
    model_loo = hsmm(stacked_loo.data.data.T, stacked_loo.starts.data, stacked_loo.ends.data, sf=sfreq, bump_width=bump_width)
    fit = model_loo.fit_single(n_bumps, iterative_fits.magnitudes, iterative_fits.parameters, 1, False, verbose=False)

    #Evaluating likelihood for left out subject
    #Extracting data of left out subject
    stacked_left_out = stack_data(data.sel(participant=subject, drop=False),\
                           'participant',single=True)

    model_left_out = hsmm(stacked_left_out.data.T, stacked_left_out.starts.data, stacked_left_out.ends.data, sf=sfreq, bump_width=bump_width)
    likelihood = model_left_out.calc_EEG_50h(fit.magnitudes, fit.parameters, n_bumps,True)
    return likelihood, subject


def reconstruct(magnitudes, PCs, eigen, means):
    '''
    Reconstruct electrode activity from PCA
    Parameters
    ----------
    magnitudes:  
        2D or 3D ndarray with [n_components * n_bumps] can also contain several estimations\ [estimation * n_components * n_bumps]
    PCs: 
        2D ndarray with PCA loadings [channels x n_components]
    eigen: 
        PCA eigenvalues of the covariance matrix of data [n_components]
    means: 
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
