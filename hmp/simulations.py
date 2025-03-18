"""Generating synthetic data to test HMP."""

import os
import os.path as op
from copy import deepcopy
from warnings import warn

import mne
import numpy as np
root = os.path.dirname(os.path.abspath(__file__))

def available_sources():
    """List available sources for sample subject in MNE."""
    labels = np.load(op.join(root,'simulation_parameters','sources_list.npy'))
    return labels


def simulation_sfreq():
    """Recovering sampling frequency of the sample data."""
    return simulation_info()["sfreq"]


def simulation_positions():
    """Recovering position of the simulated electrodes."""
    info = simulation_info()
    positions = np.delete(
        mne.channels.layout._find_topomap_coords(info, "eeg"), 52, axis=0
    )  # inferring channel location using MNE
    return positions


def simulation_info():
    """Recovering MNE's info file for simulated data."""
    info = mne.io.read_info(op.join(root,'simulation_parameters','info.fif'))
    return info


def event_shape(event_width, event_width_samples, steps):
    """Compute the template of a half-sine with given frequency f and sampling frequency."""
    event_idx = np.arange(event_width_samples) * steps + steps / 2
    event_frequency = 1000 / (
        event_width * 2
    )  # gives event frequency given that events are defined as half-sines
    template = np.sin(
        2 * np.pi * event_idx / 1000 * event_frequency
    )  # event morph based on a half sine with given event width and sampling frequency
    return template


def simulate(
    sources,
    n_trials,
    n_jobs,
    file,
    relations=None,
    data_type="eeg",
    n_subj=1,
    path="./",
    overwrite=False,
    verbose=False,
    noise=True,
    times=None,
    seed=None,
    sfreq=100,
    save_snr=False,
    save_noiseless=False,
    event_length_samples=None,
    proportions=None,
):
    """Simulate n_trials of EEG and/or MEG using MNE's tools based on the specified sources.

    Parameters
    ----------
    sources : list
        2D or 3D list with dimensions (n_subjects *) sources * source_parameters
        Source parameters should contain :
        - the name of the source (see the output of available_source())
        - the duration of the event (in frequency, usually 10Hz)
        - the amplitude or strength of the signal from the source, expressed in volt (e.g. 1e-8 V)
        - the duration of the preceding stage as a scipy.stats
            distribution (e.g. scipy.stats.gamma(a, scale))
    n_trials: int
        Number of trials
    n_jobs: int
        Number of jobs to use with MNE's function (multithreading)
    file: str
        Name of the file to be saved (number of the subject will be added)
    relations: list
        list of int describing to which previous event is each event connected, 1 means stimulus,
        2 means one event after stimulus, ... One event cannot be connected to an upcoming one
    data_type: str
        Whether to simulate "eeg" or "meg"
    n_subj: int
        How many subjects to simulate
    path: str
        path where to save the data
    overwrite: bool
        Whether to overwrite existing file
    verbose: bool
        Whether to display MNE's output
    noise: bool
        Adding noise to the simulated sources
    times: ndarray
        Deterministic simulation of event transitions times. Format is n_sources X n_trials
    location: float
        value in ms to add after a simulated event and before another one
    save_snr: bool
        Save the signal at peak value and electrode noise

    Returns
    -------
    generating_events: ndarray
        Times of the simulated events used to test for accurate recovery compared to estimation
    files: list
        list of file names (file + number of subject)
    """
    os.environ["SUBJECTS_DIR"] = op.join(root,'simulation_parameters')
    if not verbose:
        mne.set_log_level("warning")
    else:
        mne.set_log_level(True)
    if seed is not None:
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState(np.random.randint(low=0, high=3000))
    sources = np.array(sources, dtype=object)
    if len(np.shape(sources)) == 2:
        sources = [sources]  # If only one subject
    if relations is None:
        relations = np.arange(len(sources[0]) + 1) + 1
    if proportions is None:
        proportions = np.repeat(1, len(sources[0]) + 1)
    if np.shape(sources)[0] != n_subj:
        raise ValueError("Number of subject is not coherent with the sources provided")

    # Following code and comments largely comes from MNE examples (e.g. \
    # https://mne.tools/stable/auto_examples/simulation/simulated_raw_data_using_subject_anatomy.html)
    # It loads the data, info structure and forward solution for one example subject,
    # Note that all 'subject' will use this forward solution
    # First, we get an info structure from the test subject.
    info = simulation_info()
    # To simulate sources, we also need a source space. It can be obtained from the
    # forward solution of the sample subject.
    fwd = mne.read_forward_solution(op.join(root,'simulation_parameters','sample_fwd.fif'))
    with info._unlock():
        info["sfreq"] = sfreq
    if data_type == "eeg":
        picked_type = mne.pick_types(info, meg=False, eeg=True)
        fwd = mne.pick_types_forward(fwd, meg=False, eeg=True)
    elif data_type == "meg":
        picked_type = mne.pick_types(info, meg=True, eeg=False)
        fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)
    elif data_type == "eeg/meg":
        picked_type = mne.pick_types(info, meg=True, eeg=True)
        fwd = mne.pick_types_forward(fwd, meg=True, eeg=True)
    else:
        raise ValueError(f'Invalid data type {data_type}, expected "eeg", "meg" or "eeg/meg"')
    src = fwd["src"]
    info = mne.pick_info(info, picked_type)
    tstep = 1.0 / info["sfreq"]  # sample duration

    # For each subject, simulate associated sources
    files = []
    for subj in range(n_subj):
        sources_subj = sources[subj]
        # Pre-allocate the random times to avoid generating too long 'recordings'
        rand_times = np.zeros((len(sources_subj), n_trials))
        random_indices_list = []
        for s, source in enumerate(sources_subj):
            # How many trials wil have the event, default is all
            props_trial = int(np.round(n_trials * proportions[s]))
            if proportions[s] < 1:
                random_indices_list.append(
                    random_state.choice(np.arange(n_trials), size=props_trial, replace=False)
                )
            else:
                random_indices_list.append(np.arange(n_trials))
            rand_times[s, random_indices_list[-1]] = source[-1].rvs(
                size=len(random_indices_list[-1]), random_state=random_state
            )
        seq_index = np.diff(relations, prepend=0) > 0
        if times is None:
            if seq_index.all():  # If all events are sequential
                trial_time = np.sum(rand_times, axis=0)
            else:
                trial_time_seq = np.sum(rand_times[seq_index], axis=0)
                trial_time_nonseq = np.sum(rand_times[~seq_index], axis=0)
                
                trial_time = np.maximum(trial_time_seq, trial_time_nonseq)
        else:
            trial_time = np.sum(times, axis=1)
        trial_time /= 1000  # to seconds
        trial_time[1:] += trial_time[:-1] + 0.5
        trial_time = np.cumsum(trial_time)
        # Build simulator
        files_subj = []
        source_simulator = mne.simulation.SourceSimulator(
            src, tstep=tstep, first_samp=0, duration=trial_time[-1] + 10
        )  # add 10sec to the end of the last trial
        if n_subj == 1:
            subj_file = file + "_raw.fif"
        else:
            subj_file = file + f"_{subj}_raw.fif"
        if subj_file in os.listdir(path) and not overwrite:
            subj_file = path + subj_file
            warn(f"{subj_file} exists no new simulation performed", UserWarning)
            files_subj.append(subj_file)
            files_subj.append(subj_file.split(".fif")[0] + "_generating_events.npy")
            if save_snr:
                files_subj.append(subj_file.split(".fif")[0] + "_snr.npy")
            if save_noiseless:
                files_subj.append(file + "_noiseless_raw.fif")
            files.append(files_subj)
        else:
            subj_file = op.join(path, subj_file)
            print(f"Simulating {subj_file}")
            # stim_onset occurs every x samples.
            events = np.zeros((n_trials, 3), int)
            stim_onsets = 1 + trial_time  # offset of first stim is 1 s
            events[:, 0] = stim_onsets / tstep  # last event
            events[:, 2] = 1  # trigger 1 = stimulus

            # Fake source, actually stimulus onset
            selected_label = mne.read_labels_from_annot(
                '', regexp=sources_subj[0][0], subjects_dir=op.join(root,'simulation_parameters'), verbose=verbose
            )[0]
            label = mne.label.select_sources(
                '', selected_label, subjects_dir=op.join(root,'simulation_parameters'), random_state=random_state
            )
            source_time_series = np.array([1e-20])  # stim trigger
            source_simulator.add_data(label, source_time_series, events)

            trigger = 2
            generating_events = events
            for s, source in enumerate(sources_subj):
                if trigger == len(sources_subj) + 1:
                    source[2] = 1e-20  # Last source defines RT and is not an event per se
                selected_label = mne.read_labels_from_annot(
                    '', regexp=source[0], subjects_dir=op.join(root,'simulation_parameters'), verbose=verbose
                )[0]
                label = mne.label.select_sources(
                    '',
                    selected_label,
                    subjects_dir=op.join(root,'simulation_parameters'),
                    location=0,
                    grow_outside=False,
                    random_state=random_state,
                )
                # last two parameters ensure sources that are different enough
                # Define the time course of the activity for each source of the region to
                # activate
                if event_length_samples is None:
                    event_duration = int(((1 / source[1]) / 2) * info["sfreq"])
                    # Shift the trigger for the simulations
                    shift = event_duration // 2
                else:
                    event_duration = event_length_samples[s]
                    # Assumes the shortest event duration is a half-sine
                    shift = min(event_length_samples) // 2
                source_time_series = (
                    event_shape(((1000 / source[1]) / 2), event_duration, 1000 / info["sfreq"])
                    * source[2]
                )
                # adding source event, take as previous time the event defined by relation (default
                # is previous event)
                events = generating_events[generating_events[:, -1] == relations[s]].copy()
                random_indices = random_indices_list[s]
                if times is None:
                    # Always at least one sample later
                    rand_i = np.maximum(1, rand_times[s] / (tstep * 1000))
                    rand_i = np.round(rand_i, decimals=0)
                else:
                    rand_i = times[:, s] / (tstep * 1000)
                if len(rand_i[rand_i < 0]) > 0:
                    warn(
                        f"Negative stage duration were found, 1 is imputed for the"
                        f"{len(rand_i[rand_i < 0])} trial(s)",
                        UserWarning,
                    )
                    rand_i[rand_i < 0] = 1
                events[random_indices, 0] = (
                    events[random_indices, 0] + rand_i[random_indices]
                )  # Events sample.
                events[:, 2] = trigger  # All events have the sample id.
                trigger += 1
                generating_events = np.concatenate([generating_events, events.copy()])
                # Shift event to onset when simulating pattern
                events[random_indices, 0] = events[random_indices, 0] - shift
                # add these events
                source_simulator.add_data(label, source_time_series, events[random_indices])

            generating_events = generating_events[generating_events[:, 0].argsort()]
            # Project the source time series to sensor space and add some noise. The source
            # simulator can be given directly to the simulate_raw function.
            raw = mne.simulation.simulate_raw(
                info, source_simulator, forward=fwd, n_jobs=n_jobs, verbose=verbose
            )

            n_events = len(sources_subj) - 1
            if save_noiseless:
                raw.save(file + "_noiseless_raw.fif", overwrite=True)
            if data_type == "eeg":
                raw = raw.pick_types(meg=False, eeg=True, stim=True)
            elif data_type == "meg":
                raw = raw.pick_types(meg=True, eeg=False, stim=True)
            elif data_type == "eeg/meg":
                raw = raw.pick_types(meg=True, eeg=True, stim=True)
            if save_snr:
                snr = np.zeros((len(info["ch_names"]), n_events))
                data = deepcopy(raw.get_data())
                for event in range(n_events):
                    times_out = generating_events[generating_events[:, 2] == event + 2, 0]
                    snr[:, event] = np.mean((data[:, times_out]) ** 2, axis=-1)
            if noise:
                cov = mne.make_ad_hoc_cov(raw.info, verbose=verbose)
                mne.simulation.add_noise(
                    raw,
                    cov,
                    verbose=verbose,
                    iir_filter=[0.2, -0.2, 0.04],
                    random_state=random_state,
                )

            raw.save(subj_file, overwrite=True)
            files_subj.append(subj_file)
            np.save(subj_file.split(".fif")[0] + "_generating_events.npy", generating_events)
            files_subj.append(subj_file.split(".fif")[0] + "_generating_events.npy")
            if save_snr:
                data = raw.get_data()
                for event in range(n_events):
                    times_out = generating_events[generating_events[:, 2] == event + 2, 0]
                    snr[:, event] /= np.var(data[:, times_out], axis=-1)
                np.save(subj_file.split(".fif")[0] + "_snr.npy", snr)
                files_subj.append(subj_file.split(".fif")[0] + "_snr.npy")
            files.append(files_subj)
            if save_noiseless:
                files_subj.append(file + "_noiseless_raw.fif")
            print(f"{subj_file} simulated")

    if n_subj == 1:
        files = files[0]
    return files


def demo(cpus, n_events, seed=123):
    """Create example data for the tutorials."""
    ## Imports and code specific to the simulation (see tutorial 3 and 4 for real data)
    from scipy.stats import gamma

    from hmp.utils import read_mne_data

    random_gen = np.random.default_rng(seed=seed)

    ## Parameters for the simulations
    frequency, amplitude = (
        10.0,
        0.3e-7,
    )  # Frequency of the transition event and its amplitude in Volt
    shape = 2  # shape of the gamma distribution

    # Storing electrode position, specific to the simulations
    positions = simulation_info()  # Electrode position
    sfreq = 250
    all_source_names = available_sources()  # all brain sources you can play with
    n_trials = 50  # Number of trials to simulate

    # Randomly specify the transition events
    name_sources = random_gen.choice(
        all_source_names, n_events + 1, replace=False
    )  # randomly pick source without replacement
    times = np.random.uniform(40, 150, n_events + 1) / shape

    sources = []
    for source in range(len(name_sources)):
        sources.append(
            [name_sources[source], frequency, amplitude, gamma(shape, scale=times[source])]
        )  # gamma returns times in ms

    file = "dataset_tutorial2"  # Name of the file to save

    # Simulating and recover information on electrode location and true time of the simulated events
    files = simulate(
        sources, n_trials, cpus, file, overwrite=False, seed=seed, noise=True, sfreq=sfreq
    )

    generating_events = np.load(files[1])

    number_of_sources = len(np.unique(generating_events[:, 2])[1:])  # one trigger = one source
    random_source_times = np.reshape(
        np.diff(generating_events[:, 0], prepend=0), (n_trials, number_of_sources + 1)
    )[:, 1:]  # By-trial generated event times

    # Reading the necessary info to read the EEG data
    resp_trigger = int(
        np.max(np.unique(generating_events[:, 2]))
    )  # Resp trigger is the last source in each trial
    event_id = {"stimulus": 1}
    resp_id = {"response": resp_trigger}
    events = generating_events[
        (generating_events[:, 2] == 1) | (generating_events[:, 2] == resp_trigger)
    ]  # only retain stimulus and response triggers

    # Reading the data
    eeg_dat = read_mne_data(files[0], event_id, resp_id, events_provided=events, verbose=False)

    all_other_chans = range(len(positions.ch_names[:-61]))  # non-eeg
    chan_list = list(np.arange(len(positions.ch_names)))
    chan_list = [e for e in chan_list if e not in all_other_chans]
    chan_list.pop(52)  # Bad elec
    positions = mne.pick_info(positions, sel=chan_list)
    return eeg_dat, random_source_times, positions


def classification_true(true_topologies, test_topologies):
    """Classifies event as belonging to one of the true events.

    Parameters,
    ----------
    true_topologies : xarray.DataArray
        topologies for the true events simulated obtained from
        `init.compute_topologies(epoch_data, test_estimates, test_init, mean=True)`
    test_tolopogies : xarray.DataArray
        topologies for the events found in the estimation procedure obtained from
        `init.compute_topologies(epoch_data, true_estimates, true_init, mean=True)`


    Returns
    -------
    idx_true_positive: np.array
        index of the true events found in the test estimation
    corresp_true_idx: np.array
        index in the test estimate that correspond to the indexes in corresp_true_idx

    """
    test_topologies = (test_topologies.copy() - test_topologies.mean(axis=1)) / test_topologies.std(
        axis=1
    )
    true_topologies = (true_topologies.copy() - true_topologies.mean(axis=1)) / true_topologies.std(
        axis=1
    )
    true0 = np.vstack(
        (np.zeros(true_topologies.shape[1]), true_topologies)
    )  # add a zero electrode event
    classif = np.zeros(
        test_topologies.shape[0], dtype=int
    )  # array of categorization in true events
    classif_vals = np.zeros(test_topologies.shape[0])  # values of the squared diff
    for i, test_ev in enumerate(test_topologies):
        all_distances = np.zeros(len(true0))
        for j, true_ev in enumerate(true0):
            all_distances[j] = np.median(np.abs(true_ev - test_ev))
        classif[i] = np.argmin(all_distances)
        classif_vals[i] = all_distances[classif[i]]

    mapping_true = {}
    for test_idx, (idx, val) in enumerate(zip(classif, classif_vals)):
        if idx > 0:
            if idx not in mapping_true or val < mapping_true[idx]:
                mapping_true[idx] = test_idx

    corresp_true_idx = (
        np.array(list(mapping_true.keys())) - 1
    )  # Corresponding true index, excluding 0 event
    idx_true_positive = np.array(list(mapping_true.values()))
    return idx_true_positive, corresp_true_idx


def simulated_times_and_parameters(generating_events, init, trial_data, resampling_freq=None, data=None):
    """Recover the generating HMP parameters from the simulated EEG data.

    Parameters,
    ----------
    generating_events: ndarray
        Times of the simulated events created by the function simulate()
    init: hmp class
        Initialized HMP object
    resampling_freq: float
        Value of the new sampling frequency if there is a difference between the initialised HMP
        object and the generating_events
    data : ndarray
        Use alternative data instead of crosscorrelation contained in init.crosscorr


    Returns
    -------
    random_source_times: np.array
        index of the true events found in the test estimation
    true_parameters : list
        list of true distribution parameters (2D stage * parameter).
    true_magnitudes: ndarray
        2D ndarray n_events * components, true electrode contribution to event
    true_activities: ndarray
        Actual values at simulated event times
    """
    sfreq = init.sfreq
    n_stages = len(np.unique(generating_events[:, 2])[1:])  # one trigger = one source
    n_events = n_stages - 1
    if resampling_freq is None:
        resampling_freq = sfreq

    # Recover the actual time of the simulated events
    random_source_times = np.zeros((int(len(generating_events) / (n_stages + 1)), n_stages))
    i, x = 1, 0
    while x < len(random_source_times):
        for j in np.arange(n_stages):  # recovering the individual duration- of event onset
            random_source_times[x, j] = generating_events[i, 0] - generating_events[i - 1, 0]
            i += 1
        i += 1
        x += 1

    ## Recover parameters
    true_parameters = np.tile(init.shape, (n_stages, 2))
    true_parameters[:, 1] = init.mean_to_scale(np.mean(random_source_times, axis=0), init.shape)
    true_parameters[true_parameters[:, 1] <= 0, 1] = 1e-3  # Can happen in corner cases
    random_source_times = random_source_times * (1000 / sfreq) / (1000 / resampling_freq)
    ## Recover magnitudes
    sample_times = np.zeros((trial_data.n_trials, n_events), dtype=int)
    for event in range(n_events):
        for trial in range(trial_data.n_trials):
            trial_time = trial_data.starts[trial] + np.sum(random_source_times[trial, : event + 1])
            if trial_data.ends[trial] >= trial_time:  # exceeds RT
                sample_times[trial, event] = trial_time
            else:
                sample_times[trial, event] = trial_data.ends[trial]
    if data is None:  # use crosscorrelated data
        true_activities = trial_data.cross_corr[sample_times[:, :]]
    else:
        true_activities = data[sample_times[:, :]]
    true_magnitudes = np.mean(true_activities, axis=0)
    return random_source_times.astype(int), true_parameters, true_magnitudes, true_activities
