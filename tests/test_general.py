import os
import shutil
import numpy as np
from scipy.stats import gamma
import hmp
from hmp import simulations
import warnings
import pytest

warnings.filterwarnings("ignore", category=UserWarning)

# Simulation parameters
# Same source everywhere
names = ['bankssts-rh','bankssts-rh','bankssts-rh','bankssts-rh']
n_events = len(names)-1 # Last is simulated resp
# Remove default baseline corr
epoching=dict(baseline=None)

llks = {(100, 50):-3.588911,
        (100, 30):-2.5232670,
        (200, 25):-3.588911,
        (256, 35.5):-4.7588858,
        (1000,70):-8.8224058,
        }

@pytest.mark.parametrize("sfreq,width",
                         [(100, 50),
                          (100, 30),
                          (200, 25),
                          (256, 26), # Cannot fit, should give 2 events max
                          (1000, 70),])
def test_basic_operations(sfreq, width):
    """Simulate noiseless trial with minimum duration possible
    """
    n_trials = 1
    cpus=1
    gen_width = width
    # Minimum possible time with provided halfsine and 3 events
    times_a = np.array([[1000/sfreq, gen_width, gen_width, 1000/sfreq]], dtype='float64')
    sources = []
    for cur_name in names:
        sources.append([cur_name, 1000/(gen_width*2), 1e-4, None])
    files = simulations.simulate(sources, n_trials, cpus, 'test_raw', overwrite=True,
        sfreq=sfreq, times=times_a, noise=False, seed=1)
    eeg_file = [files[0][0]]
    event_file = files[0][1]
    #load electrode position, specific to the simulations
    info = simulations.positions()
    #Recovering the events to epoch the data (in the number of trials defined above)
    events = np.load(event_file)
    resp_trigger = int(np.max(np.unique(events[:,2])))#Resp trigger is the last source in each trial
    event_id = {'stimulus/dummy':1}#trigger 1 = stimulus
    resp_id = {'response/dummy':resp_trigger}
    #Keeping only stimulus and response triggers
    
    epoch_data, info = hmp.io.read_mne_raw(eeg_file,
                                 centering_id=event_id,
                                 event_id=resp_id, 
                                 epoching_kwargs=epoching,
                                 events_provided=[events],#Usually not necessary for real data
                                )
    
    weights = epoch_data.sel(sample=0).data.squeeze().expand_dims('component', axis=1)
    weights = weights.assign_coords(component=[0])
    
    hmp_data = hmp.basedata.from_io(epoch_data)
    hmp_data.crop_reject_epochs(duration_id='response_time', offsets=(width/2+1)/1000)
    hmp_data.project(hmp.projectors.Custom(weights), )

    pattern = hmp.patterns.HalfSine(width=width)
    trial_data = hmp.patterndata.PatternData.from_basedata(hmp_data, pattern=pattern)
    sim_source_times = np.ediff1d(events[:,0],to_begin=0)[events[:,2] > 1]
    peaks = sim_source_times[:-1].cumsum()

    model = hmp.models.EventModel(n_events=3, pattern=pattern)
    llk, estimates = model.fit_transform(trial_data)
    test_topos = hmp.utils.event_channels(epoch_data, estimates, mean=True)
    if (sfreq, width) == (256, 26):
         assert model._compute_max_events(trial_data, location=width) == 2, \
        'Incorrect max event calculation'
    else:
        assert (np.isclose(times_a, hmp.utils.event_times(estimates, duration=True, add_rt=True, as_time=True), atol=.01)).all(),\
            'Incorrect timing extraction'
        assert np.all(trial_data.cross_corr[peaks] == trial_data.cross_corr[peaks][0]),\
            'Unexpected peak value in crosscorrelation'
        assert np.isclose(llk, llks[(sfreq, width)], atol=1e-6), \
            f'Incorrect likelihood from model, expected {llks[(sfreq, width)]} got {llk}'
        assert (test_topos == test_topos.isel(event=0)).all(),\
            'Incorrect topo calculation from estimates'
        assert model._compute_max_events(trial_data, location=width) == 3, \
            'Incorrect max event calculation'

    if (sfreq, width) == (100, 50):
        starts = trial_data.starts
        ends = trial_data.ends
        durations = ends - starts + 1
        max_duration = np.max(durations)

        # Compute pmf with locations enforced as called by model during estim_probs:
        pmf = model.location_pdf(model.time_pars[0],
                                max_duration,
                                model.n_events+1,
                                trial_data.sfreq,
                                trial_data.cross_corr.dtype)

        # Compute pmf for intervals 1 and 2 manually and extract the fifth density
        # which should be set to zero by `location_pdf` for a location of five but is not!
        pmf1 = model.distribution.pdf(np.arange(max_duration),
                                    model.time_pars[0][1,0],
                                    scale=model.time_pars[0][1,1])
        pmf1 /= np.sum(pmf1)
        m14 = pmf1[4]
        pmf2 = model.distribution.pdf(np.arange(max_duration),
                                    model.time_pars[0][2,0],
                                    scale=model.time_pars[0][2,1])
        pmf2 /= np.sum(pmf2)
        m24 = pmf2[4]

        # Assert that default location is [0,5,5,0]
        locations_samples = model._time_to_samples(model.locations, sfreq)
        assert np.all(locations_samples == [0,5,5,0])

        # These two succeed but should fail
        assert np.all(pmf[:5,1] == np.array([0, 0, 0, 0, m14],dtype=pmf.dtype))
        assert np.all(pmf[:5,2] == np.array([0, 0, 0, 0, m24],dtype=pmf.dtype))

        # These two fail but should succeed
        assert np.all(pmf[:5,1] == [0 for _ in range(locations_samples[1])])
        assert np.all(pmf[:5,2] == [0 for _ in range(locations_samples[2])])