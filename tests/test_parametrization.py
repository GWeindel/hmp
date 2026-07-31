import pytest
from pathlib import Path
from hmp import io

import hmp
from hmp.patterndata import PatternData
from hmp.patterns import Pattern, HalfSine
from hmp import distributions
from hmp.models import EventModel
import numpy as np
from test_io import init_data

def data():
    event_b, event_a, epoch_data, positions, sfreq, n_events = init_data()
    hmp_data = hmp.basedata.default(epoch_data, n_comp=5, duration_id = 'response_time')
    return event_b, event_a, epoch_data, hmp_data, positions, sfreq, n_events

@pytest.mark.parametrize("width, template",
                         [(100, None),
                          (None, np.ones(100)),
                          (30, None),
                          (30, np.ones(30)),
                         ]
                        )
def test_patterns(width, template):
    _, _, _, hmp_data, _, _, n_events = data()
    if template is None:
        pattern = HalfSine(width)
    else:
        pattern = Pattern(template, width)
    pattern_data = PatternData.from_basedata(hmp_data, pattern=pattern)

    model = EventModel(pattern=pattern, n_events=n_events)
    model.fit_transform(pattern_data)

@pytest.mark.parametrize("width, template",
                         [(100, None),
                          (None, np.ones(100)),
                          (30, None),
                          (30, np.ones(30)),
                         ]
                        )
def test_patterns(width, template):
    _, _, _, hmp_data, _, _, n_events = data()
    if template is None:
        pattern = HalfSine(width)
    else:
        pattern = Pattern(template, width)
    pattern_data = PatternData.from_basedata(hmp_data, pattern=pattern)

    model = EventModel(pattern=pattern, n_events=n_events)
    model.fit_transform(pattern_data)

def test_distributions():
    _, _, _, hmp_data, _, _, n_events = data()
    n_events = 1
    for name, dist_class in vars(distributions).items():
        if isinstance(dist_class, type):
            distribution = dist_class(shape=2)
            model = EventModel(distribution=distribution, n_events=n_events)
            model.fit(hmp_data)
            assert np.isclose(distribution.mean_to_scale(
                distribution.scale_to_mean(model.time_pars[:,:,1])
            ), model.time_pars[:,:,1], atol=1e-5, rtol=0).all()
