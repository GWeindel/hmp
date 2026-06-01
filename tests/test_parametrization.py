import pytest
from pathlib import Path
from hmp import io

from hmp.patterndata import PatternData
from hmp.patterns import Pattern, HalfSine
from hmp import distributions
from hmp.transformers import ProjPCA
from hmp.models import EventModel
import numpy as np
from test_io import init_data

DATA_DIR = Path("tests", "gen_data")
DATA_DIR_A = DATA_DIR / "dataset_a"
DATA_DIR_B = DATA_DIR / "dataset_b"

def data():
    event_b, event_a, epoch_data, positions, sfreq, n_events = init_data()
    hmp_data = ProjPCA(epoch_data, n_comp=5)
    return event_b, event_a, epoch_data, hmp_data, positions, sfreq, n_events

@pytest.mark.parametrize("width, template",
                         [(100, None),
                          (None, np.ones(100)),
                          (20, None),
                          (20, np.ones(20)),
                         ]
                        )
def test_patterns(width, template):
    _, _, _, hmp_data, _, _, n_events = data()
    if template is None:
        pattern = HalfSine(width)
    else:
        pattern = Pattern(template, width)
    print(pattern.template)
    pattern_data = PatternData.from_transformer(hmp_data, pattern=pattern)

    model = EventModel(pattern=pattern, n_events=n_events)
    model.fit_transform(pattern_data)

@pytest.mark.parametrize("width, template",
                         [(100, None),
                          (None, np.ones(100)),
                          (20, None),
                          (20, np.ones(20)),
                         ]
                        )
def test_patterns(width, template):
    _, _, _, hmp_data, _, _, n_events = data()
    if template is None:
        pattern = HalfSine(width)
    else:
        pattern = Pattern(template, width)
    pattern_data = PatternData.from_transformer(hmp_data, pattern=pattern)

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