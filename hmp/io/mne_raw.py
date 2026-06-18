"""Models to estimate event probabilities."""
from abc import ABC, abstractmethod
from typing import Any



def read_mne_raw():
    ...


def _extract_mne_events(data, events, stimulus_id, response_id, verbose):
    if events is None:
        try:
            events = mne.find_events(
                data, verbose=verbose, min_duration=1 / data.info["sfreq"]
            )
        except ValueError:
            events, event_id = mne.events_from_annotations(data, verbose=verbose)[0]
        if (
            events[0, 1] > 0
        ):  # bug from some stim channel, should be 0 otherwise indicates offset in triggers
            print(
                f"Correcting event values as trigger channel has offset "
                f"{np.unique(events[:, 1])}"
            )
            events[:, 2] = events[:, 2] - events[:, 1]  # correction on event value
    events_values = np.concatenate(
        [
            np.array([x for x in stimulus_id.values()]),
            np.array([x for x in response_id.values()]),
        ]
    )
    events = np.array(
        [list(x) for x in events if x[2] in events_values]
    )  # only keeps events with stim or response
    event_id = {**stimulus_id, **response_id}  # stimulus_id | response_id
    stim = list(stimulus_id.keys())

    return events, event_id, stim