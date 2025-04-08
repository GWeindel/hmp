"""Models to estimate event probabilities."""

import gc

import numpy as np
import pandas as pd
import xarray as xr

from hmp.models.base import BaseModel
from hmp.models.fixedn import FixedEventModel

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

default_colors = ["cornflowerblue", "indianred", "orange", "darkblue", "darkgreen", "gold", "brown"]


class BackwardEstimationModel(BaseModel):
    def __init__(self, *args, max_events=None, min_events=0, max_starting_points=1,
                 tolerance=1e-4, max_iteration=1e3, **kwargs):
        self.max_events = max_events
        self.min_events = min_events
        self.max_starting_points = max_starting_points
        self.tolerance = tolerance
        self.max_iteration = max_iteration
        self.submodels = {}
        super().__init__(*args, **kwargs)

    def fit(
        self,
        trial_data,
        max_events=None,
        min_events=0,
        base_fit=None,
        cpus=1,
    ):
        """Perform the backward estimation.

        First read or estimate max_event solution then estimate max_event - 1 solution by
        iteratively removing one of the event and pick the one with the highest
        loglikelihood

        Parameters
        ----------
        max_events : int
            Maximum number of events to be estimated, by default the output of
            hmp.models.hmp.compute_max_events()
        min_events : int
            The minimum number of events to be estimated
        base_fit : xarray
            To avoid re-estimating the model with maximum number of events it can be provided
            with this arguments, defaults to None
        tolerance: float
            Tolerance applied to the expectation maximization in the EM() function
        maximization: bool
            If True (Default) perform the maximization phase in EM() otherwhise skip
        max_iteration: int
            Maximum number of iteration for the expectation maximization in the EM() function
        """
        if max_events is None and base_fit is None:
            max_events = self.compute_max_events(trial_data)
        if not base_fit:
            print(
                f"Estimating all solutions for maximal number of events ({max_events})"
            )
            fixed_n_model = self.get_fixed_model(n_events=max_events, starting_points=1)
            loglikelihood, eventprobs = fixed_n_model.fit_transform(trial_data, verbose=False,
                                                                    cpus=cpus)
        else:
            loglikelihood, eventprobs = base_fit
        max_events = eventprobs.event.max().values + 1
        self.submodels[max_events] = fixed_n_model

        for n_events in np.arange(max_events - 1, min_events, -1):
            fixed_n_model = self.get_fixed_model(n_events, starting_points=n_events+1)

            print(f"Estimating all solutions for {n_events} events")

            pars_prev = self.submodels[n_events+1].xrparams.dropna("stage").values
            mags_prev = self.submodels[n_events+1].xrmags.dropna("event").values

            events_temp, pars_temp = [], []

            for event in np.arange(n_events + 1):  # creating all possible starting points
                events_temp.append(mags_prev[:, np.arange(n_events + 1) != event,])

                temp_pars = np.copy(pars_prev)
                temp_pars[:, event, 1] = (
                    temp_pars[:, event, 1] + temp_pars[:, event + 1, 1]
                )  # combine two stages into one
                temp_pars = np.delete(temp_pars, event + 1, axis=1)
                pars_temp.append(temp_pars)
            fixed_n_model.fit(
                            trial_data,
                            magnitudes=np.array(events_temp),
                            parameters=np.array(pars_temp),
                            verbose=False,
                            cpus=cpus
                        )

            gc.collect()
            self.submodels[n_events] = fixed_n_model
        self._fitted = True

    def transform(self, trial_data):
        if len(self.submodels) == 0:
            raise ValueError("Model has not been (succesfully) fitted yet, no fixed models.")
        likelihoods = []
        event_probs = []
        for n_events, fixed_n_model in self.submodels.items():
            lkh, prob = fixed_n_model.transform(trial_data)
            likelihoods.append(lkh)
            event_probs.append(prob)
        xr_eventprobs = xr.concat(event_probs, dim=pd.Index(list(self.submodels), name="n_events"))
        return likelihoods, xr_eventprobs

    def _concatted_attr(self, attr_name):
        return xr.concat([getattr(model, attr_name) for model in self.submodels.values()],
                         dim=pd.Index(list(self.submodels), name="n_events"))

    def __getattribute__(self, attr):
        property_list = {
            "xrtraces": "get traces",
            "xrlikelihoods": "get likelihoods",
            "xrparam_dev": "get dev params",
            "xrmags": "get xrmags",
            "xrparams": "get xrparams"
        }
        if attr in property_list:
            self._check_fitted(property_list[attr])
            return self._concatted_attr(attr)
        return super().__getattribute__(attr)

    def get_fixed_model(self, n_events, starting_points):
        return FixedEventModel(
            self.events, self.distribution, n_events=n_events,
            starting_points=starting_points,
            tolerance=self.tolerance,
            max_iteration=self.max_iteration)
