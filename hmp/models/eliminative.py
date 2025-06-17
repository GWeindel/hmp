"""Method to estimate all possible number events starting from a base model or the maximum possible."""

import gc

import numpy as np
import pandas as pd
import xarray as xr

from hmp.models.base import BaseModel
from hmp.models.event import EventModel
from hmp.trialdata import TrialData


try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

default_colors = ["cornflowerblue", "indianred", "orange", "darkblue", "darkgreen", "gold", "brown"]


class EliminativeMethod(BaseModel):
    """Initialize the EliminativeMethod.

    Parameters
    ----------
    max_events : int, optional
        Maximum number of events to estimate. Defaults to None, this number is then inferred from the data.
    min_events : int, optional
        Minimum number of events to estimate. Defaults to 1.
    max_starting_points : int, optional
        Number of starting points for the estimation of the model with the maximum number of events.
        Defaults to 1.
    tolerance : float, optional
        Tolerance for the expectation maximization algorithm. Defaults to 1e-4.
    max_iteration : int, optional
        Maximum number of iterations for the expectation maximization algorithm. Defaults to 1000.
    """
    
    def __init__(
        self,
        *args,
        max_events: int | None = None,
        min_events: int = 0,
        max_starting_points: int = 1,
        tolerance: float = 1e-4,
        max_iteration: int = 1000,
        **kwargs,
    ):
        self.max_events: int | None = max_events
        self.min_events: int = min_events
        self.max_starting_points: int = max_starting_points
        self.tolerance: float = tolerance
        self.max_iteration: int = max_iteration
        self.submodels: dict[int, EventModel] = {}
        super().__init__(*args, **kwargs)

    def fit(
        self,
        trial_data: TrialData,
        max_events: int | None = None, #TODO remove, take attribute
        min_events: int = 0, #TODO remove, take attribute
        base_fit: tuple[np.ndarray, xr.Dataset] | None = None,
        cpus: int = 1,
    ) -> None:
        """Perform the eliminative estimation.

        First, read or estimate the max_event solution, then estimate the max_event - 1 solution 
        by iteratively removing one of the events and picking the one with the highest log-likelihood.

        Parameters
        ----------
        trial_data : TrialData
            The dataset containing the crosscorrelated data and infos on durations and trials.
        max_events : int, optional
            Maximum number of events to be estimated. By default, it is inferred using 
            `compute_max_events()` if not provided. 
        min_events : int, optional
            The minimum number of events to be estimated. Defaults to 1.
        base_fit : EventModel, optional
            To avoid re-estimating the model with the maximum number of events, this argument can 
            be provided with a fitted EventModel. Defaults to None.
        cpus : int, optional
            Number of CPUs to use for parallel processing. Defaults to 1.

        Returns
        -------
        None
        """

        if max_events is None and base_fit is None:
            max_events = self.compute_max_events(trial_data)
        if not base_fit:
            print(
                f"Estimating all solutions for maximal number of events ({max_events})"
            )
            event_model = self.get_fixed_model(n_events=max_events, starting_points=1)
            loglikelihood, eventprobs = event_model.fit_transform(trial_data, verbose=False,
                                                                    cpus=cpus)
        else:
            loglikelihood, eventprobs = base_fit
        max_events = eventprobs.event.max().values + 1
        self.submodels[max_events] = event_model

        for n_events in np.arange(max_events - 1, min_events, -1):
            event_model = self.get_fixed_model(n_events, starting_points=n_events+1)

            print(f"Estimating all solutions for {n_events} events")

            time_pars_prev = self.submodels[n_events+1].xrtime_pars.dropna("stage").values
            channel_pars_prev = self.submodels[n_events+1].xrchannel_pars.dropna("event").values

            events_temp, pars_temp = [], []

            for event in np.arange(n_events + 1):  # creating all possible starting points
                events_temp.append(channel_pars_prev[:, np.arange(n_events + 1) != event,])

                temp_pars = np.copy(time_pars_prev)
                temp_pars[:, event, 1] = (
                    temp_pars[:, event, 1] + temp_pars[:, event + 1, 1]
                )  # combine two stages into one
                temp_pars = np.delete(temp_pars, event + 1, axis=1)
                pars_temp.append(temp_pars)
            event_model.fit(
                            trial_data,
                            channel_pars=np.array(events_temp),
                            time_pars=np.array(pars_temp),
                            verbose=False,
                            cpus=cpus
                        )

            gc.collect()
            self.submodels[n_events] = event_model
        self._fitted = True

    def transform(self, trial_data):
        """
        Apply all fitted submodels to the provided trial data.

        Parameters
        ----------
        trial_data : TrialData
            The dataset containing the crosscorrelated data and information on durations and trials.

        Returns
        -------
        likelihoods : list
            List of log-likelihoods for each submodel (number of events).
        xr_eventprobs : xarray.DataArray
            Concatenated event probability arrays for all submodels, indexed by number of events.
        """
        if len(self.submodels) == 0:
            raise ValueError("Model has not been (succesfully) fitted yet, no fixed models.")
        likelihoods = []
        event_probs = []
        for n_events, event_model in self.submodels.items():
            lkh, prob = event_model.transform(trial_data)
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
            "xrtime_pars_dev": "get dev time pars",
            "xrchannel_pars": "get xrchannel_pars",
            "xrtime_pars": "get xrtime_pars"
        }
        if attr in property_list:
            self._check_fitted(property_list[attr])
            return self._concatted_attr(attr)
        return super().__getattribute__(attr)

    def get_fixed_model(self, n_events, starting_points):
        return EventModel(
            self.pattern, self.distribution, n_events=n_events,
            starting_points=starting_points,
            tolerance=self.tolerance,
            max_iteration=self.max_iteration)
