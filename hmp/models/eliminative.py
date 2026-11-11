"""Estimate all possible number events starting from a base model or the maximum possible."""

import gc
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from hmp.basedata import BaseData
from hmp.models.base import BaseModel
from hmp.models.event import EventModel
from hmp.patterndata import PatternData
from hmp.patterns import Pattern

default_colors = ["cornflowerblue", "indianred", "orange", "darkblue", "darkgreen", "gold", "brown"]


class EliminativeMethod(BaseModel):
    """Initialize the EliminativeMethod.

    Parameters
    ----------
    pattern : PatternData
        The pattern and properties to use for cross-correlation. Default is
        half sine with 50 ms width.
    location : int, optional
        Minimum duration (in ms) of the stages between two events, see ``EventModel``.
        Default is width of the event, which is by default 50 ms.
        Shorter values than the width of a pattern allow overlap of neighboring events
        but might result in the same event being duplicated in several events.
        Larger values will prevent duplication at the risk of missing neighboring events
    rho : float, optional
        Strength of the prior placed on the stage scales, see ``EventModel``. Default is 1.
    soft_min_location : float, optional
        Minimum location (in ms) used by the prior placed on the stage scales, see ``EventModel``.
        Default is None, half the width of ``pattern``.
    max_events : int, optional
        Maximum number of events to be estimated. By default, it is inferred using
        `compute_max_events()` if not provided.
    min_events : int, optional
        The minimum number of events to be estimated. Defaults to 1.
    base_fit : EventModel, optional
        To start the elimination from a specific model this argument can
        be provided with a fitted EventModel. Defaults to None.
    tolerance : float, optional
        Tolerance for the expectation maximization algorithm. Defaults to 1e-4.
    max_iteration : int, optional
        Maximum number of iterations for the expectation maximization algorithm. Defaults to 1000.
    distribution : str
        Probability distribution for the by-trial onset of stages can be
        one of 'gamma','lognormal','wald', or 'weibull'
    """

    def __init__(
        self,
        pattern: Pattern = None,
        location: float = None,
        rho: float = 1.0,
        soft_min_location: float = None,
        max_events: int | None = None,
        min_events: int = 1,
        base_fit: EventModel | None = None,
        tolerance: float = 1e-4,
        max_iteration: int = 1000,
        distribution: Any = None
    ):
        super().__init__(pattern, distribution)
        if location is None:
            location = self.pattern.width
        self.location = location
        self.rho = rho
        self.soft_min_location = soft_min_location
        self.max_events: int = max_events
        self.min_events: int = min_events
        self.base_fit: EventModel | None = base_fit
        self.tolerance: float = tolerance
        self.max_iteration: int = max_iteration
        self.submodels: dict[int, EventModel] = {}

    def fit(
        self,
        data: PatternData | BaseData | xr.DataArray,
        cpus: int = 1,
        verbose: bool = True
    ) -> None:
        """Perform the eliminative estimation.

        First, read or estimate the max_event solution, then estimate the
        max_event - 1 solution by iteratively removing one of the events
        and picking the one with the highest log-likelihood.

        Parameters
        ----------
        data : Data to fit the model on. One of two options:
            1. BaseData object or xr.DataArray containing preprocessed data.
            2. PatternData object.
            In case of option 1, data is cross-correlated with the pattern in self.pattern.
        cpus : int, optional
            Number of CPUs to use for parallel processing. Defaults to 1.

        Returns
        -------
        None
        """
        pattern_data = self._instantiate_data_pattern(data)

        if self.max_events is None:
            max_events = self._compute_max_events(pattern_data, self.location)
        else:
            max_events = self.max_events

        min_events = self.min_events

        if not self.base_fit:
            if verbose:
                print(
                    f"Estimating all solutions for maximal number of events ({max_events})"
                )
            base_fit = self.get_event_model(n_events=max_events, starting_points=1)
            base_fit.fit(pattern_data, verbose=False, cpus=cpus)
        else:
            base_fit = self.base_fit
        max_events = base_fit.n_events
        self.submodels[max_events] = base_fit

        for n_events in np.arange(max_events - 1, min_events-1, -1):
            event_model = self.get_event_model(n_events, starting_points=n_events+1)

            if verbose:
                print(f"Estimating all solutions for {n_events} events")

            time_pars_prev = self.submodels[n_events+1].xrtime_pars.dropna("stage").values
            channel_pars_prev = self.submodels[n_events+1].xrchannel_pars.dropna("event").values
            # Get Mean stage durations (in samples) of the previous model, i.e. location + mean of
            # the distribution
            durations_prev = self.submodels[n_events+1]._stage_locations(
                n_events + 2, pattern_data.sfreq
            ) + self.distribution.scale_to_mean(time_pars_prev[:, :, 1])
            # Now get stage locations of the new model
            locations = event_model._stage_locations(n_events + 1, pattern_data.sfreq)

            events_temp, pars_temp = [], []

            for event in np.arange(n_events + 1):  # creating all possible starting points
                events_temp.append(channel_pars_prev[:, np.arange(n_events + 1) != event,])

                durations = np.copy(durations_prev)
                # Full stage duration including locations get's moved into event
                durations[:, event] = (
                    durations[:, event] + durations[:, event + 1]
                )  # combine two stages into one
                # Drop event + 1 now
                durations = np.delete(durations, event + 1, axis=1)
                temp_pars = np.delete(np.copy(time_pars_prev), event + 1, axis=1)
                # Got to fix the mean at min 1 to prevent pmf from becoming zero everywhere
                # and need to evaluate scale at full duration - location (but only subtracted once)
                temp_pars[:, :, 1] = self.distribution.mean_to_scale(
                    np.maximum(durations - locations, 1)
                )
                pars_temp.append(temp_pars)
            event_model.fit(data=pattern_data,
                            channel_pars=np.array(events_temp),
                            time_pars=np.array(pars_temp),
                            verbose=False,
                            cpus=cpus
                        )

            gc.collect()
            self.submodels[n_events] = event_model
        self._fitted = True

    def transform(self,
                  data: PatternData | BaseData | xr.DataArray,
                  cpus: int = 1,
                  penalized: bool = False,
                  ):
        """
        Apply all fitted submodels to the provided data.

        Parameters
        ----------
        data : Data to fit the model on. One of two options:
            1. BaseData object or xr.DataArray containing preprocessed data.
            2. PatternData object.
            In case of option 1, data is cross-correlated with the pattern in self.pattern.
        cpus : int
            nr of cpus to use
        penalized : bool, optional
            If True, the log-likelihoods include the log-prior placed on the stage scales,
            see ``EventModel.transform``. Default is False.

        Returns
        -------
        likelihoods : list
            List of log-likelihoods for each submodel (number of events).
        xr_eventprobs : xarray.DataArray
            Concatenated event probability arrays for all submodels, indexed by number of events.
        """
        pattern_data = self._instantiate_data_pattern(data)

        if len(self.submodels) == 0:
            raise ValueError("Model has not been (succesfully) fitted yet, no fixed models.")
        likelihoods = []
        event_probs = []
        for _, event_model in self.submodels.items():
            lkh, prob = event_model.transform(pattern_data, cpus=cpus, penalized=penalized)
            likelihoods.append(lkh)
            event_probs.append(prob)
        xr_eventprobs = xr.concat(event_probs, dim=pd.Index(list(self.submodels), name="n_events"))
        return likelihoods, xr_eventprobs

    def _concatted_attr(self, attr_name):
        return xr.concat([getattr(model, attr_name) for model in self.submodels.values()],
                         dim=pd.Index(list(self.submodels), name="n_events"))

    def get_event_model(self, n_events, starting_points):
        return EventModel(
            n_events=n_events,
            pattern=self.pattern,
            location=self.location,
            rho=self.rho,
            soft_min_location=self.soft_min_location,
            starting_points=starting_points,
            tolerance=self.tolerance,
            max_iteration=self.max_iteration,
            distribution=self.distribution)
