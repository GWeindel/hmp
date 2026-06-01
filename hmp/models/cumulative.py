"""Models to estimate cumulative event models."""

from typing import Any
from warnings import warn

import numpy as np
import xarray as xr
from joblib import Parallel, delayed

from hmp.crossvalidation import pseudo_kfold
from hmp.models.base import BaseModel
from hmp.models.event import EventModel
from hmp.patterndata import PatternData
from hmp.patterns import Pattern
from hmp.transformers import BaseTransformer

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

class CumulativeMethod(BaseModel):
    """Initialize the CumulativeMethod.

    This method initializes the model and sets up parameters for fitting a cumulative event model.
    The fitting process starts with a 1-event model and iteratively adds events based on the
    convergence of the expectation maximization algorithm.

    Parameters
    ----------
    pattern :
        The pattern and properties to use for cross-correlation. Default is
        half sine with 50 ms width.
    location : float, optional
        How much milliseconds should be censored in the EM() step of model fitting.
        Default is width of the event.
        Shorter values than the width of a pattern allow overlap of neighboring events
        but might result in the same event being duplicated in several events.
        Larger values will prevent duplication at the risk of missing neighboring events
        Defaults to width of pattern, which is by default 50 ms.
    step : float, optional
        The size of the step from 0 to the mean RT. Defaults to `location`.
        Small values ensure a complete exploration of the parameter space but can be slow.
        Higher values fasten the estimation but risk missing event due to unexplored spaces.
    end : int, optional
        The maximum time to explore for new solutions. Defaults to the mean duration in the data.
    sequential: bool
        If True (Default), iteratively test all samples from 0 to `end`, retain times at
        which likelihood increased regardless of whether a subsequent event was found.
        If False, Testing new starting point solely starting from the last detected event.
    fastforward : bool, optional
        If True when proposal got rejected, start again with the furthest time point explored
        with previous proposition.
    tolerance : float, optional
        The tolerance used for convergence in the EM() function for the cumulative step.
        Defaults to 1e-4.
    base_fit: EventModel
        To start adding events from a specfic model, this argument can
        be provided with a fitted EventModel. Defaults to None.
    max_n_events: int
        Maximum number of events to be estimated. If None (default) uses the minim RT to estimated
        the maximim possible number of events.
    distribution : str
        Probability distribution for the by-trial onset of stages can be
        one of 'gamma','lognormal','wald', or 'weibull'
    """

    def __init__(
        self,
        pattern: Pattern = None,
        location: float = None,
        step: float = None,
        end: int = None,
        sequential: bool = True,
        fastforward: bool = True,
        tolerance: float = 1e-4,
        base_fit: EventModel | None = None,
        max_n_events: int | None = None,
        distribution: Any = None
    ):
        super().__init__(pattern, distribution)
        if location is None:
            location = self.pattern.width
        self.location = location
        self.step = step
        self.end = end
        self.sequential = sequential
        self.fastforward = fastforward
        self.tolerance = tolerance
        self.base_fit = base_fit
        self.max_n_events = max_n_events
        self.submodels = []

    def fit(# noqa: PLR0912, PLR0915
        self,
        data: PatternData | BaseTransformer | xr.DataArray,
        verbose: bool = True,
        kfold: int = 1,
        cpus: int = 1,

    ) -> None:
        """
        Fit the model starting with a 1-event model and iteratively add events.

        This method fits the cumulative event model to the provided pattern data. It begins with a
        single-event model and incrementally adds events based on the convergence of the expectation
        maximization algorithm. The process continues until the maximum number of events (given the
        minimum duration) is reached or the likelihood no longer improves.

        Parameters
        ----------
        data : Data to fit the model on. One of two options:
            1. data from BaseTransformer or xr.DataArray containing transformed data.
            2. PatternData object.
            In case of option 1, data is cross-correlated with the pattern in self.pattern.
        verbose : bool, optional
            If True, provides detailed output about the fitting process. Defaults to True.
        cpus : int, optional
            The number of CPU cores to use for computation. Defaults to 1.
        kfold: float
            Number of folds in a k-fold scheme to use for the estimation of new events. If kfold > 1
            performs crossvalidation on deterministically shuffled data.

        Returns
        -------
        None
        """
        pattern_data = self._instantiate_data_pattern(data)

        end = pattern_data.durations.values.mean() if self.end is None else self.end
        self.step = self.location*pattern_data.sfreq/1000 if self.step is None else self.step
        if self.max_n_events is None:
            max_n_events = int(np.floor(np.min(pattern_data.durations.values) /\
                                        (self.location*pattern_data.sfreq/1000))) + 1
        else:
            max_n_events = self.max_n_events
        #stop when not possible to insert event
        end = int(np.rint((end - self.location*pattern_data.sfreq/1000 - 1)/self.step))

        pbar = tqdm(total=end)  # progress bar
        n_events, j = 1, 1 # j = sample after last placed event

        # final time/chan parameters
        time_pars = np.zeros((end, 2))
        time_pars[:, 0] = self.distribution.shape
        # Initialize last stage of n=1
        time_pars[0, 1] = self.distribution.mean_to_scale(pattern_data.durations.values.mean())
        channel_pars = np.zeros((end, pattern_data.cross_corr.shape[1]))
        llk_prev = np.repeat(-np.inf, kfold)

        if self.base_fit is not None :
            n_events = self.base_fit.n_events+1
            time_pars[:n_events] = self.base_fit.time_pars.copy()
            channel_pars[:n_events-1] = self.base_fit.channel_pars.copy()
            llk_prev = self.base_fit.transform(pattern_data)[0]

        # Iterative fit
        while j < end and n_events <= max_n_events:
            prev_j = j
            event_model = EventModel(n_events=n_events, pattern=self.pattern,
                                     tolerance=self.tolerance,distribution=self.distribution)
            # get new parameters
            j, channel_pars_props, time_pars_props = self._propose_fit_params(
                n_events, j, channel_pars, time_pars
            )
            # Estimate model based on these propositions
            channel_pars_res, time_pars_res, llk, max_scale = self._fit_proposition(
                 pattern_data, n_events, channel_pars_props, time_pars_props, cpus, kfold
            )
            # check solution
            diff_llk = llk - llk_prev
            if all(llk_prev != -np.inf):
                diff_llk /= np.abs(llk_prev)

            if np.median(diff_llk) > self.tolerance:  # accept solution if likelihood improved
                llk_prev = llk

                # update channel_pars, params,
                channel_pars[:n_events] = channel_pars_res
                time_pars[: n_events + 1] = time_pars_res

                if verbose:
                    # Just to track advancement
                    events_so_far = [int(np.round(self.distribution.scale_to_mean(x))
                                         *(1000/pattern_data.sfreq))
                                         for x in
                                     np.cumsum(time_pars[:n_events, 1])
                    ]
                    print(
                        f"{n_events} events found around times "
                        f"{events_so_far}"
                    )
                # Search for additional event
                n_events += 1
                if self.sequential:
                    j += 1
            elif self.fastforward:
                # If ffwd, the next sample tested follows the max explored time for the last event
                max_sample = int(np.round(self.distribution.scale_to_mean(max_scale)))
                j = np.max([max_sample, (j + 1) * self.step]) / self.step
            else:
                j += 1
            pbar.update(int(np.rint(j-prev_j)))
        pbar.update(int(np.rint(end -j)+1))

        # done estimating
        n_events = n_events - 1
        if n_events > 0:
            self._fitted = True
            event_model = EventModel(pattern=self.pattern, distribution=self.distribution,
                                     location=self.location,
                                     tolerance=self.tolerance, n_events=n_events)
            event_model.fit(
                pattern_data,
                channel_pars=np.array([[channel_pars[:n_events, :]]]),
                time_pars=np.array([[time_pars[: n_events + 1, :]]]),
                verbose=False,
                cpus=1,
            )
            self.submodels.append(event_model)
        else:
            warn("Failed to find more than two stages, returning None")
            self._fitted = False

    def transform(self, *args, **kwargs):
        """
        Transform the input data using the last model fitted in the cumulative method.

        This method applies the transformation defined by the final model to the provided data.

        Returns
        -------
        Transformed data as returned by the final model's transform method.

        """
        self._check_fitted("transform data")
        return self.submodels[-1].transform(*args, **kwargs)

    def _fit_proposition(self, pattern_data, n_events,
                         channel_pars_props, time_pars_props,
                         cpus, kfold):

        event_model = EventModel(pattern=self.pattern, distribution=self.distribution,
                                 location=self.location,
                                 tolerance=self.tolerance, n_events=n_events)
        if kfold > 1:
            folds = list(pseudo_kfold(pattern_data, kfold))

            results = Parallel(n_jobs=cpus)(
                delayed(self.run_fold)(n_events, train_td, test_td,
                                       channel_pars_props, time_pars_props)
                for train_td, test_td in folds
            )
            llk, channel_pars_res, time_pars_res, max_scale = zip(*results)
            llk = np.array(llk)
            channel_pars_res = np.median(np.array(channel_pars_res), axis=0)
            time_pars_res = np.median(np.array(time_pars_res), axis=0)
            max_scale = np.median(max_scale)
        else:
            event_model.fit(
                pattern_data,
                np.array([channel_pars_props]),
                np.array([time_pars_props]),
                verbose=False,
                cpus=cpus,
            )
            channel_pars_res = event_model.channel_pars
            time_pars_res = event_model.time_pars
            llk = event_model.lkhs
            max_scale = np.max(
                [np.sum(x[0, :n_events-1, 1]) for x in event_model.time_pars_dev]
            )
        return channel_pars_res, time_pars_res, llk, max_scale

    def _propose_fit_params(self, n_events, j, channel_pars, time_pars):

        if (
            self.sequential and n_events > 1
        ):  # go through the whole range sample-by-sample, j is sample since start
            scale_j = self.distribution.mean_to_scale(self.step * j)

            # New parameter proposition
            time_pars_props = time_pars[:n_events].copy()  # time_pars so far
            # look between which event the next proposition should go
            n_event_j = np.argwhere(scale_j >= np.cumsum(time_pars_props[:, 1])) + 2
            n_event_j = np.max(n_event_j) if len(n_event_j) > 0 else 1
            n_event_j = np.min([n_event_j, n_events])  # do not insert even after last stage

            # insert j at right spot, subtract prev scales
            time_pars_props = np.insert(
                time_pars_props,
                n_event_j - 1,
                [self.distribution.shape,
                    scale_j - np.sum(time_pars_props[: n_event_j - 1, 1])],
                axis=0,
            )
            # subtract inserted scale from next event
            time_pars_props[n_event_j, 1] = (time_pars_props[n_event_j, 1]
                                             - time_pars_props[n_event_j - 1, 1])
            channel_pars_props = np.zeros((1, n_events, channel_pars.shape[-1]))  # always 0

            channel_pars_props[:, : n_events - 1, :] = np.tile(
                channel_pars[: n_events - 1, :], (len(channel_pars_props), 1, 1)
            )
            # shift new event to correct position
            channel_pars_props = np.insert(
                channel_pars_props[:, :-1, :], n_event_j - 1, channel_pars_props[:, -1, :], axis=1
            )

            j = self.distribution.scale_to_mean(np.sum(time_pars_props[:n_event_j, 1]))/self.step
        else:
            time_pars_props = time_pars[: n_events + 1].copy()
            time_pars_props[n_events,1] = time_pars_props[n_events-1,1]
            # New parameter proposition for the new event based on previous run
            new_event_prop = (np.max([self.distribution.mean_to_scale(j * self.step) -
                np.sum(time_pars_props[:n_events-1, 1]),self.distribution.mean_to_scale(self.step)])
            )
            time_pars_props[n_events-1, 1] = new_event_prop
            # Subtract new proposition from last stage
            time_pars_props[-1, 1] -= new_event_prop
            # Add a neutral event as new proposition
            channel_pars_props = np.zeros((1, n_events, channel_pars.shape[-1]))
            channel_pars_props[:, :n_events-1, :] = channel_pars[:n_events-1]
            j = self.distribution.scale_to_mean(np.sum(time_pars_props[:n_events, 1]))/self.step

        time_pars_props[:, 1] = np.maximum(time_pars_props[:, 1],
                               self.distribution.mean_to_scale(1))

        return j, channel_pars_props, np.array([time_pars_props])

    def __getattribute__(self, attr):
        property_list = {
            "xrtraces": "get traces",
            "xrlikelihoods": "get likelihoods",
            "xrtime_pars_dev": "get dev time time_pars",
            "xrchannel_pars": "get xrchannel_pars",
            "xrtime_pars": "get xrtime_pars"
        }
        if attr in property_list:
            self._check_fitted(property_list[attr])
            return getattr(self.submodels[-1], attr)
        return super().__getattribute__(attr)

    def run_fold(self, n_events, train_td, test_td, channel_pars_props, time_pars_props):
        event_model = EventModel(pattern=self.pattern, distribution=self.distribution,
                                 location=self.location,
                                 tolerance=self.tolerance, n_events=n_events)

        event_model.fit(
            train_td,
            np.array([channel_pars_props]),
            np.array([time_pars_props]),
            verbose=False,
            cpus=1,
        )

        llk = event_model.transform(test_td)[0].sum()
        max_scale = np.max(
                    [np.sum(x[0, :n_events-1, 1]) for x in event_model.time_pars_dev]
                )
        return llk, event_model.channel_pars, event_model.time_pars, max_scale
