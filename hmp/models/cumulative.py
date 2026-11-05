"""Models to estimate cumulative event models."""

from typing import Any
from warnings import warn

import numpy as np
import xarray as xr

from hmp.basedata import BaseData
from hmp.crossvalidation.pseudo_kfold import KFoldCV
from hmp.models.base import BaseModel
from hmp.models.event import EventModel
from hmp.patterndata import PatternData
from hmp.patterns import Pattern

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
        Default is width of the event, which is by default 50 ms.
        Shorter values than the width of a pattern allow overlap of neighboring events
        but might result in the same event being duplicated in several events.
        Larger values will prevent duplication at the risk of missing neighboring events
    step : float, optional
        The size of the step (in milliseconds) from 0 to the mean RT in milliseconds. Defaults to `location`.
        Small values ensure a complete exploration of the parameter space but can be slow.
        Higher values fasten the estimation but risk missing event due to unexplored spaces.
    end : int, optional
        The maximum time (in milliseconds) to explore for new solutions. Defaults to the mean duration in the data.
    sequential: bool
        If True, iteratively test all samples from 0 to `end`, retain times at
        which likelihood increased regardless of whether a subsequent event was found.
        If False (Default), Testing new starting point solely starting from the last detected event.
    tolerance : float, optional
        The tolerance used for convergence in the EM() function for the cumulative step.
        Defaults to 1e-4.
    kfold: float
        Number of folds in a k-fold scheme to use for the estimation of new events. If kfold > 1
        performs crossvalidation on deterministically shuffled data.
    lazy: bool = True,
        Whether to just test the starting point on left-out data or actually fit it
    max_events: int
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
        tolerance: float = 1e-4,
        consensus: float = .5,
        kfold: int = 5,
        lazy: bool = True,
        sequential: bool = True,
        max_events: int | None = None,
        distribution: Any = None
    ):
        super().__init__(pattern, distribution)
        if location is None:
            location = self.pattern.width
        self.location = location
        self.step = step
        self.end = end
        self.tolerance = tolerance
        self.consensus = consensus
        self.kfold = kfold
        self.lazy = lazy
        self.sequential = sequential
        self.max_events = max_events
        self.submodels = []

    def fit(
        self,
        data: PatternData | BaseData | xr.DataArray,
        base_fit: EventModel | None = None,
        verbose: bool = True,
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
            1. data from BasePreprocessor or xr.DataArray containing preprocessed data.
            2. PatternData object.
            In case of option 1, data is cross-correlated with the pattern in self.pattern.
        base_fit: EventModel
            To start adding events from a specfic model, this argument can
            be provided with a fitted EventModel. Defaults to None.
        verbose : bool, optional
            If True, provides detailed output about the fitting process. Defaults to True.
        cpus : int, optional

        Returns
        -------
        None
        """
        pattern_data = self._instantiate_data_pattern(data)
    
        if self.kfold > 1:
            cv = KFoldCV(self.kfold)
            cv.prepare(pattern_data)
        else:
            cv = None

        loc_samples = self._time_to_samples(self.location, pattern_data.sfreq)
        if self.step is None:
            step = loc_samples / 2
        else:
            step = self._time_to_samples(self.step, pattern_data.sfreq)
        
        if self.max_events is None:
            max_events = self._compute_max_events(pattern_data, self.location)
        else:
            max_events = self.max_events
        end = pattern_data.durations.values.mean() if self.end is None else self.end
        end = int(np.rint(end / step))

        pbar = tqdm(total=end)  # progress bar
        n_events, j = 1, 0 # j = sample after last placed event
        tolerance=self.tolerance
        
        # matrices time/chan parameters
        time_pars = np.zeros((end, 2))
        time_pars[:, 0] = self.distribution.shape
        
        # Initialize last stage of n=1
        time_pars[0, 1] = self.distribution.mean_to_scale(pattern_data.durations.values.mean())
        channel_pars = np.zeros((end, pattern_data.cross_corr.shape[1]))
        llk_prev = -np.inf

        if base_fit is not None :
            n_events = base_fit.n_events+1
            time_pars[:n_events] = base_fit.time_pars.copy()
            channel_pars[:n_events-1] = base_fit.channel_pars.copy()
            llk_prev = base_fit.transform(pattern_data)[0] #TODO pbm when using CV! llk must be by trial
            j = self.distribution.scale_to_mean(np.sum(time_pars[:n_events, 1]))/step

        llk_improvement, llk_timecourse = [], []
        # Iterative fit
        while j < end and n_events <= max_events:
            prev_j = j

            # get new parameters
            j, channel_pars_props, time_pars_props = self._propose_fit_params(
                n_events, j, channel_pars, time_pars, step
            )
            if self.lazy:
                tolerance=1
            # # Estimate model based on these propositions
            channel_pars_res, time_pars_res, llk = self._fit_proposition(
                 pattern_data, n_events, channel_pars_props, time_pars_props, cpus, cv, tolerance
            )
            llk_improvement.append(np.mean(llk - llk_prev > 0))
            llk_timecourse.append(llk)
            if llk_improvement[-1] > self.consensus:
                
                if self.lazy:
                    # Re-estimate good starting points using provided tolerance
                    channel_pars_res, time_pars_res, llk = self._fit_proposition(
                         pattern_data, n_events, channel_pars_res, time_pars_res, cpus, cv, self.tolerance
                    )
                llk_prev = llk
                
                # update channel_pars, params,
                channel_pars[:n_events] = channel_pars_res.copy()
                time_pars[: n_events + 1] = time_pars_res.copy()

                if verbose:
                    # Just to track advancement
                    events_so_far = [int((np.round(x))
                                         *(1000/pattern_data.sfreq))
                                         for x in
                                     np.cumsum(self.distribution.scale_to_mean(time_pars[:n_events, 1]))
                    ]
                    print(
                        f"{n_events} events found around times "
                        f"{events_so_far}"
                    )
                # Search for additional event
                n_events += 1
            else:
                j += 1
            pbar.update(int(np.rint(j-prev_j)))

        pbar.update(1)
        # done estimating
        n_events = n_events - 1
        if n_events == max_events and verbose:
            warn("Cumulative reached maximum possible number of event")
        if n_events > 0:
            self._fitted = True
            self.diagnostic = {}
            self.diagnostic["llk_improvement"] = np.array(llk_improvement)
            self.diagnostic["llk_timecourse"] = np.array(llk_timecourse)
            event_model = EventModel(n_events=n_events, pattern=self.pattern,
                                     location=self.location, tolerance=self.tolerance,
                                     distribution=self.distribution)
            event_model.fit(
                pattern_data,
                channel_pars=channel_pars[:n_events, :],
                time_pars=time_pars[: n_events + 1, :],
                verbose=False,
                cpus=cpus
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
                         cpus, cv, tolerance):

        event_model = EventModel(n_events=n_events, pattern=self.pattern, location=self.location, 
                                 tolerance=tolerance, distribution=self.distribution)
        if cv is not None:
            llk, train_models = cv.fit(event_model, None, channel_pars_props, time_pars_props, cpus)
            evprobs = cv.get_eventprobs(source='train').mean('fold').dropna('trial', how='all').fillna(0)
            event_model.n_dims = pattern_data.cross_corr.shape[1]
            channel_pars_res, time_pars_res = event_model.get_channel_time_parameters_expectation(
                pattern_data, evprobs.values, np.arange(evprobs.sizes['trial'])
            )
            llk = np.concatenate(llk)
        else:
            event_model.fit(
                pattern_data,
                channel_pars_props,
                time_pars_props,
                verbose=False,
                cpus=cpus
            )
            channel_pars_res = event_model.channel_pars
            time_pars_res = event_model.time_pars
            llk = np.array([event_model.lkhs.sum()])
        return channel_pars_res, time_pars_res, llk

    def _propose_fit_params(self, n_events, j, channel_pars, time_pars, step):

        # New proposition to be inserted
        time_pars_props = time_pars[:n_events+1].copy()
        # Manipulate in sample space
        times = self.distribution.scale_to_mean(time_pars_props[:-1,1])
        # Times retained previously
        if not self.sequential:
            j = np.max([
                        np.sum(
                            times[:-1] # j is the sum of all times excluding last stage
                        ) / step + 1, # advance of a step compared to max prev ev.
                    j #previous rejection
                ])

        time_j = step * j
        
        # look between which event the next proposition should go (always last if not seq)
        event_j = np.argwhere(time_j >= np.cumsum(times)) + 1
        event_j = np.max(event_j) if len(event_j) > 0 else 0
        event_j = np.min([event_j, n_events-1])  # do not insert even after last stage

        # Time proposition
        scale_j_prop = np.max([j * step - np.sum(times[:event_j]), step])
        times = np.insert(times, event_j, scale_j_prop)
        
        # subtract inserted proposition from next event
        times[event_j+1] -= times[event_j]
        
        # Ensure valid durations
        times = np.maximum(times, 1)
        # Turn back to scale params
        time_pars_props[:, 1] = self.distribution.mean_to_scale(times)

        # Channel proposiiton
        channel_pars_props = channel_pars[:n_events-1,:].copy()
        # Insert 0 channel event
        channel_pars_props = np.insert(channel_pars_props, event_j, 0, axis=0)
        return j, channel_pars_props, time_pars_props

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
