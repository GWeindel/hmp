"""Definition of ``EventModel`` class.

The ``EventModel`` class is the base model for estimating
hidden multivariate pattern models.
"""

import multiprocessing as mp
from itertools import product
from typing import Any
from warnings import warn

import numpy as np
import xarray as xr

from hmp.basedata import BaseData
from hmp.estimators.base import BaseEstimator
from hmp.estimators.em import EMEstimator, worker_estim_probs
from hmp.models.base import BaseModel
from hmp.patterndata import PatternData
from hmp.patterns import Pattern


class EventModel(BaseModel):
    """
    A model for estimating HMP events.

    Parameters
    ----------
    n_events : int
        The number of HMP events to estimate.
    pattern :
        The pattern and properties to use for cross-correlation. Default is
        half sine with 50 ms width.
    location : int | np.ndarray, optional
        How many milliseconds should be censored in the EM() step of model fitting.
        Default is width of the pattern template, which by defaul is 50 ms.
        Shorter values than the width of a pattern allow overlap of neighboring events
        but might result in the same event being duplicated in several events.
        Larger values will prevent duplication at the risk of missing neighboring events.
        If array, should be length n_events+1, setting location for each event.
    fixed_time_pars : list, optional
        List of time parameters to fix during estimation.
        If None, all time parameters are estimated.
    fixed_channel_pars : list, optional
        List of channel parameters to fix during estimation.
        If None, all channel parameters are estimated.
    grouping_dict : dict, optional
        Dictionary defining groups for grouping modeling. Keys are group names,
        and values are lists of groups. If grouping_dict is provided, a channel_map
        and/or a time_map is also required, and vice versa.
        If one group, use a dict with the name in the metadata and a list of the
        levels in the same order as the rows of the maps. E.g., {'cue': ['SP', 'AC']}
        If multiple gropus need to be crossed, specify them as separate entries in
        the dictionary. E.g., {'cue': ['SP', 'AC',], 'resp': ['left', 'right']}.
        These are crossed by repeating the first condition as many times as there are
        levels in the second condition. E.g., SP-left, SP-right, AC-left, AC-right.
        Default is {}.
    channel_map : ndarray, optional
        2D ndarray (n_groups * n_events) indicating which channel contributions
        are shared between groups. Default is None.
    time_map : ndarray, optional
        2D ndarray (n_groups * n_stages) indicating which time parameters are
        shared between groups. Default is None.
    tolerance : float, optional
        Convergence tolerance for the expectation maximization algorithm. Default is 1e-4.
    max_iteration : int, optional
        Maximum number of iterations for the expectation maximization algorithm. Default is 1e3.
    min_iteration : int, optional
        Minimum number of iterations for the expectation maximization algorithm. Default is 1.
    starting_points : int, optional
        Number of random starting points to use for initialization. Default is 1.
    max_duration : float, optional
        Maximum mean distance between events, used when generating random starting points.
        Default is None.
    distribution : str
        Probability distribution for the by-trial onset of stages can be
        one of 'gamma','lognormal','wald', or 'weibull'
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        n_events: int,
        pattern: Pattern = None,
        location: float | np.ndarray = None,
        fixed_time_pars: list = [],
        fixed_channel_pars: list = [],
        channel_map: np.ndarray = None,
        time_map: np.ndarray = None,
        grouping_dict: dict = {},
        tolerance: float = 1e-4,
        max_iteration: int = 1e3,
        min_iteration: int = 1,
        starting_points: int = 1,
        max_duration: float = None,
        distribution: Any = None,
    ):
        assert np.issubdtype(type(n_events), np.integer), (
            f"An integer for the number of expected transition events"
            f" is expected, got {type(n_events).__name__} instead"
        )
        if grouping_dict != {} or channel_map is not None or time_map is not None:
            assert isinstance(grouping_dict, dict), "groups have to be specified as a dictionary"
            assert grouping_dict != {}, (
                "If time_map or channel_map is provided,a grouping_dict is required."
            )
            assert channel_map is not None or time_map is not None, (
                "If grouping_dict is provided, time_map or channel_map is required."
            )
            assert n_events == (time_map.shape[-1] - 1) or n_events == channel_map.shape[-1], (
                "n_events, time_map and channel_map must indicate same number of max events"
            )

            # already add group names
            group_names = []
            group_mods = []
            for group, mod in grouping_dict.items():
                group_names.append(group)
                group_mods.append(mod)
            group_mods = list(product(*group_mods))
            group_mods = np.array(group_mods, dtype=object)
            self.group_labels = (str(group_names), group_mods)
        else:
            self.group_labels = ("group all", np.array([[""]], dtype=object))

        super().__init__(pattern, distribution)
        self.n_events = n_events
        self._set_locations(location)
        self.n_dims = None
        self.fixed_time_pars = fixed_time_pars
        self.fixed_channel_pars = fixed_channel_pars
        self.tolerance = tolerance
        self.max_iteration = max_iteration
        self.min_iteration = min_iteration
        self.starting_points = starting_points
        self.max_duration = max_duration
        self.grouping_dict = grouping_dict
        self.time_map = np.zeros((1, self.n_events + 1)) if time_map is None else time_map
        self.channel_map = np.zeros((1, self.n_events)) if channel_map is None else channel_map

    def _set_locations(self, location):
        """Set minimum distance between successive events."""
        # array, must be array of length n_events + 1
        if location is not None and isinstance(location, np.ndarray) and len(location) > 0:
            assert len(location) == self.n_events + 1, (
                "If location is np:array, should have length n_events + 1."
            )
            self.locations = location
        else:
            self.locations = np.zeros(self.n_events + 1, dtype=int)
            if self.n_events > 1:
                self.locations[1:-1] = self.pattern.width if location is None else location

        if self.n_events > 1 and any(self.locations[1:-1] < np.round(self.pattern.width)):
            warn(
                "For n_event > 1, locations must be greater or equal than pattern.width"
            f" but received locations ({self.locations}) is smaller than ({self.pattern.width})."
            )

    def _format_parameters(  # noqa: PLR0912, PLR0915
        self, channel_pars, time_pars, groups, n_groups, durations, sfreq
    ):

        # Formatting parameters
        if isinstance(time_pars, (xr.DataArray, xr.Dataset)):
            time_pars = time_pars.dropna(dim="stage").values
        elif isinstance(time_pars, np.ndarray):
            time_pars = time_pars.copy()
        if isinstance(channel_pars, (xr.DataArray, xr.Dataset)):
            channel_pars = channel_pars.dropna(dim="event").values
        elif isinstance(channel_pars, np.ndarray):
            channel_pars = channel_pars.copy()

        if time_pars is None:
            # If no time parameters starting points are provided generate standard ones
            # Or random ones if starting_points > 1
            time_pars = (
                np.zeros((n_groups, self.n_events + 1, 2)) * np.nan
            )  # by default nan for missing stages
            for cur_group in range(n_groups):
                time_group = np.where(self.time_map[cur_group, :] >= 0)[0]
                n_stage_group = len(time_group)
                # by default starting point is to split the average duration in equal bins
                time_pars[cur_group, time_group, :] = np.tile(
                    [
                        self.distribution.shape,
                        self.distribution.mean_to_scale(
                            np.mean(durations.values[groups == cur_group]) / (n_stage_group)
                        ),
                    ],
                    (n_stage_group, 1),
                )

            initial_p = time_pars
            time_pars = [initial_p]

            # deal with multiple random starting pionts
            if self.starting_points > 1:
                if self.max_duration is None:
                    self.max_duration = durations.mean()
                for _ in np.arange(self.starting_points):
                    proposal_p = (
                        np.zeros((n_groups, self.n_events + 1, 2)) * np.nan
                    )  # by default nan for missing stages
                    for cur_group in range(n_groups):
                        time_group = np.where(self.time_map[cur_group, :] >= 0)[0]
                        n_stage_group = len(time_group)
                        proposal_p[cur_group, time_group, :] = self.gen_random_stages(
                            n_stage_group - 1, sfreq
                        )
                        proposal_p[cur_group, self.fixed_time_pars, :] = initial_p[
                            0, self.fixed_time_pars
                        ]
                    time_pars.append(proposal_p)
                time_pars = np.array(time_pars)

        elif time_pars.ndim < 4:
            # if 3 dims, and first dim is empty, might be groups
            # or starting points. Add groups to make sure and wrap
            # again for starting points.
            time_pars = np.squeeze(time_pars)
            if time_pars.ndim == 2:
                time_pars = np.array([np.tile(time_pars, (n_groups, 1, 1))])
            else:
                time_pars = np.array([time_pars])

            # set params missing stages to nan to make it obvious in the results
            if (self.time_map < 0).any():
                for c in range(n_groups):
                    time_pars[0, c, np.where(self.time_map[c, :] < 0)[0], :] = np.nan

        if channel_pars is None:
            # By defaults c_pars are initiated to 0
            channel_pars = np.zeros((n_groups, self.n_events, self.n_dims), dtype=np.float32)

        if channel_pars.ndim < 4:
            if channel_pars.ndim == 2:
                channel_pars = np.tile(channel_pars, (n_groups, 1, 1))

            if (self.channel_map < 0).any():  # set missing c_pars to nan
                for cur_group in range(n_groups):
                    channel_pars[cur_group, np.where(self.channel_map[cur_group, :] < 0)[0], :] = (
                        np.nan
                    )

            initial_m = channel_pars
            channel_pars = np.tile(initial_m, (self.starting_points, 1, 1, 1))

        return channel_pars, time_pars

    def fit(  # noqa: PLR0913, PLR0917
        self,
        data: BaseData | PatternData,
        channel_pars: np.ndarray = None,
        time_pars: np.ndarray = None,
        verbose: bool = True,
        cpus: int = 1,
        estimator: BaseEstimator = None,
    ):
        """
        Fit HMP for a single n_events model.

        Parameters
        ----------
        data : Data to fit the model on. One of two options:
            1. data from BasePreprocessor or xr.DataArray containing preprocessed data.
            2. PatternData object.
            In case of option 1, data is cross-correlated with the pattern in self.pattern.
        channel_pars : ndarray, optional
            3D ndarray (n_groups * n_events * n_channels) or
            4D (starting_points * n_groups * n_events * n_channels)
            initial conditions for event channel contributions. Default is None.
        time_pars : ndarray, optional
            3D ndarray (n_groups * n_stages * 2) or 4D (starting_points * n_groups * n_stages * 2)
            initial conditions for time distribution parameters. Default is None.
        verbose : bool, optional
            If True, displays output useful for debugging. Default is True.
        cpus : int, optional
            Number of cores to use in multiprocessing functions. Default is 1.
        estimator : BaseEstimator, optional
            Estimation method used to optimize the parameters. Defaults to
            :class:`~hmp.estimators.em.EMEstimator` built from the tolerance and
            iteration limits given to the model.

        Returns
        -------
        EstimationResult
            The full result of the estimation, also stored on the model as
            ``estimation_result``. Fitted parameters remain available through
            the usual attributes, so callers may ignore this return value.
        """
        pattern_data = self._instantiate_data_pattern(data)
        self.n_dims = pattern_data.cross_corr.shape[1]
        n_groups, groups, self.group_labels = self.group_constructor(
            pattern_data.durations, verbose
        )

        channel_pars, time_pars = self._format_parameters(
            channel_pars, time_pars, groups, n_groups, pattern_data.durations, pattern_data.sfreq
        )

        if verbose:
            if time_pars is None:
                print(
                    f"Estimating {self.n_events} events model with {self.starting_points} "
                    "starting point(s)"
                )
            else:
                print(f"Estimating {self.n_events} events model")

        if estimator is None:
            estimator = EMEstimator(
                tolerance=self.tolerance,
                max_iteration=self.max_iteration,
                min_iteration=self.min_iteration,
            )

        result = estimator.fit(self, pattern_data, channel_pars, time_pars, groups, cpus)

        self._fitted = True
        self.estimation_result = result
        self.lkhs = result.likelihood
        self.channel_pars = result.channel_pars
        self.time_pars = result.time_pars
        # Iteration traces are specific to iterative estimators; a sampler has none.
        self.traces = result.diagnostics.get("traces")
        self.traces_group = result.diagnostics.get("traces_group")
        self.time_pars_dev = result.diagnostics.get("time_pars_dev")
        self.group = groups
        return result

    def transform(self, data: Any, cpus: int = 1) -> tuple[np.ndarray, xr.DataArray]:
        """
        Transform the trial data using the fitted model.

        Parameters
        ----------
        data : Data to fit the model on. One of two options:
            1. data from BasePreprocessor or xr.DataArray containing preprocessed data.
            2. PatternData object.
            In case of option 1, data is cross-correlated with the pattern in self.pattern.
        cpus : int, nr of cpus to use

        Returns
        -------
        likelihoods : list
            List of log-likelihoods for each submodel (number of events).
        xr_eventprobs : xr.DataArray
            Concatenated event probability arrays for all submodels, indexed by number of events.
        """
        pattern_data = self._instantiate_data_pattern(data)

        _, groups, _ = self.group_constructor(pattern_data.durations)
        likelihoods, xreventprobs = self._estim_probs_groups(
            pattern_data, self.channel_pars, self.time_pars, groups, cpus=cpus
        )

        return likelihoods, xreventprobs

    def event_probabilities(
        self,
        pattern_data: PatternData,
        channel_pars: np.ndarray,
        time_pars: np.ndarray,
        groups: np.ndarray = None,
        cpus: int = 1,
        pool: mp.Pool = None,
    ) -> xr.DataArray:
        """
        Event probabilities for an arbitrary set of parameters.

        Unlike :meth:`transform`, this does not require the model to be fitted and does
        not read the fitted parameters, so it can be evaluated at any point in parameter
        space. Sampling-based estimators use it to obtain the by-trial event
        probabilities implied by each posterior draw.

        Parameters
        ----------
        pattern_data : PatternData
            Preprocessed data cross-correlated with the pattern of the model.
        channel_pars : np.ndarray
            3D ndarray (n_groups * n_events * n_channels) of channel contributions.
        time_pars : np.ndarray
            3D ndarray (n_groups * n_stages * 2) of time distribution parameters.
        groups : np.ndarray, optional
            Array indicating the groups for grouping modeling. Default is None, in which
            case groups are derived from the durations.
        cpus : int, optional
            Number of cores to use in multiprocessing functions. Default is 1.
        pool : mp.Pool, optional
            Multiprocessing pool to use for parallelization. Default is None.

        Returns
        -------
        xr.DataArray
            Probabilities with dimensions ("trial", "sample", "event"). The summed,
            per-group and per-trial log-likelihoods are attached as the ``likelihood``,
            ``group_lkh`` and ``trial_lkh`` attributes.
        """
        if groups is None:
            _, groups, _ = self.group_constructor(pattern_data.durations)
        _, eventprobs = self._estim_probs_groups(
            pattern_data, channel_pars, time_pars, groups, cpus=cpus, pool=pool
        )
        return eventprobs

    def log_likelihood(  # noqa: PLR0913, PLR0917
        self,
        pattern_data: PatternData,
        channel_pars: np.ndarray,
        time_pars: np.ndarray,
        groups: np.ndarray = None,
        cpus: int = 1,
        per_trial: bool = False,
    ) -> float | np.ndarray:
        """
        Log-likelihood of the data under the given parameters.

        This is the model's contract with an estimator: everything an estimation method
        needs in order to score a set of parameters, without reaching into the model's
        internals.

        Parameters
        ----------
        pattern_data : PatternData
            Preprocessed data cross-correlated with the pattern of the model.
        channel_pars : np.ndarray
            3D ndarray (n_groups * n_events * n_channels) of channel contributions.
        time_pars : np.ndarray
            3D ndarray (n_groups * n_stages * 2) of time distribution parameters.
        groups : np.ndarray, optional
            Array indicating the groups for grouping modeling. Default is None, in which
            case groups are derived from the durations.
        cpus : int, optional
            Number of cores to use in multiprocessing functions. Default is 1.
        per_trial : bool, optional
            If True, return the log-likelihood of each trial rather than their sum.
            Pointwise values are required for model comparison such as LOO or WAIC,
            which is how sampling-based fits compare numbers of events.

        Returns
        -------
        float or np.ndarray
            Summed log-likelihood, or a 1D array of length n_trials if ``per_trial``.
        """
        eventprobs = self.event_probabilities(
            pattern_data, channel_pars, time_pars, groups, cpus=cpus
        )
        return eventprobs.trial_lkh if per_trial else eventprobs.likelihood

    @property
    def xrtraces(self):
        """
        Returns the traces of the log-likelihood for each EM iteration as an xarray DataArray.

        Returns
        -------
        xr.DataArray
            An xarray DataArray with dimensions ("em_iteration", "group") containing
            the log-likelihood traces.
        """
        self._check_fitted("get traces")
        return xr.DataArray(
            self.traces_group,
            dims=("em_iteration", "group"),
            name="traces",
            coords={
                "em_iteration": range(self.traces_group.shape[0]),
                "group": range(self.traces_group.shape[1]),
            },
        )

    @property
    def xrlikelihoods(self):
        """
        Returns the log-likelihoods as an xarray DataArray.

        Returns
        -------
        xr.DataArray
            An xarray DataArray containing the log-likelihood values.
        """
        self._check_fitted("get likelihoods")
        return xr.DataArray(self.lkhs, name="loglikelihood")

    @property
    def xrtime_pars_dev(self):
        """
        Returns the time parameter for each EM iteration as an xarray DataArray.

        Returns
        -------
        xr.DataArray
            An xarray DataArray with dimensions ("em_iteration", "group", "stage", "time_pars")
            containing the time parameter deviations.
        """
        self._check_fitted("get dev time pars")
        return xr.DataArray(
            self.time_pars_dev,
            dims=("em_iteration", "group", "stage", "time_pars"),
            name="time_pars_dev",
            coords=[
                range(self.time_pars_dev.shape[0]),
                range(self.time_pars_dev.shape[1]),
                range(self.n_events + 1),
                ["shape", "scale"],
            ],
        )

    @property
    def xrtime_pars(self):
        """
        Returns the time parameters as an xarray DataArray.

        Returns
        -------
        xr.DataArray
            An xarray DataArray with dimensions ("group", "stage", "parameter") containing the time
            parameters.
        """
        self._check_fitted("get xrtime_pars")
        return xr.DataArray(
            self.time_pars,
            dims=("group", "stage", "parameter"),
            name="time_pars",
            coords={
                "group": range(self.time_pars.shape[0]),
                "stage": range(self.n_events + 1),
                "parameter": ["shape", "scale"],
            },
        )

    @property
    def xrchannel_pars(self):
        """
        Returns the channel parameters as an xarray DataArray.

        Returns
        -------
        xr.DataArray
            An xarray DataArray with dimensions ("group", "event", "channel") containing the channel
            parameters.
        """
        self._check_fitted("get xrchannel_pars")
        return xr.DataArray(
            self.channel_pars,
            dims=("group", "event", "channel"),
            name="channel_pars",
            coords={
                "group": range(self.channel_pars.shape[0]),
                "event": range(self.n_events),
                "channel": range(self.n_dims),
            },
        )

    def get_channel_time_parameters_expectation(
        self, pattern_data: PatternData, eventprobs: np.ndarray, subset_epochs: list[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the channel and time parameters using the expectation step.

        Parameters
        ----------
        pattern_data : PatternData
            Preprocessed data cross-correlated with the pattern of the model
        eventprobs : np.ndarray
            A 3D array of shape (n_trials, max_duration, n_events) containing the event
            probabilities.
        subset_epochs : list[int], optional
            A list of trial indices to consider for the computation. If None, all trials are used.

        Returns
        -------
        channel_pars : np.ndarray
            A 2D array of shape (n_events, n_dims) with the estimated channel parameters.
        time_pars : np.ndarray
            A 2D array of shape (n_stages, 2) with the estimated time parameters (shape and scale).
        """
        # Channel contribution from Expectation, Eq 11 from 2024 paper
        max_dur = int(np.max(pattern_data.durations.values[subset_epochs]))
        # padded (n_sub, max_dur, n_dims) data for this subset, sized like eventprobs
        data = np.zeros(
            (len(subset_epochs), max_dur, self.n_dims), dtype=pattern_data.cross_corr.dtype
        )
        for trial_idx, trial in enumerate(subset_epochs):
            start, end = pattern_data.starts[trial], pattern_data.ends[trial]
            data[trial_idx, : end - start + 1, :] = pattern_data.cross_corr[start : end + 1, :]

        ep = eventprobs[subset_epochs]  # (n_sub, max_dur, n_events)
        # mean over trials of sum over samples: (n_events, n_dims)
        channel_pars = np.einsum("tse,tsc->ec", ep, data).astype(np.float64) / len(subset_epochs)

        # Time parameters from Expectation Eq 10 from 2024 paper
        # calc averagepos here as mean_d can be group dependent, whereas scale_parameters() assumes
        # it's general
        event_times_mean = np.concatenate(
            [
                np.arange(np.max(pattern_data.durations.values[subset_epochs]))
                @ eventprobs[subset_epochs].mean(axis=0),
                [np.mean(pattern_data.durations.values[subset_epochs]) - 1],
            ]
        )
        time_pars = self.scale_parameters(averagepos=event_times_mean)
        return channel_pars, time_pars

    def gen_random_stages(self, n_events: int, sfreq) -> np.ndarray:
        """
        Compute random stage durations.

        Generates random stage durations between 0 and the mean reaction time (RT) by iteratively
        drawing samples from a uniform distribution. The last stage duration is computed as
        1 minus the cumulative duration of previous stages.
        The stages are then scaled to the mean RT.

        Parameters
        ----------
        n_events : int
            The number of events to generate random durations for.

        Returns
        -------
        np.ndarray
            A 2D array where each row contains the shape and scale parameters for a stage.
        """
        rnd_durations = np.zeros(n_events + 1)
        while any(rnd_durations < max(self._time_to_samples(self.locations, sfreq))):
            rnd_events = np.random.default_rng().integers(
                low=0, high=self.max_duration, size=n_events
            )  # n_events between 0 and mean_d
            rnd_events = np.sort(rnd_events)
            rnd_durations = np.hstack((rnd_events, self.max_duration)) - np.hstack(
                (0, rnd_events)
            )  # associated durations
        random_stages = np.array(
            [[self.distribution.shape, self.distribution.mean_to_scale(x)] for x in rnd_durations]
        )
        return random_stages

    def scale_parameters(self, averagepos: np.ndarray) -> np.ndarray:
        """
        Scale parameters from the average position of events.

        This method is used during the re-estimation step in the EM procedure.
        It computes the likeliest location of events from `eventprobs` and calculates
        the scale parameters as the average distance between consecutive events.

        Parameters
        ----------
        averagepos : np.ndarray
            A 1D array containing the average positions of events.

        Returns
        -------
        np.ndarray
            A 2D array where each row contains the shape and scale parameters
            for the corresponding event distribution.
        """
        params = np.zeros((len(averagepos), 2), dtype=np.float32)
        params[:, 0] = self.distribution.shape
        params[:, 1] = np.diff(averagepos, prepend=0)
        params[:, 1] = self.distribution.mean_to_scale(params[:, 1])
        return params

    def estim_probs(
        self,
        pattern_data: PatternData,
        channel_pars: np.ndarray,
        time_pars: np.ndarray,
        subset_epochs: list[int] | None = None,
        max_duration: int | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Estimate probabilities for events and compute the log-likelihood.

        Parameters
        ----------
        pattern_data : PatternData
            Preprocessed data cross-correlated with the pattern of the model
        channel_pars : np.ndarray
            A 2D array of shape (n_events, n_channels) or a 3D array of shape
            (iteration, n_events, n_channels) containing initial conditions for
            channel contributions to events.
        time_pars : np.ndarray
            A 2D array of shape (n_stages, n_parameters) or a 3D array of shape
            (iteration, n_stages, n_parameters) containing initial conditions for
            the distribution parameters.
        subset_epochs : list[int] or None, optional
            A list of trial indices to consider for the computation. If None, all trials
            are used. Default is None.

        Returns
        -------
        loglikelihood : float
            The summed log probabilities.
        eventprobs : np.ndarray
            A 3D array of shape (n_trials, max_samples, n_events) containing the probabilities
            for each event.
        trial_loglikelihood : np.ndarray
            A 1D array of length n_trials with the log probability of each trial, before
            summing. Needed for pointwise model comparison such as LOO/WAIC.
        """
        n_events = channel_pars.shape[0]
        n_stages = n_events + 1
        if subset_epochs is not None:
            if len(subset_epochs) == len(pattern_data.starts):  # boolean indices
                subset_epochs = np.where(subset_epochs)[0]
        n_trials = len(subset_epochs)
        starts = pattern_data.starts[subset_epochs]
        ends = pattern_data.ends[subset_epochs]
        durations = ends - starts + 1
        cross_corr = np.vstack([pattern_data.cross_corr[s : e + 1] for s, e in zip(starts, ends)])
        dtype = pattern_data.cross_corr.dtype
        if max_duration is None:
            max_duration = np.max(durations)
        gains = np.zeros((cross_corr.shape[0], n_events), dtype=dtype)
        for i in range(cross_corr.shape[1]):
            # computes the gains, i.e. congruence between the pattern shape
            # and the data given the magnitudes of the sensors
            gains = (
                gains
                + cross_corr[:, i][np.newaxis].T * channel_pars[:, i]
                - channel_pars[:, i] ** 2 / 2
            )
        gains = np.exp(gains)
        probs = np.zeros([max_duration, n_trials, n_events], dtype=dtype)  # prob per trial
        probs_b = np.zeros(
            [max_duration, n_trials, n_events], dtype=dtype
        )  # Sample and state reversed

        trial_slice = np.cumsum(durations)
        for trial in np.arange(n_trials):
            # Following assigns gain per trial to variable probs
            probs[: durations[trial], trial, :] = gains[
                trial_slice[trial] - durations[trial] : trial_slice[trial], :
            ]
            # Same but sample and events are reversed, this allows to compute
            # fwd and bwd in the same way in the following steps
            probs_b[: durations[trial], trial, :] = probs[: durations[trial], trial, :][::-1, ::-1]

        pmf = self.location_pdf(time_pars, max_duration, n_stages, pattern_data.sfreq, dtype)
        pmf_b = pmf[:, ::-1]  # Stage reversed gamma pmf, same order as prob_b

        forward = np.zeros((max_duration, n_trials, n_events), dtype=dtype)
        backward = np.zeros((max_duration, n_trials, n_events), dtype=dtype)
        # Computing forward and backward helper variable
        #  when stage = 0:
        forward[:, :, 0] = (
            np.tile(pmf[:, 0][np.newaxis].T, (1, n_trials)) * probs[:, :, 0]
        )  # first stage transition is p(B) * p(d)
        backward[:, :, 0] = np.tile(
            pmf_b[:, 0][np.newaxis].T, (1, n_trials)
        )  # Reversed gamma (i.e. last stage) without probs as last event ends at time T

        for event in np.arange(
            1, n_events
        ):  # Following stage transitions integrate previous transitions
            add_b = backward[:, :, event - 1] * probs_b[:, :, event - 1]  # Next stage in back
            for trial in np.arange(n_trials):
                # convolution between gamma * gains at previous event and event
                forward[:, trial, event] = np.convolve(forward[:, trial, event - 1], pmf[:, event])[
                    :max_duration
                ]
                # same but backwards
                backward[:, trial, event] = np.convolve(add_b[:, trial], pmf_b[:, event])[
                    :max_duration
                ]
            forward[:, :, event] = forward[:, :, event] * probs[:, :, event]
        # re-arranging backward to the expected variable
        backward = backward[:, :, ::-1]  # undoes stage inversion
        for trial in np.arange(n_trials):  # Undoes sample inversion
            backward[: durations[trial], trial, :] = backward[: durations[trial], trial, :][::-1]
        eventprobs = forward * backward
        eventprobs = np.clip(eventprobs, 0, None)  # floating point precision error
        trial_likelihood = np.log(
            eventprobs[:, :, 0].sum(axis=0)
        )  # sum over max_samples to avoid 0s in log
        likelihood = np.sum(trial_likelihood)
        eventprobs = eventprobs / eventprobs.sum(axis=0)
        eventprobs[np.isnan(eventprobs)] = 0
        eventprobs = eventprobs.transpose((1, 0, 2))
        return [likelihood, eventprobs, trial_likelihood]

    def chunked_estim_probs(
        self,
        pattern_data,
        channel_pars,
        time_pars,
        subset_epochs=None,
        cpus=1,
        pool=None,
    ):
        """Chunk estim_probs across trial-chunks and reassemble.

        Returns [likelihood, eventprobs, trial_likelihood]

        Parameters
        ----------
        pattern_data, channel_pars, time_pars, subset_epochs :
            Passed through to estim_probs.
        cpus : int
            Number of trial-chunks to split into. cpus=1 reproduces the single call exactly
            (one chunk == the whole subset).
        pool : multiprocessing.Pool or None
            If given and cpus > 1, chunks run via pool.starmap. If None, chunks run serially
            in-process
        """
        # estim_probs accepts a boolean mask of full length; normalize to indices
        # here so chunk slicing and reassembly both work on positional indices.
        if subset_epochs is None:
            subset_epochs = np.arange(len(pattern_data.starts))
        if cpus <= 1 or pool is None:
            return self.estim_probs(
                pattern_data, channel_pars, time_pars, subset_epochs=subset_epochs
            )

        subset_epochs = np.asarray(subset_epochs)
        if subset_epochs.dtype == bool:
            subset_epochs = np.where(subset_epochs)[0]

        n_trials = len(subset_epochs)
        n_events = channel_pars.shape[0]
        full_max_dur = int(np.max(pattern_data.durations.values[subset_epochs]))

        # split trials into disjoint chunks
        chunks = [c for c in np.array_split(subset_epochs, cpus) if len(c) > 0]

        # run estim_probs per chunk
        if pool is not None:
            results = pool.starmap(
                worker_estim_probs,
                [(channel_pars, time_pars, c, full_max_dur) for c in chunks],
            )
        else:
            results = [
                self.estim_probs(
                    pattern_data,
                    channel_pars,
                    time_pars,
                    subset_epochs=chunk,
                    max_duration=full_max_dur,
                )
                for chunk in chunks
            ]

        # reassemble
        # likelihood: sum over trials, so chunk log-likelihoods add exactly.
        likelihood = float(np.sum([r[0] for r in results]))

        # eventprobs: each chunk is (chunk_trials, full_max_dur, n_events)
        eventprobs = np.zeros((n_trials, full_max_dur, n_events), dtype=results[0][1].dtype)
        trial_likelihood = np.empty(n_trials, dtype=results[0][2].dtype)

        row = 0
        for _, chunk_ep, chunk_tll in results:
            c_trials = chunk_ep.shape[0]
            c_dur = chunk_ep.shape[1]
            eventprobs[row : row + c_trials, :c_dur, :] = chunk_ep
            trial_likelihood[row : row + c_trials] = chunk_tll
            row += c_trials

        return [likelihood, eventprobs, trial_likelihood]

    def _estim_probs_groups(
        self,
        pattern_data: PatternData,
        channel_pars: np.ndarray,
        time_pars: np.ndarray,
        groups: np.ndarray,
        cpus: int = 1,
        pool: mp.Pool = None,
    ) -> tuple[np.ndarray, xr.DataArray]:
        """
        Estimate probability groups for grouping models.

        This method computes the log-likelihood and event probabilities for each group
        in the grouping model, using the provided channel and time parameters.

        Parameters
        ----------
        pattern_data : PatternData
            Preprocessed data cross-correlated with the pattern of the model
        channel_pars : np.ndarray
            A 3D array of shape (groups, n_events, n_channels) containing
            initial channel contributions to events.
        time_pars : np.ndarray
            A 3D array of shape (n_groups, n_stages, n_parameters) containing
            initial distribution parameters.
        groups : np.ndarray
            An array indicating the groups for grouping modeling.
        cpus : int, optional
            Number of cores to use in multiprocessing functions. Default is 1.
        pool : mp.Pool, optional
            Multiprocessing pool to use for parallelization. Default is None.

        Returns
        -------
        loglikelihood : np.ndarray
            A 1D array of log-likelihood values for each group.
        all_xreventprobs : xr.DataArray
            An xarray DataArray containing event probabilities with dimensions
            ("trial", "sample", "event").
        """
        data_groups = np.unique(groups)
        likes_events_group = []

        for cur_group in data_groups:
            channel_pars_group = channel_pars[
                cur_group, self.channel_map[cur_group, :] >= 0, :
            ]  # select existing magnitudes
            # select existing params
            time_pars_group = time_pars[cur_group, self.time_map[cur_group, :] >= 0, :]
            likes_events_group.append(
                self.chunked_estim_probs(
                    pattern_data,
                    channel_pars_group,
                    time_pars_group,
                    subset_epochs=(groups == cur_group),
                    cpus=cpus,
                    pool=pool,
                )
            )

        likelihood = np.array([x[0] for x in likes_events_group])

        # Per-trial log-likelihood, scattered back into the original trial order
        trial_likelihood = np.zeros(len(pattern_data.durations))
        for cur_group in data_groups:
            trial_likelihood[groups == cur_group] = likes_events_group[cur_group][2]

        # all_xreventprobs must have same order as pattern_data because
        # subset_epochs is used later on eventprobs!
        all_xreventprobs = xr.DataArray(
            np.zeros(
                (len(pattern_data.durations), np.max(pattern_data.durations.values), self.n_events)
            ),
            dims=("trial", "sample", "event"),
            coords=pattern_data.durations.coords,
        )
        all_xreventprobs = all_xreventprobs.assign_coords(
            group=("trial", groups),
            event=("event", np.arange(self.n_events)),
            sample=("sample", range(np.max(pattern_data.durations.values))),
        )
        for cur_group in data_groups:
            all_xreventprobs.data[
                np.ix_(
                    groups == cur_group,
                    range(likes_events_group[cur_group][1].shape[1]),
                    self.channel_map[cur_group, :] >= 0,
                )
            ] = likes_events_group[cur_group][1]

        all_xreventprobs.attrs["sfreq"] = pattern_data.sfreq
        all_xreventprobs.attrs["event_width"] = len(pattern_data.template)
        all_xreventprobs.attrs["likelihood"] = np.sum(np.array(likelihood))
        all_xreventprobs.attrs["group_lkh"] = np.array(likelihood)
        all_xreventprobs.attrs["trial_lkh"] = trial_likelihood
        all_xreventprobs.attrs["group_labels"] = self.group_labels

        return [likelihood.sum(), all_xreventprobs]

    def location_pdf(
        self, time_pars: np.ndarray, max_duration: int, n_stages: int, sfreq: float, dtype: Any
    ) -> np.ndarray:
        """
        Compute pmf with locations enforced.

        Parameters
        ----------
        time_pars : np.ndarray
            A 2D array of shape (n_stages, n_parameters) containing current estimates for
            the distribution parameters.
        max_duration: int
            The maximum trial length (in samples).
        n_stages: int
            The number of intervals (number of events + 1).
        sfreq: float
            The sampling frequency of the data.
        dtype: Any
            The desired ``dtype`` of the return array

        Returns
        -------
        pmf : np.ndarray
            The probability mass function of dimension (max_duration, n_stages) censored
            appropriately based on the location parameter specified for the model.
        """
        pmf = np.zeros([max_duration, n_stages], dtype=dtype)  # Gamma pmf for each stage scale
        locations_samples = self._time_to_samples(self.locations, sfreq)
        locations_samples[1:-1] -= self.distribution.shift
        for stage in range(n_stages):
            pmf[:, stage] = np.concatenate(
                (
                    np.repeat(0, locations_samples[stage]),
                    self.distribution_pdf(time_pars[stage, 0], time_pars[stage, 1], max_duration)[
                        locations_samples[stage] :
                    ],
                )
            )
        return pmf

    def distribution_pdf(self, shape: float, scale: float, max_duration: int) -> np.ndarray:
        """
        Return a discretized probability density function (PDF) for a provided scipy distribution.

        This method computes the PDF using the given shape and scale parameters over a range
        from 0 to `max_duration`, and normalizes it to ensure the probabilities sum to 1.

        Parameters
        ----------
        shape : float
            The shape parameter of the distribution.
        scale : float
            The scale parameter of the distribution.
        max_duration : int
            The maximum duration (range) for which the PDF is computed.

        Returns
        -------
        np.ndarray
            A 1D array representing the probability mass function for the distribution
            with the given shape and scale parameters, normalized to sum to 1.
        """
        shift = self.distribution.shift
        p = self.distribution.pdf(np.arange(max_duration), shape, scale=scale)
        p[:shift] = 0  # PR-270
        p = p / np.sum(p)
        p[np.isnan(p)] = 0  # remove potential nans
        return p

    def group_constructor(  # noqa: PLR0912
        self, durations: xr.DataArray, verbose: bool = False
    ) -> tuple[int, np.ndarray, dict]:
        """
        Adapt the model to groups by constructing group mappings and validating provided maps.

        Parameters
        ----------
        durations : xr.DataArray
            By-trial durations as obtained with PatternData.durations, which include
            condition information
        verbose : bool, optional
            If True, prints detailed information about the group construction process.
            Default is False.

        Returns
        -------
        n_groups : int
            The number of unique groups.
        groups : np.ndarray
            An array indicating the group assignment for each trial.
        glabels : dict
            A dictionary containing group names and their corresponding modalities.
        """
        ## if no groups, directly return
        if len(self.grouping_dict.keys()) == 0:
            return (
                1,
                np.zeros(len(durations.values), dtype=np.int8),
                ("group all", np.array([[""]], dtype=object)),
            )

        # collect group names, groups, and trial coding
        group_names = []
        group_mods = []
        group_trials = []
        for group, mod in self.grouping_dict.items():
            group_names.append(group)
            group_mods.append(mod)
            group_trials.append(durations.coords[group])
            if verbose:
                print('group "' + group_names[-1] + '" analyzed, with groups:', group_mods[-1])

        group_mods = list(product(*group_mods))
        group_mods = np.array(group_mods, dtype=object)
        n_groups = len(group_mods)

        # build group array with digit indicating the combined groups
        if n_groups > 1:
            group_trials = np.vstack(group_trials).T
            groups = np.zeros((group_trials.shape[0])) * np.nan
            if verbose:
                print("\nCoded as follows: ")
            for i, mod in enumerate(group_mods):
                assert len(np.where((group_trials == mod).all(axis=1))[0]) > 0, (
                    f"Group {mod} does not occur in the data."
                )
                groups[np.where((group_trials == mod).all(axis=1))] = i
                if verbose:
                    print(str(i) + ": " + str(mod))
        else:  # in case dict provided but only one group
            groups = np.zeros(len(durations.values))
        groups = np.int8(groups)
        glabels = (str(group_names), group_mods)

        # check maps
        n_groups_channel = 0 if self.channel_map is None else self.channel_map.shape[0]
        n_groups_time = 0 if self.time_map is None else self.time_map.shape[0]
        # either both maps should have the same number of groups, or 0
        if n_groups_channel > 0 and n_groups_time > 0:
            assert n_groups_channel == n_groups_time, (
                "Channel and time parameter maps have to indicate the same number of groups."
            )
            # make sure nr of events correspond per row
            for cur_group in range(n_groups):
                assert sum(self.channel_map[cur_group, :] >= 0) + 1 == sum(
                    self.time_map[cur_group, :] >= 0
                ), "Nr of events in channel map and time map do not correspond on row " + str(
                    cur_group
                )
        elif n_groups_channel == 0:
            assert not (self.time_map < 0).any(), (
                "If negative time parameter are provided, channel map is required."
            )
            self.channel_map = np.zeros((n_groups, self.time_map.shape[1] - 1), dtype=int)
        else:  # n_groups_time == 0
            self.time_map = np.zeros((n_groups, self.channel_map.shape[1] + 1), dtype=int)
            if (self.channel_map < 0).any():
                for cur_group in range(n_groups):
                    self.time_map[cur_group, np.where(self.channel_map[cur_group, :] < 0)[0]] = -1
                    self.time_map[
                        cur_group, np.where(self.channel_map[cur_group, :] < 0)[0] + 1
                    ] = 1

        # at this point, all should indicate the same number of groups
        assert n_groups == self.channel_map.shape[0] == self.time_map.shape[0], (
            "number of unique groups should correspond to number of rows in map(s)"
        )

        if verbose:
            print("\nChannel map:")
            for cnt in range(n_groups):
                print(str(cnt) + ": ", self.channel_map[cnt, :])

            print("\nTime map:")
            for cnt in range(n_groups):
                print(str(cnt) + ": ", self.time_map[cnt, :])

        return n_groups, groups, glabels
