import itertools
import multiprocessing as mp
from itertools import product
from warnings import resetwarnings, warn

import numpy as np
import xarray as xr
from pandas import MultiIndex

from hmp.models.base import BaseModel
from hmp.trialdata import TrialData

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm



class FixedEventModel(BaseModel):
    def __init__(
        self, *args, n_events, fixed_time_pars=None, fixed_channel_pars=None,
        tolerance=1e-4,
        max_iteration=1e3,
        min_iteration=1,
        starting_points=1,
        max_scale=None,
        **kwargs
    ):
        assert np.issubdtype(type(n_events), np.integer), \
         (
             f"An integer for the number of expected transition events"
             f" is expected, got {n_events} instead"
         )
        self.n_events = n_events
        self.n_dims = None
        self.fixed_time_pars = fixed_time_pars
        self.fixed_channel_pars = fixed_channel_pars
        self.tolerance = tolerance
        self.max_iteration = max_iteration
        self.min_iteration = min_iteration
        self.starting_points = starting_points
        self.max_scale = max_scale
        self.level_dict = {}
        self.time_map = np.zeros((1,self.n_events+1))
        self.channel_map = np.zeros((1,self.n_events))
        super().__init__(*args, **kwargs)

    def fit(
        self,
        trial_data,
        channel_pars=None,
        time_pars=None,
        fixed_time_pars=None,
        fixed_channel_pars=None,
        verbose=True,
        cpus=1,
        channel_map=None,
        time_map=None,
        level_dict=None,
    ):
        """Fit HMP for a single n_events model.

        Parameters
        ----------
        n_events : int
            how many events are estimated
        channel_pars : ndarray
            2D ndarray n_events * channels (or 3D iteration * n_events * n_channels),
            initial conditions for events channel contribution. 
        time_pars : list
            list of initial conditions for time distribution parameters
            (2D stage * parameter or 3D iteration * n_events * n_channels).
        fixed_time_pars : bool
            To fix (True) or to estimate (False, default) the time distribution parameters
        fixed_channel_pars: bool
            To fix (True) or to estimate (False, default) the channel contribution
            to the events
        tolerance: float
            Tolerance applied to the expectation maximization in the EM() function
        max_iteration: int
            Maximum number of iteration for the expectation maximization in the EM() function
        min_iteration: int
            Minimum number of iteration for the expectation maximization in the EM() function
        starting_points: int
            How many starting points to use for the EM() function
        verbose: bool
            True displays output useful for debugging, recommended for first use
        cpus: int
            number of cores to use in the multiprocessing functions
        channel_map: 2D nd_array n_level * n_events indicating which channel contribution are shared
        between levels.
        time_map: 2D nd_array n_level * n_stages indicating which time parameters are shared
        between levels.
        levels: dict | list
            if one level, use a dict with the name in the metadata and a list of the levels
            in the same order as the rows of the map(s). E.g., {'cue': ['SP', 'AC']}
            if multiple levels need to be crossed, use a list of dictionaries per level. E.g.,
            [{'cue': ['SP', 'AC',]}, {'resp': ['left', 'right']}]. These are crossed by repeating
            the first level as many times as there are levels in the selevel level. E.g., SP-left,
            SP-right, AC-left, AC-right.
        max_scale: int
            expected maximum mean distance between events, only used when generating random starting points
        """
        # A dict containing all the info we want to keep, populated along the func
        infos_to_store = {}
        infos_to_store["sfreq"] = self.sfreq
        infos_to_store["event_width_samples"] = self.event_width_samples
        infos_to_store["tolerance"] = self.tolerance

        self.n_dims = trial_data.n_dims

        if level_dict is None:
            level_dict = self.level_dict
            channel_map = self.channel_map
            time_map = self.time_map
        n_levels, levels, clabels = self.level_constructor(
            trial_data, level_dict, channel_map, time_map, verbose
        )
        infos_to_store["channel_map"] = channel_map
        infos_to_store["time_map"] = time_map
        infos_to_store["clabels"] = clabels
        infos_to_store["level_dict"] = level_dict
        if verbose:
            if time_pars is None:
                print(
                    f"Estimating {self.n_events} events model with {self.starting_points} starting point(s)"
                )
            else:
                print(f"Estimating {self.n_events} events model")

        # Formatting parameters
        if isinstance(time_pars, (xr.DataArray, xr.Dataset)):
            time_pars = time_pars.dropna(dim="stage").values
        if isinstance(channel_pars, (xr.DataArray, xr.Dataset)):
            channel_pars = channel_pars.dropna(dim="event").values
        if isinstance(channel_pars, np.ndarray):
            channel_pars = channel_pars.copy()
        if isinstance(time_pars, np.ndarray):
            time_pars = time_pars.copy()
        if self.fixed_time_pars is None:
            fixed_time_pars = []
        else:
            fixed_time_pars = self.fixed_time_pars
            infos_to_store["fixed_time_pars"] = fixed_time_pars
        if self.fixed_channel_pars is None:
            fixed_channel_pars = []
        else:
            fixed_channel_pars = self.fixed_channel_pars
            infos_to_store["fixed_channel_pars"] = fixed_channel_pars

        if time_pars is None:
            # If no time parameters starting points are provided generate standard ones
            # Or random ones if starting_points > 1
            time_pars = (
                np.zeros((n_levels, self.n_events + 1, 2)) * np.nan
            )  # by default nan for missing stages
            for cur_level in range(n_levels):
                time_level = np.where(time_map[cur_level, :] >= 0)[0]
                n_stage_level = len(time_level)
                # by default starting point is to split the average duration in equal bins
                time_pars[cur_level, time_level, :] = np.tile(
                    [
                        self.distribution.shape,
                        self.distribution.mean_to_scale(
                            np.mean(trial_data.durations[levels == cur_level]) / (n_stage_level)
                        ),
                    ],
                    (n_stage_level, 1),
                )
            initial_p = time_pars
            time_pars = [initial_p]
            if self.starting_points > 1:
                if self.max_scale is None:
                    raise ValueError(
                            "If using multiple starting points, a maximum distance between events needs "
                            " to be provided using the max_scale argument."
                        )
                infos_to_store["starting_points"] = self.starting_points
                for _ in np.arange(self.starting_points):
                    proposal_p = (
                        np.zeros((n_levels, self.n_events + 1, 2)) * np.nan
                    )  # by default nan for missing stages
                    for cur_level in range(n_levels):
                        time_level = np.where(time_map[cur_level, :] >= 0)[0]
                        n_stage_level = len(time_level)
                        proposal_p[cur_level, time_level, :] = self.gen_random_stages(n_stage_level - 1)
                        proposal_p[cur_level, fixed_time_pars, :] = initial_p[0, fixed_time_pars]
                    time_pars.append(proposal_p)
                time_pars = np.array(time_pars)
        else:
            infos_to_store["sp_time_pars"] = time_pars

        if channel_pars is None:
            # By defaults c_pars are initiated to 0
            channel_pars = np.zeros((n_levels, self.n_events, self.n_dims), dtype=np.float64)
            if (channel_map < 0).any():  # set missing c_pars to nan
                for cur_level in range(n_levels):
                    channel_pars[cur_level, np.where(channel_map[cur_level, :] < 0)[0], :] = np.nan
            initial_m = channel_pars
            channel_pars = np.tile(initial_m, (self.starting_points + 1, 1, 1, 1))
        else:
            infos_to_store["sp_channel_pars"] = channel_pars

        if cpus > 1:
            inputs = zip(
                trial_data,
                channel_pars,
                time_pars,
                itertools.repeat(fixed_channel_pars),
                itertools.repeat(fixed_time_pars),
                itertools.repeat(self.max_iteration),
                itertools.repeat(self.tolerance),
                itertools.repeat(self.min_iteration),
                itertools.repeat(channel_map),
                itertools.repeat(time_map),
                itertools.repeat(levels),
                itertools.repeat(1),
            )
            with mp.Pool(processes=cpus) as pool:
                if self.starting_points > 1:
                    estimates = list(tqdm(pool.imap(self._EM_star, inputs), total=len(channel_pars)))
                else:
                    estimates = pool.starmap(self.EM, inputs)

        else:  # avoids problems if called in an already parallel function
            estimates = []
            for t_pars, c_pars in zip(time_pars, channel_pars):
                estimates.append(
                    self.EM(
                        trial_data,
                        c_pars,
                        t_pars,
                        fixed_channel_pars,
                        fixed_time_pars,
                        self.max_iteration,
                        self.tolerance,
                        self.min_iteration,
                        channel_map,
                        time_map,
                        levels,
                        1,
                    )
                )
            resetwarnings()

        lkhs = np.array([x[0] for x in estimates])
        if self.starting_points > 1 :
            max_lkhs = np.argmax(lkhs)
        else:
            max_lkhs = 0

        if np.isneginf(lkhs.sum()):
            raise ValueError("Fit failed, inspect provided starting points")
        else:
            self._fitted = True
            self.lkhs = lkhs[max_lkhs]
            self.channel_pars =  np.array(estimates[max_lkhs][1])
            self.time_pars = np.array(estimates[max_lkhs][2])
            self.traces = np.array(estimates[max_lkhs][3])
            self.time_pars_dev = np.array(estimates[max_lkhs][4])
            self.level_dict = level_dict
            self.level = levels
            self.channel_map = channel_map
            self.time_map = time_map

    def transform(self, trial_data):
        _, levels, clabels = self.level_constructor(
                trial_data, self.level_dict
            )
        likelihoods, xreventprobs = self._distribute_levels(
            trial_data, self.channel_pars, self.time_pars,
            self.channel_map, self.time_map, levels, False
        )
        return likelihoods, xreventprobs



    @property
    def xrtraces(self):
        self._check_fitted("get traces")
        return xr.DataArray(
            self.traces,
            dims=("em_iteration","level"),
            name="traces",
            coords={
                "em_iteration": range(self.traces.shape[0]),
                "level": range(self.traces.shape[1]),
            }
        )

    @property
    def xrlikelihoods(self):
        self._check_fitted("get likelihoods")
        return xr.DataArray(self.lkhs, name="loglikelihood")

    @property
    def xrtime_pars_dev(self):
        self._check_fitted("get dev time pars")
        return xr.DataArray(
                self.time_pars_dev,
                dims=("em_iteration", "level", "stage", "time_pars"),
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
        self._check_fitted("get xrtime_pars")
        return xr.DataArray(
                self.time_pars,
                dims=("level", "stage", "parameter"),
                name="time_pars",
                coords={
                    "level": range(self.time_pars.shape[0]),
                    "stage": range(self.n_events + 1),
                    "parameter": ["shape", "scale"],
                },
        )

    @property
    def xrchannel_pars(self):
        self._check_fitted("get xrchannel_pars")
        return xr.DataArray(
                self.channel_pars,
                dims=("level", "event", "channel"),
                name="channel_pars",
                coords={
                    "level": range(self.channel_pars.shape[0]),
                    "event": range(self.n_events),
                    "channel": range(self.n_dims),
                },
            )




    def _EM_star(self, args):  # for tqdm usage  #noqa
        return self.EM(*args)

    def EM(  #noqa
        self,
        trial_data,
        initial_channel_pars,
        initial_time_pars,
        fixed_channel_pars=None,
        fixed_time_pars=None,
        max_iteration=1e3,
        tolerance=1e-4,
        min_iteration=1,
        channel_map=None,
        time_map=None,
        levels=None,
        cpus=1,
    ):
        """Fit using expectation maximization.

        Parameters
        ----------
        n_events : int
            how many events are estimated
        initial_channel_pars : ndarray
            2D ndarray n_events * channels (or 3D iteration * n_events * n_channels),
            initial conditions for events channel_pars.
        initial_time_pars : list
            list of initial conditions for time distribution parameters parameter
            (2D stage * parameter or 3D iteration * n_events * n_channels).
        fixed_channel_pars: list
            Which parameters of the channel contribution to the events to estimate
        fixed_time_pars : list
            Which parameters of the time distributions to estimate
        max_iteration: int
            Maximum number of iteration for the expectation maximization
        tolerance: float
            Tolerance applied to the expectation maximization
        min_iteration: int
            Minimum number of iteration for the expectation maximization in the EM() function

        Returns
        -------
        lkh : float
            Summed log probabilities
        channel_pars : ndarray
            Estimated channel contribution to each event
        time_pars: ndarray
            Estimated distribution time parameters of each stage
        eventprobs: ndarray
            Probabilities with shape max_samples*n_trials*n_events
        traces: ndarray
            Values of the log-likelihood for each EM iteration
        time_pars_dev : ndarray
            paramters for each iteration of EM
        """
        assert channel_map.shape[0] == time_map.shape[0], (
            "Both maps need to indicate the same number of levels."
        )

        lkh, eventprobs = self._distribute_levels(
            trial_data, initial_channel_pars, initial_time_pars, 
            channel_map, time_map, levels, cpus=cpus
        )
        data_levels = np.unique(levels)
        channel_pars = initial_channel_pars.copy() 
        time_pars = initial_time_pars.copy()
        traces = [lkh]
        time_pars_dev = [time_pars.copy()] 
        i = 0

        while i < max_iteration:  # Expectation-Maximization algorithm
            if i >= min_iteration and (
                np.isneginf(lkh.sum()) or \
                tolerance > (lkh.sum() - lkh_prev.sum()) / np.abs(lkh_prev.sum())
            ):
                break

            # As long as new run gives better likelihood, go on
            lkh_prev = lkh.copy()

            for cur_level in data_levels:  # get params/c_pars
                channel_map_level = np.where(channel_map[cur_level, :] >= 0)[0]
                time_map_level = np.where(time_map[cur_level, :] >= 0)[0]
                epochs_level = np.where(levels == cur_level)[0]
                # get c_pars/t_pars by level
                channel_pars[cur_level, channel_map_level, :], time_pars[cur_level, time_map_level, :] = (
                    self.get_channel_time_parameters_expectation(
                        trial_data,
                        eventprobs.values[:, :np.max(trial_data.durations[epochs_level]), channel_map_level],
                        subset_epochs=epochs_level,
                    )
                )

                channel_pars[cur_level, fixed_channel_pars, :] = initial_channel_pars[
                    cur_level, fixed_channel_pars, :
                ].copy()
                time_pars[cur_level, fixed_time_pars, :] = initial_time_pars[
                    cur_level, fixed_time_pars, :
                ].copy()

            # set c_pars to mean if requested in map
            for m in range(self.n_events):
                for m_set in np.unique(channel_map[:, m]):
                    if m_set >= 0:
                        channel_pars[channel_map[:, m] == m_set, m, :] = np.mean(
                            channel_pars[channel_map[:, m] == m_set, m, :], axis=0
                        )

            # set param to mean if requested in map
            for p in range(self.n_events + 1):
                for p_set in np.unique(time_map[:, p]):
                    if p_set >= 0:
                        time_pars[time_map[:, p] == p_set, p, :] = np.mean(
                            time_pars[time_map[:, p] == p_set, p, :], axis=0
                        )

            
            lkh, eventprobs = self._distribute_levels(
                trial_data, channel_pars, time_pars, channel_map, time_map, levels, cpus=cpus
            )
            traces.append(lkh)
            time_pars_dev.append(time_pars.copy())
            i += 1

        if i == max_iteration:
            warn(
                f"Convergence failed, estimation hit the maximum number of iterations: "
                f"({int(max_iteration)})",
                RuntimeWarning,
            )
        return lkh, channel_pars, time_pars, np.array(traces), np.array(time_pars_dev)

    def get_channel_time_parameters_expectation(self, trial_data, eventprobs, subset_epochs=None):
        channel_pars = np.zeros((eventprobs.shape[2], self.n_dims))
        # Channel contribution from Expectation, Eq 11 from 2024 paper
        for event in range(eventprobs.shape[2]):
            for comp in range(self.n_dims):
                event_data = np.zeros((len(subset_epochs), np.max(trial_data.durations[subset_epochs])))
                for trial_idx, trial in enumerate(subset_epochs):
                    start, end = trial_data.starts[trial], trial_data.ends[trial]
                    duration = end - start + 1
                    event_data[trial_idx, :duration] = trial_data.cross_corr[start : end + 1, comp]
                channel_pars[event, comp] = np.mean(
                    np.sum(eventprobs[subset_epochs, :, event] * event_data, axis=1)
                )
            # scale cross-correlation with likelihood of the transition
            # sum by-trial these scaled activation for each transition events
            # average across trial

        # Time parameters from Expectation Eq 10 from 2024 paper
        # calc averagepos here as mean_d can be level dependent, whereas scale_parameters() assumes
        # it's general
        event_times_mean = np.concatenate(
            [
                np.arange(np.max(trial_data.durations[subset_epochs])) @ eventprobs[subset_epochs].mean(axis=0),
                [np.mean(trial_data.durations[subset_epochs]) - 1],
            ]
        )
        time_pars = self.scale_parameters(averagepos=event_times_mean)
        return [channel_pars, time_pars]

    def gen_random_stages(self, n_events):
        """Compute random stage duration.

        Returns random stage duration between 0 and mean RT by iteratively drawind sample from a
        uniform distribution between the last stage duration (equal to 0 for first iteration) and 1.
        Last stage is equal to 1-previous stage duration.
        The stages are then scaled to the mean RT

        Parameters
        ----------
        n_events : int
            how many events

        Returns
        -------
        random_stages : ndarray
            random partition between 0 and mean_d
        """
        rnd_durations = np.zeros(n_events + 1)
        assert self.event_width_samples*(n_events + 1) < self.max_scale, \
            f"Max_scale too short, need to be more than {self.event_width_samples*(n_events+1)}"
        while any(rnd_durations < self.event_width_samples):  # at least event_width
            rnd_events = np.random.default_rng().integers(
                low=0, high=self.max_scale, size=n_events
            )  # n_events between 0 and mean_d
            rnd_events = np.sort(rnd_events)
            rnd_durations = np.hstack((rnd_events, self.max_scale)) - np.hstack(
                (0, rnd_events)
            )  # associated durations
        random_stages = np.array(
            [[self.distribution.shape, self.distribution.mean_to_scale(x)] for x in rnd_durations]
        )
        return random_stages


    def scale_parameters(self, averagepos):
        """Scale parameters from average position of event.

        Used for the re-estimation in the EM procdure. The likeliest location of
        the event is computed from eventprobs. The scale parameter are then taken as the average
        distance between the events

        Parameters
        ----------


        Returns
        -------
        params : ndarray
            shape and scale for the distributions
        """
        params = np.zeros((len(averagepos), 2), dtype=np.float64)
        params[:, 0] = self.distribution.shape
        params[:, 1] = np.diff(averagepos, prepend=0)
        params[:, 1] = self.distribution.mean_to_scale(params[:, 1])
        return params

    def estim_probs(
        self,
        trial_data : TrialData,
        channel_pars,
        time_pars,
        location=True,
        subset_epochs=None,
        by_trial_lkh=False,
    ):
        """Estimate probabilities.

        Parameters
        ----------
        channel_pars : ndarray
            2D ndarray n_events * channels (or 3D iteration * n_events * n_channels),
            initial conditions for channel contributio to event.
        time_pars : list
            list of initial conditions for the distribution parameters
            (2D stage * parameter or 3D iteration * n_events * n_channels).
        locations : ndarray
            1D ndarray of int with size n_events+1, locations for events

        subset_epochs : list
            boolean array indicating which epoch should be taken into account for level-based calcs

        Returns
        -------
        loglikelihood : float
            Summed log probabilities
        eventprobs : ndarray
            Probabilities with shape max_samples*n_trials*n_events
        """
        n_events = channel_pars.shape[0]
        n_stages = n_events + 1
        locations = np.zeros(n_stages, dtype=int)
        if location:
            locations[1:-1] = self.location
        if subset_epochs is not None:
            if len(subset_epochs) == trial_data.n_trials:  # boolean indices
                subset_epochs = np.where(subset_epochs)[0]
        n_trials = len(subset_epochs)
        durations = trial_data.durations[subset_epochs]
        starts = trial_data.starts[subset_epochs]
        ends = trial_data.ends[subset_epochs]
        max_duration = np.max(durations)
        gains = np.zeros((trial_data.n_samples, n_events), dtype=np.float64)
        for i in range(trial_data.n_dims):
            # computes the gains, i.e. congruence between the pattern shape
            # and the data given the magnitudes of the sensors
            gains = (
                gains
                + trial_data.cross_corr[:, i][np.newaxis].T * channel_pars[:, i]
                - channel_pars[:, i] ** 2 / 2
            )
        gains = np.exp(gains)
        probs = np.zeros([max_duration, n_trials, n_events], dtype=np.float64)  # prob per trial
        probs_b = np.zeros(
            [max_duration, n_trials, n_events], dtype=np.float64
        )  # Sample and state reversed
        for trial in np.arange(n_trials):
            # Following assigns gain per trial to variable probs
            probs[: durations[trial], trial, :] = gains[starts[trial] : ends[trial] + 1, :]
            # Same but sample and events are reversed, this allows to compute
            # fwd and bwd in the same way in the following steps
            probs_b[: durations[trial], trial, :] = gains[starts[trial] : ends[trial] + 1, :][
                ::-1, ::-1
            ]

        pmf = np.zeros([max_duration, n_stages], dtype=np.float64)  # Gamma pmf for each stage scale
        for stage in range(n_stages):
            pmf[:, stage] = np.concatenate(
                (
                    np.repeat(1e-15, locations[stage]),
                    self.distribution_pdf(time_pars[stage, 0], time_pars[stage, 1], max_duration)[
                        locations[stage] :
                    ],
                )
            )
        pmf_b = pmf[:, ::-1]  # Stage reversed gamma pmf, same order as prob_b

        forward = np.zeros((max_duration, n_trials, n_events), dtype=np.float64)
        backward = np.zeros((max_duration, n_trials, n_events), dtype=np.float64)
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
                    : max_duration
                ]
                # same but backwards
                backward[:, trial, event] = np.convolve(add_b[:, trial], pmf_b[:, event])[
                    : max_duration
                ]
            forward[:, :, event] = forward[:, :, event] * probs[:, :, event]
        # re-arranging backward to the expected variable
        backward = backward[:, :, ::-1]  # undoes stage inversion
        for trial in np.arange(n_trials):  # Undoes sample inversion
            backward[: durations[trial], trial, :] = backward[: durations[trial], trial, :][::-1]
        eventprobs = forward * backward
        eventprobs = np.clip(eventprobs, 0, None)  # floating point precision error
        likelihood = np.sum(
            np.log(eventprobs[:, :, 0].sum(axis=0))
        )  # sum over max_samples to avoid 0s in log
        eventprobs = eventprobs / eventprobs.sum(axis=0)
        eventprobs = eventprobs.transpose((1,0,2))
        if by_trial_lkh:
            return forward * backward
        else:
            return [likelihood, eventprobs]

    def _distribute_levels(
        self, trial_data, channel_pars, time_pars, channel_map, time_map, levels, 
        location=True, cpus=1
    ):
        """Estimate probability levels.

        Parameters
        ----------
        channel_pars : ndarray
            2D ndarray n_events * channels (or 3D iteration * n_events * n_channels),
            initial conditions for events channel contribution. 
        time_pars : list
            list of initial conditions for time distribution parameters
        location : bool
            Whether to add a minumum distance between events, useful to avoid event collapse during EM
        n_events : int
            how many events are estimated
            
        Returns
        -------
        loglikelihood : float
            Summed log probabilities
        eventprobs : ndarray
            Probabilities with shape max_samples*n_trials*n_events
        """
        data_levels = np.unique(levels)
        likes_events_level = []
        all_xreventprobs = []
        if cpus > 1:
            with mp.Pool(processes=cpus) as pool:
                likes_events_level = pool.starmap(
                    self.estim_probs,
                    zip(
                        itertools.repeat(trial_data),
                        [channel_pars[cur_level, channel_map[cur_level, :] >= 0, :] for cur_level in data_levels],
                        [time_pars[cur_level, time_map[cur_level, :] >= 0, :] for cur_level in data_levels],
                        itertools.repeat(location),
                        [levels == cur_level for cur_level in data_levels],
                        itertools.repeat(False),
                    ),
                )
        else:
            for cur_level in data_levels:
                channel_pars_level = channel_pars[
                    cur_level, channel_map[cur_level, :] >= 0, :
                ]  # select existing magnitudes
                time_pars_level = time_pars[cur_level, time_map[cur_level, :] >= 0, :]  # select existing params
                likes_events_level.append(
                    self.estim_probs(
                        trial_data,
                        channel_pars_level,
                        time_pars_level,
                        location,
                        subset_epochs=(levels == cur_level),
                    )
                )

        likelihood = np.array([x[0] for x in likes_events_level])

        for i, cur_level in enumerate(data_levels):
            part = trial_data.coords["participant"].values[(levels == cur_level)]
            trial = trial_data.coords["trial"].values[(levels == cur_level)]
            data_events =  channel_map[cur_level, :] >= 0
            trial_x_part = xr.Coordinates.from_pandas_multiindex(
                MultiIndex.from_arrays([part, trial], names=("participant", "epoch")),
                "trial",
            )
            xreventprobs = xr.DataArray(likes_events_level[i][1], dims=("trial", "sample", "event"),
                coords={
                    "event": ("event", np.arange(self.n_events)[data_events]),
                    "sample": ("sample", range(np.shape(likes_events_level[i][1])[1])),
                },
            )
            xreventprobs = xreventprobs.assign_coords(trial_x_part)
            xreventprobs = xreventprobs.assign_coords(level=("trial", levels[levels == cur_level],))
            all_xreventprobs.append(xreventprobs)
        all_xreventprobs = xr.concat(all_xreventprobs, dim="trial")
        all_xreventprobs.attrs['sfreq'] = self.sfreq
        all_xreventprobs.attrs['event_width_samples'] = self.event_width_samples
        return [np.array(likelihood), all_xreventprobs]

    def distribution_pdf(self, shape, scale, max_duration):
        """Return discretized PDF for a provided scipy disttribution.

        Uses the shape and scale, on a range from 0 to max_length.

        Parameters
        ----------
        shape : float
            shape parameter
        scale : float
            scale parameter

        Returns
        -------
        p : ndarray
            probabilty mass function for the distribution with given scale
        """
        p = self.distribution.pdf(np.arange(max_duration), shape, scale=scale)
        p = p / np.sum(p)
        p[np.isnan(p)] = 0  # remove potential nans
        return p

    def level_constructor(self, trial_data, level_dict, channel_map=None, time_map=None, verbose=False):
        """Adapt model to levels.
        """
        ## levels
        assert isinstance(level_dict, dict), "levels have to be specified as a dictionary"
        if len(level_dict.keys()) == 0:
            verbose = False
        # collect level names, levels, and trial coding
        level_names = []
        level_mods = []
        level_trials = []
        for level in level_dict.keys():
            level_names.append(level)
            level_mods.append(level_dict[level])
            level_trials.append(trial_data.trial_coords[level])
            if verbose:
                print('Level "' + level_names[-1] + '" analyzed, with levels:', level_mods[-1])

        level_mods = list(product(*level_mods))
        level_mods = np.array(level_mods, dtype=object)
        n_levels = len(level_mods)

        # build level array with digit indicating the combined levels
        if n_levels > 1:
            level_trials = np.vstack(level_trials).T
            levels = np.zeros((level_trials.shape[0])) * np.nan
            if verbose:
                print("\nCoded as follows: ")
            for i, mod in enumerate(level_mods):
                # assert len(np.where((level_trials == mod).all(axis=1))[0]) > 0, (
                #     f"Modality {mod} of level does not occur in the data"
                # )
                levels[np.where((level_trials == mod).all(axis=1))] = i
                if verbose:
                    print(str(i) + ": " + str(level))
        else:
            levels = np.zeros(trial_data.n_trials)
        levels = np.int8(levels)
        clabels = {"level " + str(level_names): level_mods}

        # check maps if provided
        if channel_map is not None and time_map is not None:
            n_levels_mags = 0 if channel_map is None else channel_map.shape[0]
            n_levels_pars = 0 if time_map is None else time_map.shape[0]
            if (
                n_levels_mags > 0 and n_levels_pars > 0
            ):  # either both maps should have the same number of levels, or 0
                assert n_levels_mags == n_levels_pars, (
                    "Channel and time parameter maps have to indicate the same number of levels"
                )
                # make sure nr of events correspond per row
                for cur_level in range(n_levels):
                    assert sum(channel_map[cur_level, :] >= 0) + 1 == sum(time_map[cur_level, :] >= 0), (
                        "nr of events in channel map and time map do not correspond on row "
                        + str(cur_level)
                    )
            elif n_levels_mags == 0:
                assert not (time_map < 0).any(), (
                    "If negative time parameter are provided, channel map is required."
                )
                channel_map = np.zeros((n_levels, time_map.shape[1] - 1), dtype=int)
            else:
                time_map = np.zeros((n_levels, channel_map.shape[1] + 1), dtype=int)
                if (channel_map < 0).any():
                    for cur_level in range(n_levels):
                        time_map[cur_level, np.where(channel_map[cur_level, :] < 0)[0]] = -1
                        time_map[cur_level, np.where(channel_map[cur_level, :] < 0)[0] + 1] = 1
    
            # at this point, all should indicate the same number of levels
            assert n_levels == channel_map.shape[0] == time_map.shape[0], (
                "number of unique levels should correspond to number of rows in map(s)"
            )
    
            if verbose:
                print("\nChannel map:")
                for cnt in range(n_levels):
                    print(str(cnt) + ": ", channel_map[cnt, :])
    
                print("\nTime map:")
                for cnt in range(n_levels):
                    print(str(cnt) + ": ", time_map[cnt, :])

            # at this point, all should indicate the same number of levels
            assert n_levels == channel_map.shape[0] == time_map.shape[0], (
                "number of unique levels should correspond to number of rows in map(s)"
            )

        return n_levels, levels, clabels
