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
        self, *args, n_events, parameters_to_fix=None, magnitudes_to_fix=None,
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
        self.parameters_to_fix = parameters_to_fix
        self.magnitudes_to_fix = magnitudes_to_fix
        self.tolerance = tolerance
        self.max_iteration = max_iteration
        self.min_iteration = min_iteration
        self.starting_points = starting_points
        self.max_scale = max_scale
        self.level_dict = {}
        self.pars_map = np.zeros((1,self.n_events+1))
        self.mags_map = np.zeros((1,self.n_events))
        super().__init__(*args, **kwargs)

    def fit(
        self,
        trial_data,
        magnitudes=None,
        parameters=None,
        parameters_to_fix=None,
        magnitudes_to_fix=None,
        verbose=True,
        cpus=1,
        mags_map=None,
        pars_map=None,
        level_dict=None,
    ):
        """Fit HMP for a single n_events model.

        Parameters
        ----------
        n_events : int
            how many events are estimated
        magnitudes : ndarray
            2D ndarray n_events * components (or 3D iteration * n_events * n_components),
            initial conditions for events magnitudes. If magnitudes are estimated, the list provided
            is used as starting point,
            if magnitudes are fixed, magnitudes estimated will be the same as the one provided.
            When providing a list, magnitudes need to be in the same order
            _n_th magnitudes parameter is  used for the _n_th event
        parameters : list
            list of initial conditions for Gamma distribution parameters
            (2D stage * parameter or 3D iteration * n_events * n_components).
            If parameters are estimated, the list provided is used as starting point,
            if parameters are fixed, parameters estimated will be the same as the one provided.
            When providing a list, stage need to be in the same order
            _n_th gamma parameter is  used for the _n_th stage
        parameters_to_fix : bool
            To fix (True) or to estimate (False, default) the parameters of the gammas
        magnitudes_to_fix: bool
            To fix (True) or to estimate (False, default) the magnitudes of the channel contribution
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
        mags_map: 2D nd_array n_level * n_events indicating which magnitudes are shared
        between levels.
        pars_map: 2D nd_array n_level * n_stages indicating which parameters are shared
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
            mags_map = self.mags_map
            pars_map = self.pars_map
        n_levels, levels, clabels = self.level_constructor(
            trial_data, level_dict, mags_map, pars_map, verbose
        )
        infos_to_store["mags_map"] = mags_map
        infos_to_store["pars_map"] = pars_map
        infos_to_store["clabels"] = clabels
        infos_to_store["level_dict"] = level_dict
        if verbose:
            if parameters is None:
                print(
                    f"Estimating {self.n_events} events model with {self.starting_points} starting point(s)"
                )
            else:
                print(f"Estimating {self.n_events} events model")

        # Formatting parameters
        if isinstance(parameters, (xr.DataArray, xr.Dataset)):
            parameters = parameters.dropna(dim="stage").values
        if isinstance(magnitudes, (xr.DataArray, xr.Dataset)):
            magnitudes = magnitudes.dropna(dim="event").values
        if isinstance(magnitudes, np.ndarray):
            magnitudes = magnitudes.copy()
        if isinstance(parameters, np.ndarray):
            parameters = parameters.copy()
        if self.parameters_to_fix is None:
            parameters_to_fix = []
        else:
            parameters_to_fix = self.parameters_to_fix
            infos_to_store["parameters_to_fix"] = parameters_to_fix
        if self.magnitudes_to_fix is None:
            magnitudes_to_fix = []
        else:
            magnitudes_to_fix = self.magnitudes_to_fix
            infos_to_store["magnitudes_to_fix"] = magnitudes_to_fix

        if parameters is None:
            # If no parameters starting points are provided generate standard ones
            # Or random ones if starting_points > 1
            parameters = (
                np.zeros((n_levels, self.n_events + 1, 2)) * np.nan
            )  # by default nan for missing stages
            for cur_level in range(n_levels):
                pars_level = np.where(pars_map[cur_level, :] >= 0)[0]
                n_stage_level = len(pars_level)
                # by default starting point is to split the average duration in equal bins
                parameters[cur_level, pars_level, :] = np.tile(
                    [
                        self.shape,
                        self.mean_to_scale(
                            np.mean(trial_data.durations[levels == cur_level]) / (n_stage_level), self.shape
                        ),
                    ],
                    (n_stage_level, 1),
                )
            initial_p = parameters
            parameters = [initial_p]
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
                        pars_level = np.where(pars_map[cur_level, :] >= 0)[0]
                        n_stage_level = len(pars_level)
                        proposal_p[cur_level, pars_level, :] = self.gen_random_stages(n_stage_level - 1)
                        proposal_p[cur_level, parameters_to_fix, :] = initial_p[0, parameters_to_fix]
                    parameters.append(proposal_p)
                parameters = np.array(parameters)
        else:
            infos_to_store["sp_parameters"] = parameters

        if magnitudes is None:
            # By defaults mags are initiated to 0
            magnitudes = np.zeros((n_levels, self.n_events, self.n_dims), dtype=np.float64)
            if (mags_map < 0).any():  # set missing mags to nan
                for cur_level in range(n_levels):
                    magnitudes[cur_level, np.where(mags_map[cur_level, :] < 0)[0], :] = np.nan
            initial_m = magnitudes
            magnitudes = np.tile(initial_m, (self.starting_points + 1, 1, 1, 1))
        else:
            infos_to_store["sp_magnitudes"] = magnitudes

        if cpus > 1:
            inputs = zip(
                trial_data,
                magnitudes,
                parameters,
                itertools.repeat(magnitudes_to_fix),
                itertools.repeat(parameters_to_fix),
                itertools.repeat(self.max_iteration),
                itertools.repeat(self.tolerance),
                itertools.repeat(self.min_iteration),
                itertools.repeat(mags_map),
                itertools.repeat(pars_map),
                itertools.repeat(levels),
                itertools.repeat(1),
            )
            with mp.Pool(processes=cpus) as pool:
                if self.starting_points > 1:
                    estimates = list(tqdm(pool.imap(self._EM_star, inputs), total=len(magnitudes)))
                else:
                    estimates = pool.starmap(self.EM, inputs)

        else:  # avoids problems if called in an already parallel function
            estimates = []
            for pars, mags in zip(parameters, magnitudes):
                estimates.append(
                    self.EM(
                        trial_data,
                        mags,
                        pars,
                        magnitudes_to_fix,
                        parameters_to_fix,
                        self.max_iteration,
                        self.tolerance,
                        self.min_iteration,
                        mags_map,
                        pars_map,
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
            self.magnitudes =  np.array(estimates[max_lkhs][1])
            self.parameters = np.array(estimates[max_lkhs][2])
            self.traces = np.array(estimates[max_lkhs][3])
            self.param_dev = np.array(estimates[max_lkhs][4])
            self.level_dict = level_dict
            self.levels = levels
            self.mags_map = mags_map
            self.pars_map = pars_map

    def transform(self, trial_data):
        _, levels, clabels = self.level_constructor(
                trial_data, self.level_dict
            )
        likelihoods, xreventprobs = self._distribute_levels(
            trial_data, self.magnitudes, self.parameters,
            self.mags_map, self.pars_map, levels, False
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
    def xrparam_dev(self):
        self._check_fitted("get dev params")
        return xr.DataArray(
                self.param_dev,
                dims=("em_iteration", "level", "stage", "parameter"),
                name="param_dev",
                coords=[
                    range(self.param_dev.shape[0]),
                    range(self.param_dev.shape[1]),
                    range(self.n_events + 1),
                    ["shape", "scale"],
                ],
        )

    @property
    def xrparams(self):
        self._check_fitted("get xrparams")
        return xr.DataArray(
                self.parameters,
                dims=("level", "stage", "parameter"),
                name="parameters",
                coords={
                    "level": range(self.parameters.shape[0]),
                    "stage": range(self.n_events + 1),
                    "parameter": ["shape", "scale"],
                },
        )

    @property
    def xrmags(self):
        self._check_fitted("get xrmags")
        return xr.DataArray(
                self.magnitudes,
                dims=("level", "event", "component"),
                name="magnitudes",
                coords={
                    "level": range(self.magnitudes.shape[0]),
                    "event": range(self.n_events),
                    "component": range(self.n_dims),
                },
            )




    def _EM_star(self, args):  # for tqdm usage  #noqa
        return self.EM(*args)

    def EM(  #noqa
        self,
        trial_data,
        magnitudes,
        parameters,
        magnitudes_to_fix=None,
        parameters_to_fix=None,
        max_iteration=1e3,
        tolerance=1e-4,
        min_iteration=1,
        mags_map=None,
        pars_map=None,
        levels=None,
        cpus=1,
    ):
        """Fit using expectation maximization.

        Parameters
        ----------
        n_events : int
            how many events are estimated
        magnitudes : ndarray
            2D ndarray n_events * components (or 3D iteration * n_events * n_components),
            initial levelitions for events magnitudes. If magnitudes are estimated,
            the list provided is used as starting point,
            if magnitudes are fixed, magnitudes estimated will be the same as the one provided.
            When providing a list, magnitudes need to be in the same order
            _n_th magnitudes parameter is  used for the _n_th event
        parameters : list
            list of initial conditions for Gamma distribution parameters parameter
            (2D stage * parameter or 3D iteration * n_events * n_components).
            If parameters are estimated, the list provided is used as starting point,
            if parameters are fixed, parameters estimated will be the same as the one provided.
            When providing a list, stage need to be in the same order
            _n_th gamma parameter is  used for the _n_th stage
        magnitudes_to_fix: bool
            To fix (True) or to estimate (False, default) the magnitudes of the channel contribution
            to the events
        parameters_to_fix : bool
            To fix (True) or to estimate (False, default) the parameters of the gammas
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
        magnitudes : ndarray
            Magnitudes of the channel contribution to each event
        parameters: ndarray
            parameters for the gammas of each stage
        eventprobs: ndarray
            Probabilities with shape max_samples*n_trials*n_events
        traces: ndarray
            Values of the log-likelihood for each EM iteration
        param_dev : ndarray
            paramters for each iteration of EM
        """
        assert mags_map.shape[0] == pars_map.shape[0], (
            "Both maps need to indicate the same number of levels."
        )

        lkh, eventprobs = self._distribute_levels(
            trial_data, magnitudes, parameters, mags_map, pars_map, levels, cpus=cpus
        )
        data_levels = np.unique(levels)
        initial_magnitudes = magnitudes.copy()  # Reverse this
        initial_parameters = parameters.copy()
        traces = [lkh]
        param_dev = [parameters.copy()]  # ... and parameters
        i = 0

        lkh_prev = lkh
        parameters_prev = parameters.copy()

        while i < max_iteration:  # Expectation-Maximization algorithm
            if i >= min_iteration and (
                np.isneginf(lkh.sum()) or \
                tolerance > (lkh.sum() - lkh_prev.sum()) / np.abs(lkh_prev.sum())
            ):
                break

            # As long as new run gives better likelihood, go on
            lkh_prev = lkh.copy()
            parameters_prev = parameters.copy()

            for cur_level in data_levels:  # get params/mags
                mags_map_level = np.where(mags_map[cur_level, :] >= 0)[0]
                pars_map_level = np.where(pars_map[cur_level, :] >= 0)[0]
                epochs_level = np.where(levels == cur_level)[0]
                # get mags/pars by level
                magnitudes[cur_level, mags_map_level, :], parameters[cur_level, pars_map_level, :] = (
                    self.get_magnitudes_parameters_expectation(
                        trial_data,
                        eventprobs.values[:, :np.max(trial_data.durations[epochs_level]), mags_map_level],
                        subset_epochs=epochs_level,
                    )
                )

                magnitudes[cur_level, magnitudes_to_fix, :] = initial_magnitudes[
                    cur_level, magnitudes_to_fix, :
                ].copy()
                parameters[cur_level, parameters_to_fix, :] = initial_parameters[
                    cur_level, parameters_to_fix, :
                ].copy()

            # set mags to mean if requested in map
            for m in range(self.n_events):
                for m_set in np.unique(mags_map[:, m]):
                    if m_set >= 0:
                        magnitudes[mags_map[:, m] == m_set, m, :] = np.mean(
                            magnitudes[mags_map[:, m] == m_set, m, :], axis=0
                        )

            # set param to mean if requested in map
            for p in range(self.n_events + 1):
                for p_set in np.unique(pars_map[:, p]):
                    if p_set >= 0:
                        parameters[pars_map[:, p] == p_set, p, :] = np.mean(
                            parameters[pars_map[:, p] == p_set, p, :], axis=0
                        )

            
            lkh, eventprobs = self._distribute_levels(
                trial_data, magnitudes, parameters, mags_map, pars_map, levels, cpus=cpus
            )
            traces.append(lkh)
            param_dev.append(parameters.copy())
            i += 1

        if i == max_iteration:
            warn(
                f"Convergence failed, estimation hit the maximum number of iterations: "
                f"({int(max_iteration)})",
                RuntimeWarning,
            )
        return lkh, magnitudes, parameters, np.array(traces), np.array(param_dev)

    def get_magnitudes_parameters_expectation(self, trial_data, eventprobs, subset_epochs=None):
        magnitudes = np.zeros((eventprobs.shape[2], self.n_dims))
        # Magnitudes from Expectation, Eq 11 from 2024 paper
        for event in range(eventprobs.shape[2]):
            for comp in range(self.n_dims):
                event_data = np.zeros((len(subset_epochs), np.max(trial_data.durations[subset_epochs])))
                for trial_idx, trial in enumerate(subset_epochs):
                    start, end = trial_data.starts[trial], trial_data.ends[trial]
                    duration = end - start + 1
                    event_data[trial_idx, :duration] = trial_data.cross_corr[start : end + 1, comp]
                magnitudes[event, comp] = np.mean(
                    np.sum(eventprobs[subset_epochs, :, event] * event_data, axis=1)
                )
            # scale cross-correlation with likelihood of the transition
            # sum by-trial these scaled activation for each transition events
            # average across trials

        # Gamma parameters from Expectation Eq 10 from 2024 paper
        # calc averagepos here as mean_d can be level dependent, whereas scale_parameters() assumes
        # it's general
        event_times_mean = np.concatenate(
            [
                np.arange(np.max(trial_data.durations[subset_epochs])) @ eventprobs[subset_epochs].mean(axis=0),
                [np.mean(trial_data.durations[subset_epochs]) - 1],
            ]
        )
        parameters = self.scale_parameters(averagepos=event_times_mean)
        return [magnitudes, parameters]

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
            [[self.shape, self.mean_to_scale(x, self.shape)] for x in rnd_durations]
        )
        return random_stages


    def scale_parameters(self, averagepos):
        """Scale the parameters for the distribution.

        Used for the re-estimation in the EM procdure. The likeliest location of
        the event is computed from eventprobs. The scale parameter are then taken as the average
        distance between the events

        Parameters
        ----------
        eventprobs : ndarray
            [samples(max_d)*n_trials*n_events] = [max_d*trials*nTransition events]
        durations : ndarray
            1D array of trial length
        mags : ndarray
            2D ndarray components * nTransition events, initial conditions for events magnitudes
        shape : float
            shape parameter for the gamma, defaults to 2

        Returns
        -------
        params : ndarray
            shape and scale for the gamma distributions
        """
        params = np.zeros((len(averagepos), 2), dtype=np.float64)
        params[:, 0] = self.shape
        params[:, 1] = np.diff(averagepos, prepend=0)
        params[:, 1] = [self.mean_to_scale(x[1], x[0]) for x in params]
        return params

    def estim_probs(
        self,
        trial_data : TrialData,
        magnitudes,
        parameters,
        location=True,
        subset_epochs=None,
        by_trial_lkh=False,
    ):
        """Estimate probabilities.

        Parameters
        ----------
        magnitudes : ndarray
            2D ndarray n_events * components (or 3D iteration * n_events * n_components),
            initial conditions for events magnitudes. If magnitudes are estimated,
            the list provided is used as starting point,
            if magnitudes are fixed, magnitudes estimated will be the same as the one provided.
            When providing a list, magnitudes need to be in the same order
            _n_th magnitudes parameter is  used for the _n_th event
        parameters : list
            list of initial conditions for Gamma distribution parameters parameter
            (2D stage * parameter or 3D iteration * n_events * n_components).
            If parameters are estimated, the list provided is used as starting point,
            if parameters are fixed, parameters estimated will be the same as the one provided.
            When providing a list, stage need to be in the same order
            _n_th gamma parameter is  used for the _n_th stage
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
        n_events = magnitudes.shape[0]
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
                + trial_data.cross_corr[:, i][np.newaxis].T * magnitudes[:, i]
                - magnitudes[:, i] ** 2 / 2
            )
        gains = np.exp(gains)
        probs = np.zeros([max_duration, n_trials, n_events], dtype=np.float64)  # prob per trial
        probs_b = np.zeros(
            [max_duration, n_trials, n_events], dtype=np.float64
        )  # Sample and state reversed
        for trial in np.arange(n_trials):
            # Following assigns gain per trial to variable probs
            probs[: durations[trial], trial, :] = gains[starts[trial] : ends[trial] + 1, :]
            # Same but samples and events are reversed, this allows to compute
            # fwd and bwd in the same way in the following steps
            probs_b[: durations[trial], trial, :] = gains[starts[trial] : ends[trial] + 1, :][
                ::-1, ::-1
            ]

        pmf = np.zeros([max_duration, n_stages], dtype=np.float64)  # Gamma pmf for each stage scale
        for stage in range(n_stages):
            pmf[:, stage] = np.concatenate(
                (
                    np.repeat(1e-15, locations[stage]),
                    self.distribution_pmf(parameters[stage, 0], parameters[stage, 1], max_duration)[
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
        self, trial_data, magnitudes, parameters, mags_map, pars_map, levels, 
        location=True, cpus=1
    ):
        """Estimate probability levels.

        Parameters
        ----------
        magnitudes : ndarray
            2D ndarray n_events * components (or 3D iteration * n_events * n_components),
            initial conditions for events magnitudes. If magnitudes are estimated,
            the list provided is used as starting point,
            if magnitudes are fixed, magnitudes estimated will be the same as the one provided.
            When providing a list, magnitudes need to be in the same order
            _n_th magnitudes parameter is  used for the _n_th event
        parameters : list
            list of initial conditions for Gamma distribution parameters parameter
            (2D stage * parameter or 3D iteration * n_events * n_components).
            If parameters are estimated, the list provided is used as starting point,
            if parameters are fixed, parameters estimated will be the same as the one provided.
            When providing a list, stage need to be in the same order
            _n_th gamma parameter is  used for the _n_th stage
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
                        [magnitudes[cur_level, mags_map[cur_level, :] >= 0, :] for cur_level in data_levels],
                        [parameters[cur_level, pars_map[cur_level, :] >= 0, :] for cur_level in data_levels],
                        itertools.repeat(location),
                        [levels == cur_level for cur_level in data_levels],
                        itertools.repeat(False),
                    ),
                )
        else:
            for cur_level in data_levels:
                magnitudes_level = magnitudes[
                    cur_level, mags_map[cur_level, :] >= 0, :
                ]  # select existing magnitudes
                parameters_level = parameters[cur_level, pars_map[cur_level, :] >= 0, :]  # select existing params
                likes_events_level.append(
                    self.estim_probs(
                        trial_data,
                        magnitudes_level,
                        parameters_level,
                        location,
                        subset_epochs=(levels == cur_level),
                    )
                )

        likelihood = np.array([x[0] for x in likes_events_level])

        for i, cur_level in enumerate(data_levels):
            part = trial_data.coords["participant"].values[(levels == cur_level)]
            trial = trial_data.coords["trials"].values[(levels == cur_level)]
            data_events =  mags_map[cur_level, :] >= 0
            trial_x_part = xr.Coordinates.from_pandas_multiindex(
                MultiIndex.from_arrays([part, trial], names=("participant", "trials")),
                "trial_x_participant",
            )
            xreventprobs = xr.DataArray(likes_events_level[i][1], dims=("trial_x_participant", "samples", "event"),
                coords={
                    "event": ("event", np.arange(self.n_events)[data_events]),
                    "samples": ("samples", range(np.shape(likes_events_level[i][1])[1])),
                },
            )
            xreventprobs = xreventprobs.assign_coords(trial_x_part)
            xreventprobs = xreventprobs.assign_coords(levels=("trial_x_participant", levels[levels == cur_level],))
            all_xreventprobs.append(xreventprobs)
        all_xreventprobs = xr.concat(all_xreventprobs,dim="trial_x_participant")
        all_xreventprobs.attrs['sfreq'] = self.sfreq
        all_xreventprobs.attrs['event_width_samples'] = self.event_width_samples
        return [np.array(likelihood), all_xreventprobs]

    def distribution_pmf(self, shape, scale, max_duration):
        """Return PMF for a provided scipy disttribution.

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
        p = self.pdf(np.arange(max_duration), shape, scale=scale)
        p = p / np.sum(p)
        p[np.isnan(p)] = 0  # remove potential nans
        return p

    def level_constructor(self, trial_data, level_dict, mags_map=None, pars_map=None, verbose=False):
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
        if mags_map is not None and pars_map is not None:
            n_levels_mags = 0 if mags_map is None else mags_map.shape[0]
            n_levels_pars = 0 if pars_map is None else pars_map.shape[0]
            if (
                n_levels_mags > 0 and n_levels_pars > 0
            ):  # either both maps should have the same number of levels, or 0
                assert n_levels_mags == n_levels_pars, (
                    "magnitude and parameters maps have to indicate the same number of levels"
                )
                # make sure nr of events correspond per row
                for cur_level in range(n_levels):
                    assert sum(mags_map[cur_level, :] >= 0) + 1 == sum(pars_map[cur_level, :] >= 0), (
                        "nr of events in magnitudes map and parameters map do not correspond on row "
                        + str(cur_level)
                    )
            elif n_levels_mags == 0:
                assert not (pars_map < 0).any(), (
                    "If negative parameters are provided, magnitude map is required."
                )
                mags_map = np.zeros((n_levels, pars_map.shape[1] - 1), dtype=int)
            else:
                pars_map = np.zeros((n_levels, mags_map.shape[1] + 1), dtype=int)
                if (mags_map < 0).any():
                    for cur_level in range(n_levels):
                        pars_map[cur_level, np.where(mags_map[cur_level, :] < 0)[0]] = -1
                        pars_map[cur_level, np.where(mags_map[cur_level, :] < 0)[0] + 1] = 1
    
            # at this point, all should indicate the same number of levels
            assert n_levels == mags_map.shape[0] == pars_map.shape[0], (
                "number of unique levels should correspond to number of rows in map(s)"
            )
    
            if verbose:
                print("\nMagnitudes map:")
                for cnt in range(n_levels):
                    print(str(cnt) + ": ", mags_map[cnt, :])
    
                print("\nParameters map:")
                for cnt in range(n_levels):
                    print(str(cnt) + ": ", pars_map[cnt, :])

            # at this point, all should indicate the same number of levels
            assert n_levels == mags_map.shape[0] == pars_map.shape[0], (
                "number of unique levels should correspond to number of rows in map(s)"
            )

        return n_levels, levels, clabels
