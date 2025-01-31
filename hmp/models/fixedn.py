import gc
import itertools
import multiprocessing as mp
from abc import ABC, abstractmethod
from itertools import cycle, product
from warnings import resetwarnings, warn

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pandas import MultiIndex
from scipy.signal import correlate
from scipy.stats import norm as norm_pval

from hmp.models.base import BaseModel

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm



class FixedEventModel(BaseModel):
    def fit(
        self,
        n_events=None,
        magnitudes=None,
        parameters=None,
        parameters_to_fix=None,
        magnitudes_to_fix=None,
        tolerance=1e-4,
        max_iteration=1e3,
        maximization=True,
        min_iteration=1,
        starting_points=1,
        return_max=True,
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
            list of initial conditions for Gamma distribution parameters parameter
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
        maximization: bool
            If True (Default) perform the maximization phase in EM() otherwhise skip
        min_iteration: int
            Minimum number of iteration for the expectation maximization in the EM() function
        starting_points: int
            How many starting points to use for the EM() function
        return_max: bool
            In the case of multiple starting points, dictates whether to only return
            the max loglikelihood model (True, default) or all of the models (False)
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
        """
        # A dict containing all the info we want to keep, populated along the func
        infos_to_store = {}
        infos_to_store["sfreq"] = self.sfreq
        infos_to_store["event_width_samples"] = self.event_width_samples
        infos_to_store["tolerance"] = tolerance
        infos_to_store["maximization"] = int(maximization)

        if n_events is None:
            if parameters is not None:
                n_events = len(parameters) - 1
            elif magnitudes is not None:
                n_events = len(magnitudes)
            else:
                raise ValueError(
                    "The fit_n() function needs to be provided with a number of expected transition"
                    " events"
                )
        assert n_events <= self.compute_max_events(), (
            f"{n_events} events do not fit given the minimum duration of {min(self.durations)}"
            " and a location of {self.location}"
        )

        if level_dict is None:
            (
                n_levels,
                levels,
            ) = 1, np.zeros(self.n_trials)
            pars_map, mags_map = np.zeros((1, n_events + 1)), np.zeros((1, n_events))
        else:
            n_levels, levels, clabels, pars_map, mags_map = self._level_constructor(
                magnitudes, parameters, mags_map, pars_map, level_dict, verbose
            )
            infos_to_store["mags_map"] = mags_map
            infos_to_store["pars_map"] = pars_map
            infos_to_store["clabels"] = clabels
            infos_to_store["level_dict"] = level_dict
        if verbose:
            if parameters is None:
                print(
                    f"Estimating {n_events} events model with {starting_points} starting point(s)"
                )
            else:
                print(f"Estimating {n_events} events model")
        # if cpus is None:
            # cpus = self.cpus

        # Formatting parameters
        if isinstance(parameters, (xr.DataArray, xr.Dataset)):
            parameters = parameters.dropna(dim="stage").values
        if isinstance(magnitudes, (xr.DataArray, xr.Dataset)):
            magnitudes = magnitudes.dropna(dim="event").values
        if isinstance(magnitudes, np.ndarray):
            magnitudes = magnitudes.copy()
        if isinstance(parameters, np.ndarray):
            parameters = parameters.copy()
        if parameters_to_fix is None:
            parameters_to_fix = []
        else:
            infos_to_store["parameters_to_fix"] = parameters_to_fix
        if magnitudes_to_fix is None:
            magnitudes_to_fix = []
        else:
            infos_to_store["magnitudes_to_fix"] = magnitudes_to_fix

        if parameters is None:
            parameters = (
                np.zeros((n_levels, n_events + 1, 2)) * np.nan
            )  # by default nan for missing stages
            for c in range(n_levels):
                pars_level = np.where(pars_map[c, :] >= 0)[0]
                n_stage_level = len(pars_level)
                # by default starting point is to split the average duration in equal bins
                parameters[c, pars_level, :] = np.tile(
                    [
                        self.shape,
                        self.mean_to_scale(
                            np.mean(self.durations[levels == c]) / (n_stage_level), self.shape
                        ),
                    ],
                    (n_stage_level, 1),
                )
        else:
            infos_to_store["sp_parameters"] = parameters
            if len(np.shape(parameters)) == 2:  # broadcast provided parameters across levels
                parameters = np.tile(parameters, (n_levels, 1, 1))
            assert parameters.shape[1] == n_events + 1, (
                f"Provided parameters ({parameters.shape[1]} should match number of "
                f"stages {n_events + 1}"
            )

            # set params missing stages to nan to make it obvious in the results
            if (pars_map < 0).any():
                for c in range(n_levels):
                    parameters[c, np.where(pars_map[c, :] < 0)[0], :] = np.nan

        if magnitudes is None:
            # By defaults mags are initiated to 0
            magnitudes = np.zeros((n_levels, n_events, self.n_dims), dtype=np.float64)
            if (mags_map < 0).any():  # set missing mags to nan
                for c in range(n_levels):
                    magnitudes[c, np.where(mags_map[c, :] < 0)[0], :] = np.nan
        else:
            infos_to_store["sp_magnitudes"] = magnitudes
            if len(np.shape(magnitudes)) == 2:  # broadcast provided magnitudes across levels
                magnitudes = np.tile(magnitudes, (n_levels, 1, 1))
            assert magnitudes.shape[1] == n_events, (
                "Provided magnitudes should match number of events in magnitudes map"
            )

            # set mags missing events to nan to make it obvious in the results
            if (mags_map < 0).any():
                for c in range(n_levels):
                    magnitudes[c, np.where(mags_map[c, :] < 0)[0], :] = np.nan
        initial_p = parameters
        initial_m = magnitudes
        parameters = [initial_p]
        magnitudes = np.tile(initial_m, (starting_points + 1, 1, 1, 1))
        if starting_points > 1:
            infos_to_store["starting_points"] = starting_points
            for _ in np.arange(starting_points):
                proposal_p = (
                    np.zeros((n_levels, n_events + 1, 2)) * np.nan
                )  # by default nan for missing stages
                for c in range(n_levels):
                    pars_level = np.where(pars_map[c, :] >= 0)[0]
                    n_stage_level = len(pars_level)
                    proposal_p[c, pars_level, :] = self.gen_random_stages(n_stage_level - 1)
                    proposal_p[c, parameters_to_fix, :] = initial_p[0, parameters_to_fix]
                parameters.append(proposal_p)
            parameters = np.array(parameters)

        if cpus > 1:
            inputs = zip(
                magnitudes,
                parameters,
                itertools.repeat(maximization),
                itertools.repeat(magnitudes_to_fix),
                itertools.repeat(parameters_to_fix),
                itertools.repeat(max_iteration),
                itertools.repeat(tolerance),
                itertools.repeat(min_iteration),
                itertools.repeat(mags_map),
                itertools.repeat(pars_map),
                itertools.repeat(levels),
                itertools.repeat(1),
            )
            with mp.Pool(processes=cpus) as pool:
                if starting_points > 1:
                    estimates = list(tqdm(pool.imap(self._EM_star, inputs), total=len(magnitudes)))
                else:
                    estimates = pool.starmap(self.EM, inputs)

        else:  # avoids problems if called in an already parallel function
            estimates = []
            for pars, mags in zip(parameters, magnitudes):
                estimates.append(
                    self.EM(
                        mags,
                        pars,
                        maximization,
                        magnitudes_to_fix,
                        parameters_to_fix,
                        max_iteration,
                        tolerance,
                        min_iteration,
                        mags_map,
                        pars_map,
                        levels,
                        1,
                    )
                )
            resetwarnings()

        lkhs_sp = [x[0] for x in estimates]
        mags_sp = [x[1] for x in estimates]
        pars_sp = [x[2] for x in estimates]
        eventprobs_sp = [x[3] for x in estimates]
        traces_sp = [x[4] for x in estimates]
        param_dev_sp = [x[5] for x in estimates]
        if starting_points > 1 and return_max:
            max_lkhs = np.argmax(lkhs_sp)
            lkhs_sp = [lkhs_sp[max_lkhs]]
            mags_sp = [mags_sp[max_lkhs]]
            pars_sp = [pars_sp[max_lkhs]]
            eventprobs_sp = [eventprobs_sp[max_lkhs]]
            traces_sp = [traces_sp[max_lkhs]]
            param_dev_sp = [param_dev_sp[max_lkhs]]
            starting_points = 1

        # make output object
        estimated = []
        for sp in range(starting_points):
            xrlikelihoods = xr.DataArray(lkhs_sp[sp], name="loglikelihood")
            xrtraces = xr.DataArray(
                traces_sp[sp],
                dims=("em_iteration"),
                name="traces",
                coords={"em_iteration": range(len(traces_sp[sp]))},
            )
            xrparam_dev = xr.DataArray(
                param_dev_sp[sp],
                dims=("em_iteration", "level", "stage", "parameter"),
                name="param_dev",
                coords=[
                    range(len(param_dev_sp[sp])),
                    range(n_levels),
                    range(n_events + 1),
                    ["shape", "scale"],
                ],
            )
            xrparams = xr.DataArray(
                pars_sp[sp],
                dims=("level", "stage", "parameter"),
                name="parameters",
                coords={
                    "level": range(n_levels),
                    "stage": range(n_events + 1),
                    "parameter": ["shape", "scale"],
                },
            )
            xrmags = xr.DataArray(
                mags_sp[sp],
                dims=("level", "event", "component"),
                name="magnitudes",
                coords={
                    "level": range(n_levels),
                    "event": range(n_events),
                    "component": range(self.n_dims),
                },
            )
            part, trial = self.coords["participant"].values, self.coords["trials"].values
            trial_x_part = xr.Coordinates.from_pandas_multiindex(
                MultiIndex.from_arrays([part, trial], names=("participant", "trials")),
                "trial_x_participant",
            )
            xreventprobs = xr.Dataset(
                {"eventprobs": (("event", "trial_x_participant", "samples"), eventprobs_sp[sp].T)},
                {
                    "event": ("event", range(n_events)),
                    "samples": ("samples", range(np.shape(eventprobs_sp[sp])[0])),
                },
            )
            xreventprobs = xreventprobs.assign_coords(trial_x_part)
            if n_levels > 1:
                xreventprobs = xreventprobs.assign_coords(levels=("trial_x_participant", levels))
            xreventprobs = xreventprobs.transpose("trial_x_participant", "samples", "event")

            estimated.append(
                xr.merge((xrlikelihoods, xrparams, xrmags, xreventprobs, xrtraces, xrparam_dev))
            )

        if starting_points > 1:
            estimated = xr.concat(estimated, "starting_points")
            estimated = estimated.assign_coords(
                starting_points=("starting_points", np.arange(starting_points))
            )
        else:
            estimated = estimated[0]

        # Adding infos
        estimated = estimated.assign_coords(rts=("trial_x_participant", self.named_durations.data))

        for x, y in infos_to_store.items():
            estimated.attrs[x] = y

        if n_levels == 1:
            estimated = estimated.squeeze(dim="level")
            # Drops empty coords incuced by squeeze
            estimated = estimated.drop_vars(
                lambda x: [v for v, da in x.coords.items() if not da.dims]
            )

        if verbose:
            print(f"parameters estimated for {n_events} events model")
        return estimated


    def _EM_star(self, args):  # for tqdm usage  #noqa
        return self.EM(*args)

    def EM(  #noqa
        self,
        magnitudes,
        parameters,
        maximization=True,
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
        maximization: bool
            If True (Default) perform the maximization phase in EM() otherwhise skip
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
        n_levels = mags_map.shape[0]

        n_events = magnitudes.shape[magnitudes.ndim - 2]
        locations = np.zeros((n_events + 1,), dtype=int)  # location per stage
        locations[1:-1] = self.location
        locations = np.tile(locations, (n_levels, 1))
        lkh, eventprobs = self._estim_probs_levels(
            magnitudes, parameters, locations, mags_map, pars_map, levels, cpus=cpus
        )
        initial_magnitudes = magnitudes.copy()
        initial_parameters = parameters.copy()

        traces = [lkh]
        param_dev = [parameters.copy()]  # ... and parameters
        i = 0
        if not maximization:
            lkh_prev = lkh
        else:
            lkh_prev = lkh
            parameters_prev = parameters.copy()

            while i < max_iteration:  # Expectation-Maximization algorithm
                if i >= min_iteration and (
                    np.isneginf(lkh) or tolerance > (lkh - lkh_prev) / np.abs(lkh_prev)
                ):
                    break

                # As long as new run gives better likelihood, go on
                lkh_prev = lkh.copy()
                parameters_prev = parameters.copy()

                for c in range(n_levels):  # get params/mags
                    mags_map_level = np.where(mags_map[c, :] >= 0)[0]
                    pars_map_level = np.where(pars_map[c, :] >= 0)[0]
                    epochs_level = np.where(levels == c)[0]

                    # get mags/pars by level
                    magnitudes[c, mags_map_level, :], parameters[c, pars_map_level, :] = (
                        self.get_magnitudes_parameters_expectation(
                            eventprobs[np.ix_(range(self.max_d), epochs_level, mags_map_level)],
                            subset_epochs=epochs_level,
                        )
                    )

                    magnitudes[c, magnitudes_to_fix, :] = initial_magnitudes[
                        c, magnitudes_to_fix, :
                    ].copy()
                    parameters[c, parameters_to_fix, :] = initial_parameters[
                        c, parameters_to_fix, :
                    ].copy()

                # set mags to mean if requested in map
                for m in range(n_events):
                    for m_set in np.unique(mags_map[:, m]):
                        if m_set >= 0:
                            magnitudes[mags_map[:, m] == m_set, m, :] = np.mean(
                                magnitudes[mags_map[:, m] == m_set, m, :], axis=0
                            )

                # set param to mean if requested in map
                for p in range(n_events + 1):
                    for p_set in np.unique(pars_map[:, p]):
                        if p_set >= 0:
                            parameters[pars_map[:, p] == p_set, p, :] = np.mean(
                                parameters[pars_map[:, p] == p_set, p, :], axis=0
                            )
                lkh, eventprobs = self._estim_probs_levels(
                    magnitudes, parameters, locations, mags_map, pars_map, levels, cpus=cpus
                )
                traces.append(lkh)
                param_dev.append(parameters.copy())
                i += 1
        _, eventprobs = self._estim_probs_levels(
            magnitudes,
            parameters,
            np.zeros(locations.shape).astype(int),
            mags_map,
            pars_map,
            levels,
            cpus=cpus,
        )
        if i == max_iteration:
            warn(
                f"Convergence failed, estimation hitted the maximum number of iteration "
                f"({int(max_iteration)})",
                RuntimeWarning,
            )
        return lkh, magnitudes, parameters, eventprobs, np.array(traces), np.array(param_dev)

    def get_magnitudes_parameters_expectation(self, eventprobs, subset_epochs=None):
        n_events = eventprobs.shape[2]
        n_trials = eventprobs.shape[1]
        if subset_epochs is None:  # all trials
            subset_epochs = range(n_trials)

        magnitudes = np.zeros((n_events, self.n_dims))

        # Magnitudes from Expectation, Eq 11 from 2024 paper
        for event in range(n_events):
            for comp in range(self.n_dims):
                event_data = np.zeros((self.max_d, len(subset_epochs)))
                for trial_idx, trial in enumerate(subset_epochs):
                    start, end = self.starts[trial], self.ends[trial]
                    duration = end - start + 1
                    event_data[:duration, trial_idx] = self.crosscorr[start : end + 1, comp]
                magnitudes[event, comp] = np.mean(
                    np.sum(eventprobs[:, :, event] * event_data, axis=0)
                )
            # scale cross-correlation with likelihood of the transition
            # sum by-trial these scaled activation for each transition events
            # average across trials

        # Gamma parameters from Expectation Eq 10 from 2024 paper
        # calc averagepos here as mean_d can be level dependent, whereas scale_parameters() assumes
        # it's general
        event_times_mean = np.concatenate(
            [
                np.arange(self.max_d) @ eventprobs.mean(axis=1),
                [np.mean(self.durations[subset_epochs]) - 1],
            ]
        )
        parameters = self.scale_parameters(
            eventprobs=None, n_events=n_events, averagepos=event_times_mean
        )

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
        mean_d = int(self.mean_d)
        rnd_durations = np.zeros(n_events + 1)
        while any(rnd_durations < self.event_width_samples):  # at least event_width
            rnd_events = np.random.default_rng().integers(
                low=0, high=mean_d, size=n_events
            )  # n_events between 0 and mean_d
            rnd_events = np.sort(rnd_events)
            rnd_durations = np.hstack((rnd_events, mean_d)) - np.hstack(
                (0, rnd_events)
            )  # associated durations
        random_stages = np.array(
            [[self.shape, self.mean_to_scale(x, self.shape)] for x in rnd_durations]
        )
        return random_stages


    def scale_parameters(self, eventprobs=None, n_events=None, averagepos=None):
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
        params = np.zeros((n_events + 1, 2), dtype=np.float64)
        params[:, 0] = self.shape
        params[:, 1] = np.diff(averagepos, prepend=0)
        params[:, 1] = [self.mean_to_scale(x[1], x[0]) for x in params]
        return params

    def estim_probs(
        self,
        magnitudes,
        parameters,
        locations,
        n_events=None,
        subset_epochs=None,
        lkh_only=False,
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
        n_events : int
            how many events are estimated
        subset_epochs : list
            boolean array indicating which epoch should be taken into account for level-based calcs
        lkh_only: bool
            Returning eventprobs (True) or not (False)

        Returns
        -------
        loglikelihood : float
            Summed log probabilities
        eventprobs : ndarray
            Probabilities with shape max_samples*n_trials*n_events
        """
        if n_events is None:
            n_events = magnitudes.shape[0]
        n_stages = n_events + 1

        if subset_epochs is not None:
            if len(subset_epochs) == self.n_trials:  # boolean indices
                subset_epochs = np.where(subset_epochs)[0]
            n_trials = len(subset_epochs)
            durations = self.durations[subset_epochs]
            starts = self.starts[subset_epochs]
            ends = self.ends[subset_epochs]
        else:
            n_trials = self.n_trials
            durations = self.durations
            starts = self.starts
            ends = self.ends

        gains = np.zeros((self.n_samples, n_events), dtype=np.float64)
        for i in range(self.n_dims):
            # computes the gains, i.e. congruence between the pattern shape
            # and the data given the magnitudes of the sensors
            gains = (
                gains
                + self.crosscorr[:, i][np.newaxis].T * magnitudes[:, i]
                - magnitudes[:, i] ** 2 / 2
            )
        gains = np.exp(gains)
        probs = np.zeros([self.max_d, n_trials, n_events], dtype=np.float64)  # prob per trial
        probs_b = np.zeros(
            [self.max_d, n_trials, n_events], dtype=np.float64
        )  # Sample and state reversed
        for trial in np.arange(n_trials):
            # Following assigns gain per trial to variable probs
            probs[: durations[trial], trial, :] = gains[starts[trial] : ends[trial] + 1, :]
            # Same but samples and events are reversed, this allows to compute
            # fwd and bwd in the same way in the following steps
            probs_b[: durations[trial], trial, :] = gains[starts[trial] : ends[trial] + 1, :][
                ::-1, ::-1
            ]

        pmf = np.zeros([self.max_d, n_stages], dtype=np.float64)  # Gamma pmf for each stage scale
        for stage in range(n_stages):
            pmf[:, stage] = np.concatenate(
                (
                    np.repeat(0, locations[stage]),
                    self.distribution_pmf(parameters[stage, 0], parameters[stage, 1])[
                        locations[stage] :
                    ],
                )
            )
        pmf_b = pmf[:, ::-1]  # Stage reversed gamma pmf, same order as prob_b

        forward = np.zeros((self.max_d, n_trials, n_events), dtype=np.float64)
        backward = np.zeros((self.max_d, n_trials, n_events), dtype=np.float64)
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
                    : self.max_d
                ]
                # same but backwards
                backward[:, trial, event] = np.convolve(add_b[:, trial], pmf_b[:, event])[
                    : self.max_d
                ]
            forward[:, :, event] = forward[:, :, event] * probs[:, :, event]
        # re-arranging backward to the expected variable
        backward = backward[:, :, ::-1]  # undoes stage inversion
        for trial in np.arange(n_trials):  # Undoes sample inversion
            backward[: durations[trial], trial, :] = backward[: durations[trial], trial, :][::-1]

        eventprobs = forward * backward
        eventprobs = np.clip(eventprobs, 0, None)  # floating point precision error
        # eventprobs can be so low as to be 0, avoid dividing by 0
        # this only happens when magnitudes are 0 and gammas are randomly determined
        if (eventprobs.sum(axis=0) == 0).any() or (eventprobs[:, :, 0].sum(axis=0) == 0).any():
            # set likelihood
            eventsums = eventprobs[:, :, 0].sum(axis=0)
            eventsums[eventsums != 0] = np.log(eventsums[eventsums != 0])
            eventsums[eventsums == 0] = -np.inf
            likelihood = np.sum(eventsums)

            # set eventprobs, check if any are 0
            eventsums = eventprobs.sum(axis=0)
            if (eventsums == 0).any():
                for i in range(eventprobs.shape[0]):
                    eventprobs[i, :, :][eventsums == 0] = 0
                    eventprobs[i, :, :][eventsums != 0] = (
                        eventprobs[i, :, :][eventsums != 0] / eventsums[eventsums != 0]
                    )
            else:
                eventprobs = eventprobs / eventprobs.sum(axis=0)

        else:
            likelihood = np.sum(
                np.log(eventprobs[:, :, 0].sum(axis=0))
            )  # sum over max_samples to avoid 0s in log
            eventprobs = eventprobs / eventprobs.sum(axis=0)

        if lkh_only:
            return likelihood
        elif by_trial_lkh:
            return forward * backward
        else:
            return [likelihood, eventprobs]

    def _estim_probs_levels(
        self, magnitudes, parameters, locations, mags_map, pars_map, levels, lkh_only=False, cpus=1
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
        locations : ndarray
            2D n_level * n_events array indication locations for all events
        n_events : int
            how many events are estimated
        lkh_only: bool
            Returning eventprobs (True) or not (False)

        Returns
        -------
        loglikelihood : float
            Summed log probabilities
        eventprobs : ndarray
            Probabilities with shape max_samples*n_trials*n_events
        """
        n_levels = mags_map.shape[0]
        likes_events_level = []
        if cpus > 1:
            with mp.Pool(processes=cpus) as pool:
                likes_events_level = pool.starmap(
                    self.estim_probs,
                    zip(
                        [magnitudes[c, mags_map[c, :] >= 0, :] for c in range(n_levels)],
                        [parameters[c, pars_map[c, :] >= 0, :] for c in range(n_levels)],
                        [locations[c, pars_map[c, :] >= 0] for c in range(n_levels)],
                        itertools.repeat(None),
                        [levels == c for c in range(n_levels)],
                        itertools.repeat(False),
                    ),
                )
        else:
            for c in range(n_levels):
                magnitudes_level = magnitudes[
                    c, mags_map[c, :] >= 0, :
                ]  # select existing magnitudes
                parameters_level = parameters[c, pars_map[c, :] >= 0, :]  # select existing params
                likes_events_level.append(
                    self.estim_probs(
                        magnitudes_level,
                        parameters_level,
                        locations[c, pars_map[c, :] >= 0],
                        subset_epochs=(levels == c),
                    )
                )

        likelihood = np.sum([x[0] for x in likes_events_level])
        eventprobs = np.zeros((self.max_d, len(levels), mags_map.shape[1]))
        for c in range(n_levels):
            eventprobs[np.ix_(range(self.max_d), levels == c, mags_map[c, :] >= 0)] = (
                likes_events_level[c][1]
            )

        if lkh_only:
            return likelihood
        else:
            return [likelihood, eventprobs]

    def distribution_pmf(self, shape, scale):
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
        p = self.pdf(np.arange(self.max_d), shape, scale=scale)
        p = p / np.sum(p)
        p[np.isnan(p)] = 0  # remove potential nans
        return p

    def _level_constructor(self, magnitudes, parameters, mags_map, pars_map, level_dict, verbose):
        """Adapt model to levels."""
        ## levels
        assert isinstance(level_dict, dict), "levels have to be specified as a dictionary"

        # collect level names, levels, and trial coding
        level_names = []
        level_mods = []
        level_trials = []
        for level in level_dict.keys():
            level_names.append(level)
            level_mods.append(level_dict[level])
            level_trials.append(self.trial_coords[level])
            if verbose:
                print('Level "' + level_names[-1] + '" analyzed, with levels:', level_mods[-1])

        level_mods = list(product(*level_mods))
        level_mods = np.array(level_mods, dtype=object)
        print(level_mods)
        n_levels = len(level_mods)

        # build level array with digit indicating the combined levels
        level_trials = np.vstack(level_trials).T
        levels = np.zeros((level_trials.shape[0])) * np.nan
        if verbose:
            print("\nCoded as follows: ")
        for i, mod in enumerate(level_mods):
            assert len(np.where((level_trials == mod).all(axis=1))[0]) > 0, (
                f"Modality {mod} of level does not occur in the data"
            )
            levels[np.where((level_trials == mod).all(axis=1))] = i
            if verbose:
                print(str(i) + ": " + str(level))
        levels = np.int8(levels)
        clabels = {"level " + str(level_names): level_mods}

        # check maps
        n_levels_mags = 0 if mags_map is None else mags_map.shape[0]
        n_levels_pars = 0 if pars_map is None else pars_map.shape[0]
        if (
            n_levels_mags > 0 and n_levels_pars > 0
        ):  # either both maps should have the same number of levels, or 0
            assert n_levels_mags == n_levels_pars, (
                "magnitude and parameters maps have to indicate the same number of levels"
            )
            # make sure nr of events correspond per row
            for c in range(n_levels):
                assert sum(mags_map[c, :] >= 0) + 1 == sum(pars_map[c, :] >= 0), (
                    "nr of events in magnitudes map and parameters map do not correspond on row "
                    + str(c)
                )
        elif n_levels_mags == 0:
            assert not (pars_map < 0).any(), (
                "If negative parameters are provided, magnitude map is required."
            )
            mags_map = np.zeros((n_levels, pars_map.shape[1] - 1), dtype=int)
        else:
            pars_map = np.zeros((n_levels, mags_map.shape[1] + 1), dtype=int)
            if (mags_map < 0).any():
                for c in range(n_levels):
                    pars_map[c, np.where(mags_map[c, :] < 0)[0]] = -1
                    pars_map[c, np.where(mags_map[c, :] < 0)[0] + 1] = 1

        # print maps to check level/row mathcing
        if verbose:
            print("\nMagnitudes map:")
            for cnt in range(n_levels):
                print(str(cnt) + ": ", mags_map[cnt, :])

            print("\nParameters map:")
            for cnt in range(n_levels):
                print(str(cnt) + ": ", pars_map[cnt, :])

            # give explanation if negative parameters:
            if (pars_map < 0).any():
                print("\n-----")
                print("Negative parameters. Note that this stage is left out, while the parameters")
                print(
                    "of the other stages are compared column by column. "
                    "In this parameter map example:"
                )
                print(np.array([[0, 0, 0, 0], [0, -1, 0, 0]]))
                print(
                    "the parameters of stage 1 are shared, as well as the parameters of stage 3 of"
                )
                print("level 1 with stage 2 (column 3) of level 2 and the last stage of both")
                print("levels.")
                print("Given that event 2 is probably missing in level 2, it would typically")
                print("make more sense to let both stages around event 2 in level 1 vary as")
                print("compared to level 2:")
                print(np.array([[0, 0, 0, 0], [0, -1, 1, 0]]))
                print("-----")

        # at this point, all should indicate the same number of levels
        assert n_levels == mags_map.shape[0] == pars_map.shape[0], (
            "number of unique levels should correspond to number of rows in map(s)"
        )

        assert levels.shape[0] == self.durations.shape[0], (
            "levels parameter should contain the level per epoch."
        )
        return n_levels, levels, clabels, pars_map, mags_map
