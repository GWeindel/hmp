"""Expectation-Maximization estimator for HMP models."""

import multiprocessing as mp
from warnings import resetwarnings, warn

import numpy as np

from hmp.patterndata import PatternData
from hmp.utils import _get_mp_context

from .base import BaseEstimator, EstimationResult

_WORKER_DATA = {}

def _init_worker(pattern_data: PatternData, model):
    _WORKER_DATA["pattern_data"] = pattern_data
    _WORKER_DATA["model"] = model


def worker_estim_probs(channel_pars, time_pars, chunk, max_duration):
    """Worker function to estimate probabilities of a chunk of trials."""
    return _WORKER_DATA["model"].estim_probs(
        _WORKER_DATA["pattern_data"],
        channel_pars,
        time_pars,
        subset_epochs=chunk,
        max_duration=max_duration,
    )

class EMEstimator(BaseEstimator):
    """Expectation-Maximization parameter estimator.

    Implements the EM algorithm for HMP models, providing estimates of channel
    contributions and time distribution parameters that maximize the
    log-likelihood plus the model's log-prior on the time parameters
    (which is the log-likelihood if the prior is turned off. See ``EventModel.log_prior``).

    The estimator owns the optimization: the loop over starting points, its
    parallelization, the EM iterations themselves, and the selection of the
    best solution. The model owns data preparation, grouping and the
    formatting of initial parameters.

    Parameters
    ----------
    tolerance : float, optional
        Relative improvement of the objective (log-likelihood plus log-prior)
        below which estimation stops.
        Default is 1e-4.
    max_iteration : int, optional
        Maximum number of EM iterations. Default is 1e3.
    min_iteration : int, optional
        Minimum number of EM iterations before convergence is checked.
        Default is 1.
    n_cor : int, optional
        Maximum number of step halvings used to keep parameter updates in a
        region where the likelihood is defined. Default is 30.
    """

    def __init__(self, tolerance: float = 1e-4, max_iteration: int = 1e3,
                 min_iteration: int = 1, n_cor: int = 30):
        super().__init__()
        self.tolerance = tolerance
        self.max_iteration = max_iteration
        self.min_iteration = min_iteration
        self.n_cor = n_cor

    def fit(
        self,
        model,
        pattern_data: PatternData,
        initial_channel_pars: np.ndarray,
        initial_time_pars: np.ndarray,
        groups: np.ndarray = None,
        cpus: int = 1,
    ) -> EstimationResult:
        """Estimate parameters by expectation-maximization.

        Parameters
        ----------
        model : EventModel
            Model providing the likelihood, the group maps and the
            expectation step.
        pattern_data : PatternData
            Preprocessed data cross-correlated with the pattern of the model.
        initial_channel_pars : np.ndarray
            4D ndarray (starting_points * n_groups * n_events * n_channels),
            initial conditions for event channel contributions.
        initial_time_pars : np.ndarray
            4D ndarray (starting_points * n_groups * n_stages * 2), initial
            conditions for time distribution parameters.
        groups : np.ndarray, optional
            Array indicating the groups for grouping modeling. Default is None.
        cpus : int, optional
            Number of cores to use in multiprocessing functions. Default is 1.

        Returns
        -------
        EstimationResult
            Best solution across starting points, i.e. with the highest objective
            (log-likelihood plus log-prior). ``diagnostics`` carries the
            per-iteration objective in ``traces`` and ``traces_group``, the
            ``time_pars_dev``, plus the log-likelihood and the objective of every
            starting point under ``lkhs`` and ``objectives``.
        """
        pool = None
        try:
            if cpus > 1:
                ctx = _get_mp_context()
                pool = ctx.Pool(processes=cpus, initializer=_init_worker,
                                initargs=(pattern_data, model))
            estimates = []
            for t_pars, c_pars in zip(initial_time_pars, initial_channel_pars):
                estimates.append(self.em(
                    model,
                    pattern_data,
                    c_pars,
                    t_pars,
                    groups,
                    cpus=cpus,
                    pool=pool,
                    )
                )

        finally:
            if pool is not None:
                pool.close()
                pool.join()

        resetwarnings()

        objectives = np.array([x[0] for x in estimates])
        lkhs = np.array([x[1] for x in estimates])
        best = int(np.argmax(objectives))

        if np.isneginf(objectives.sum()):
            warn("Fit failed, inspect provided starting points")

        _, lkh, channel_pars, time_pars, traces, traces_group, time_pars_dev, n_iter = \
            estimates[best]

        self.fitted = True
        return EstimationResult(
            channel_pars=channel_pars,
            time_pars=time_pars,
            likelihood=lkh,
            log_prior=model.log_prior(pattern_data, time_pars, groups),
            converged=n_iter < self.max_iteration,
            n_iterations=n_iter,
            diagnostics={
                "traces": traces,
                "traces_group": traces_group,
                "time_pars_dev": time_pars_dev,
                "lkhs": lkhs,
                "objectives": objectives,
            },
        )

    def em(  # noqa: PLR0912, PLR0915
        self,
        model,
        pattern_data: PatternData,
        initial_channel_pars: np.ndarray,
        initial_time_pars: np.ndarray,
        groups: np.ndarray = None,
        cpus: int = 1,
        pool: mp.Pool = None,
    ) -> tuple:
        """Run expectation-maximization from a single starting point.

        Parameters
        ----------
        model : EventModel
            Model providing the likelihood, the group maps and the
            expectation step.
        pattern_data : PatternData
            Preprocessed data cross-correlated with the pattern of the model.
        initial_channel_pars : np.ndarray
            3D ndarray (n_groups * n_events * n_channels), initial conditions
            for event channel contributions.
        initial_time_pars : np.ndarray
            3D ndarray (n_groups * n_stages * 2), initial conditions for time
            distribution parameters.
        groups : np.ndarray, optional
            Array indicating the groups for grouping modeling. Default is None.
        cpus : int, optional
            Number of cores to use in multiprocessing functions. Default is 1.
        pool : mp.Pool, optional
            Multiprocessing pool to use for parallelization. Default is None.

        Returns
        -------
        tuple
            ``(objective, lkh, channel_pars, time_pars, traces, traces_group,
            time_pars_dev, n_iterations)``, with ``objective`` the log-likelihood
            ``lkh`` plus the log-prior.
        """
        eventprobs = model.event_probabilities(
            pattern_data,
            initial_channel_pars, initial_time_pars,
            groups, cpus=cpus, pool=pool,
        )
        # EM maximizes thellk + log prior
        group_objective = eventprobs.group_lkh + model.log_prior(
            pattern_data, initial_time_pars, groups, per_group=True
        )
        objective = group_objective.sum()
        lkh = eventprobs.likelihood
        data_groups = np.unique(groups)
        channel_pars = initial_channel_pars.copy()
        time_pars = initial_time_pars.copy()
        traces = [objective]
        traces_group = [group_objective]
        time_pars_dev = [time_pars.copy()]
        i = 0

        objective_prev = objective.copy()
        while i < self.max_iteration:  # Expectation-Maximization algorithm
            if i >= self.min_iteration and (
                np.isneginf(objective)
                or self.tolerance > (objective - objective_prev) / np.abs(objective_prev)):
                break

            # As long as new run gives better objective, go on
            objective_prev = objective.copy()

            # Storage for step-length control
            new_channel_pars = channel_pars.copy()
            new_time_pars = time_pars.copy()

            for cur_group in data_groups:  # get params/c_pars
                channel_map_group = np.where(model.channel_map[cur_group, :] >= 0)[0]
                time_map_group = np.where(model.time_map[cur_group, :] >= 0)[0]
                epochs_group = np.where(groups == cur_group)[0]

                # get c_pars/t_pars by group
                c_par, t_par = model.get_channel_time_parameters_expectation(pattern_data,
                        eventprobs.values[:, :np.max(pattern_data.durations.values[epochs_group]),
                                        channel_map_group],
                                        subset_epochs=epochs_group)
                new_channel_pars[cur_group, channel_map_group, :] = c_par
                new_time_pars[cur_group, time_map_group, :] = t_par

                new_channel_pars[cur_group, model.fixed_channel_pars, :] = \
                    initial_channel_pars[cur_group, model.fixed_channel_pars, :].copy()
                new_time_pars[cur_group, model.fixed_time_pars, :] = \
                    initial_time_pars[cur_group, model.fixed_time_pars, :].copy()

            # set c_pars to mean if requested in map
            for m in range(model.n_events):
                for m_set in np.unique(model.channel_map[:, m]):
                    if m_set >= 0:
                        new_channel_pars[model.channel_map[:, m] == m_set, m, :] = np.mean(
                            new_channel_pars[model.channel_map[:, m] == m_set, m, :], axis=0
                        )

            # set param to mean if requested in map
            for p in range(model.n_events + 1):
                for p_set in np.unique(model.time_map[:, p]):
                    if p_set >= 0:
                        new_time_pars[model.time_map[:, p] == p_set, p, :] = np.mean(
                            new_time_pars[model.time_map[:, p] == p_set, p, :], axis=0
                        )

            # Step length control to ensure parameter updates result in valid objective
            for icor in range(self.n_cor + 1):
                if icor == self.n_cor:  # just reset
                    warn(
                        (
                            "M step failed, after step halvings. "
                            "Falling back to previous parameter estimates."
                        ),
                        RuntimeWarning,
                    )
                    new_channel_pars = channel_pars
                    new_time_pars = time_pars

                # Compute objective under new parameters
                with np.errstate(divide='ignore', invalid='ignore'):
                    eventprobs = model.event_probabilities(
                        pattern_data,
                        new_channel_pars, new_time_pars,
                        groups, cpus=cpus, pool=pool
                    )
                    group_objective = eventprobs.group_lkh + model.log_prior(
                        pattern_data, new_time_pars, groups, per_group=True
                    )
                    objective = group_objective.sum()
                    lkh = eventprobs.likelihood

                # Stop if no update
                if np.isclose((new_time_pars - time_pars).sum(), 0):
                    break

                # Half step in case the objective is ill-defined
                if np.isneginf(objective):
                    new_channel_pars = (new_channel_pars + channel_pars)/2
                    new_time_pars = (new_time_pars + time_pars)/2
                else:
                    # Accept step
                    break

            # Accept new parameters
            time_pars = new_time_pars
            channel_pars = new_channel_pars

            traces.append(objective)
            traces_group.append(group_objective)
            time_pars_dev.append(time_pars.copy())
            i += 1

        if i == self.max_iteration:
            warn(
                f"Convergence failed, estimation hit the maximum number of iterations: "
                f"({int(self.max_iteration)})",
                RuntimeWarning,
            )
        return objective, lkh, channel_pars, time_pars, np.array(traces), \
            np.array(traces_group), np.array(time_pars_dev), i
