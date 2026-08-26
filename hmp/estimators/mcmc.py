"""Sampling-based estimation of HMP parameters with PyMC.

Uses the same likelihood as EM, reimplemented in JAX so it can be
differentiated. See :mod:`hmp.estimators.jax_likelihood` and
:mod:`hmp.estimators.pytensor_op`.

Scales are sampled on a log scale to keep them positive; the shape of the
duration distribution is held at the model's value.

The priors are placeholders and have not been agreed for the method.
"""

import numpy as np

from hmp.estimators.base import BaseEstimator, EstimationResult
from hmp.estimators.em import EMEstimator
from hmp.estimators.pytensor_op import build_trial_op


def _trials_distribution(op):
    """Build a PyMC distribution whose log density is the per-trial log-likelihood.

    Registering the vector as an observed distribution, rather than adding its
    sum as a potential, is what populates ``idata.log_likelihood``. The
    parameters are shared across trials, so the number of observations is
    declared in ``_supp_shape_from_params``.
    """
    import pymc as pm  # noqa: PLC0415
    from pymc.logprob.abstract import _logprob  # noqa: PLC0415
    from pytensor.tensor.random.op import RandomVariable  # noqa: PLC0415

    n_trials = len(op.durations)

    class HMPTrialsRV(RandomVariable):
        name = "hmp_trials"
        signature = "(e,d),(s,p)->(t)"
        dtype = "floatX"
        _print_name = ("HMPTrials", "\\operatorname{HMPTrials}")

        def _supp_shape_from_params(self, dist_params, param_shapes=None):  # noqa: ARG002
            return (n_trials,)

        @classmethod
        def rng_fn(cls, *args, **kwargs):
            raise NotImplementedError(
                "HMP defines a likelihood through the forward-backward "
                "recursion and has no forward sampler, so draws from the prior "
                "predictive are not available."
            )

    class HMPTrials(pm.Distribution):
        rv_op = HMPTrialsRV()

        @classmethod
        def dist(cls, channel_pars, time_pars, **kwargs):
            return super().dist([channel_pars, time_pars], **kwargs)

    @_logprob.register(HMPTrialsRV)
    def _hmp_logprob(  # noqa: PLR0913, PLR0917
        rv_op, values, rng, size, channel_pars, time_pars, **kwargs  # noqa: ARG001
    ):
        return op(channel_pars, time_pars)

    return HMPTrials, n_trials


class MCMCEstimator(BaseEstimator):
    """Estimate HMP parameters by sampling the posterior.

    Parameters
    ----------
    draws : int, optional
        Posterior draws per chain. Default is 1000.
    tune : int, optional
        Tuning steps per chain, discarded. Default is 1000.
    chains : int, optional
        Number of chains. Default is 4.
    target_accept : float, optional
        NUTS target acceptance rate. Default is 0.9.
    channel_prior_sd : float, optional
        Standard deviation of the normal prior on channel contributions. If
        None, several times the spread of the cross-correlated data, which is
        wide enough not to shrink the estimates. Still provisional: the priors
        for this method have not been settled.
    init_from : str, optional
        Where the chains start. None uses the starting points the model
        supplies. "em" runs expectation maximization first and starts from its
        solution, which is the only setting under which the chains reliably
        find the same region of the parameter space: the likelihood is hard to
        explore from an arbitrary point, and a maximizer is better at getting
        into a sensible one than a sampler is. What the sampler then adds is
        the spread around it, which is what EM cannot give.
    init : str, optional
        Initialisation method for the default backend, as named by
        ``pm.init_nuts``: "adapt_diag", "adapt_full", "advi", "advi+adapt_diag",
        "advi_map", "map" and the jittered variants. The external samplers do
        not accept it and take ``jitter`` instead, so it is ignored there.
    jitter : bool, optional
        Whether to perturb each chain's starting point. Off by default: the
        perturbation is uniform on (-1, 1) in the unconstrained space, which on
        a log scale is a factor of about 2.7 and is enough to move a chain into
        a different mode of this likelihood. The two backends take the setting
        through different arguments, which is handled here.
    nuts_sampler : str, optional
        Passed to ``pm.sample``. "numpyro" compiles the whole graph to JAX and
        runs chains in parallel; "pymc" uses the default backend and the
        gradient Op, one chain at a time. Sequential is the safe choice there:
        where multiprocessing starts workers by forking, JAX in the parent can
        deadlock the children.
    random_seed : int, optional
        Seed for sampling.
    progressbar : bool, optional
        Show the sampling progress bar. Default is False.
    """

    #: How much wider than the data the default channel prior is.
    CHANNEL_PRIOR_WIDTH = 10.0

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        channel_prior_sd: float = None,
        nuts_sampler: str = "numpyro",
        jitter: bool = False,
        init_from: str = None,
        init: str = None,
        random_seed: int = None,
        progressbar: bool = False,
    ):
        super().__init__()
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.channel_prior_sd = channel_prior_sd
        self.nuts_sampler = nuts_sampler
        self.jitter = jitter
        self.init_from = init_from
        self.init = init
        self.random_seed = random_seed
        self.progressbar = progressbar
        self.idata = None
        self._sampled_model = None

    def supports_uncertainty(self) -> bool:
        """Report that this estimator provides uncertainty."""
        return True

    def build_model(self, model, pattern_data, groups=None):
        """Build the PyMC model for a set of HMP parameters.

        Parameters sharing an entry in the model's channel or time map are one
        random variable indexed into several positions, rather than several
        variables reconciled afterwards. The sampler therefore moves in the free
        parameter space and the sharing holds by construction, which projecting
        a proposal onto the constraint would not achieve.

        Returns
        -------
        pymc.Model
        """
        import pymc as pm  # noqa: PLC0415
        import pytensor.tensor as pt  # noqa: PLC0415

        n_dims = np.asarray(pattern_data.cross_corr).shape[1]
        shape = float(model.distribution.shape)

        channel_map = np.atleast_2d(np.asarray(model.channel_map)).astype(int)
        time_map = np.atleast_2d(np.asarray(model.time_map)).astype(int)
        n_groups = channel_map.shape[0]

        if n_groups > 1 and (channel_map.min() < 0 or time_map.min() < 0):
            raise NotImplementedError(
                "Groups that omit events are not supported yet; every entry of "
                "channel_map and time_map must be non-negative."
            )

        # EM holds these at their starting values; sampling every parameter would
        # answer a different question from the one the model was asked
        if len(model.fixed_channel_pars) or len(model.fixed_time_pars):
            raise NotImplementedError(
                "Sampling does not support fixed_channel_pars or fixed_time_pars; "
                "every parameter is estimated."
            )

        if groups is None:
            groups = np.zeros(len(pattern_data.durations), dtype=int)
        groups = np.asarray(groups)

        subsets = [groups == group for group in range(n_groups)]
        ops = [build_trial_op(pattern_data, model, subset=subset)
               for subset in subsets]

        channel_sd = self.channel_prior_sd
        if channel_sd is None:
            # several times the data spread; at the data scale it shrinks the
            # magnitudes
            channel_sd = float(np.std(np.asarray(pattern_data.cross_corr))) or 1.0
            channel_sd *= self.CHANNEL_PRIOR_WIDTH

        # centre the scale prior where an even split of a trial would put it
        mean_duration = float(np.mean(np.concatenate([op.durations for op in ops])))
        even_scale = float(
            model.distribution.mean_to_scale(mean_duration / time_map.shape[1])
        )

        channel_index, n_channel_shared, _ = self._tie_index(channel_map)
        time_index, n_time_shared, _ = self._tie_index(time_map)

        with pm.Model() as pymc_model:
            # one variable per distinct parameter, indexed into position
            shared_channel = pm.Normal(
                "channel_pars", mu=0.0, sigma=channel_sd,
                shape=(n_channel_shared, n_dims),
            )
            shared_log_scale = pm.Normal(
                "log_scale", mu=np.log(even_scale), sigma=1.0, shape=n_time_shared
            )
            shared_scale = pm.Deterministic("scale", pt.exp(shared_log_scale))

            for group in range(n_groups):
                channel_group = shared_channel[channel_index[group]]
                scale_group = shared_scale[time_index[group]]
                time_group = pt.stack(
                    [pt.fill(scale_group, shape), scale_group], axis=1
                )  # shape of the duration distribution is held, not sampled
                name = f"trials_{group}" if n_groups > 1 else "trials"
                distribution, n_trials_group = _trials_distribution(ops[group])
                distribution(name, channel_group, time_group,
                             observed=np.zeros(n_trials_group))

        self._even_scale = even_scale
        return pymc_model

    def fit(  # noqa: PLR0913, PLR0917
        self,
        model,
        pattern_data,
        initial_channel_pars: np.ndarray,
        initial_time_pars: np.ndarray,
        groups: np.ndarray = None,
        cpus: int = 1,
    ) -> EstimationResult:
        """Sample the posterior and summarise it as an EstimationResult.

        Where the model supplies several starting points, the best of them
        becomes the initial value of every chain, rather than one each: the
        chains are pooled into a single posterior, so starting them apart mixes
        draws from the worse basins into the result.

        Returns
        -------
        EstimationResult
            ``channel_pars`` and ``time_pars`` hold the posterior mean, shaped
            as the model expects. ``uncertainty`` holds posterior standard
            deviations, and ``diagnostics["idata"]`` the full InferenceData.

            ``diagnostics["separated_modes"]`` is worth reading whenever the fit
            does not converge. The posterior is multimodal when the number of
            events is misspecified, since the events can attach to different
            features of the data, and chains then settle in different modes.
            More draws do not help with that; a different number of events, or
            treating the modes separately, does.
        """
        import pymc as pm  # noqa: PLC0415

        em_result = None
        if self.init_from == "em":
            em_result = EMEstimator().fit(
                model, pattern_data, initial_channel_pars, initial_time_pars,
                groups=groups, cpus=cpus,
            )
            initial_channel_pars = np.asarray(em_result.channel_pars)[None, ...]
            initial_time_pars = np.asarray(em_result.time_pars)[None, ...]

        pymc_model = self.build_model(model, pattern_data, groups)

        initvals = self._initial_values(
            model, initial_channel_pars, initial_time_pars, pymc_model
        )

        cores = 1 if self.nuts_sampler != "numpyro" else None

        # the external samplers do not accept `init`, and the default backend
        # does not accept `jitter`, so the same intent is routed both ways
        if self.nuts_sampler == "pymc":
            init = self.init or (
                "jitter+adapt_diag" if self.jitter else "adapt_diag"
            )
            jitter_kwargs = {"init": init}
        else:
            jitter_kwargs = {"nuts_sampler_kwargs": {"jitter": self.jitter}}

        with pymc_model:
            idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=cores,
                target_accept=self.target_accept,
                nuts_sampler=self.nuts_sampler,
                initvals=initvals,
                **jitter_kwargs,
                random_seed=self.random_seed,
                progressbar=self.progressbar,
            )

        # not computed while sampling, and needed for leave-one-out
        try:
            with pymc_model:
                pm.compute_log_likelihood(idata, progressbar=False)
        except Exception:  # noqa: BLE001 - a usable fit should survive this failing
            pass

        self.idata = idata
        self._sampled_model = model
        self.fitted = True
        return self._summarise(idata, model, pymc_model, em_result)

    def posterior_event_probabilities(  # noqa: PLR0913, PLR0917
        self,
        model,
        pattern_data,
        n_draws: int = 100,
        groups: np.ndarray = None,
        random_seed: int = None,
    ):
        """Event probabilities implied by each of a sample of posterior draws.

        A fit gives one set of by-trial event probabilities, at one set of
        parameters. Sampling gives a distribution over them, which carries the
        parameter uncertainty into the quantity the method exists to estimate.

        Computed on request rather than during sampling, because the result is
        ``(draws, trials, samples, events)`` and only some of it is usually
        wanted.

        Parameters
        ----------
        model : EventModel
        pattern_data : PatternData
        n_draws : int, optional
            How many posterior draws to evaluate, sampled without replacement
            from all chains. Default is 100.
        groups : np.ndarray, optional
            Passed through to the model.
        random_seed : int, optional
            Seed for choosing which draws to use.

        Returns
        -------
        (n_draws, n_trials, n_samples, n_events) ndarray
        """
        if self.idata is None:
            raise ValueError("Nothing sampled yet, call fit first.")
        # the search strategies reuse one estimator for every submodel, so the
        # stored posterior belongs to whichever was fitted last
        if model is not self._sampled_model:
            raise ValueError(
                "The stored posterior belongs to a different model; call fit "
                "with this model before asking for its event probabilities."
            )

        posterior = self.idata.posterior
        channel_draws = posterior["channel_pars"].stack(pooled=("chain", "draw"))
        scale_draws = posterior["scale"].stack(pooled=("chain", "draw"))
        available = channel_draws.sizes["pooled"]

        rng = np.random.default_rng(random_seed)
        chosen = rng.choice(available, size=min(n_draws, available), replace=False)

        shape = float(model.distribution.shape)
        channel_index, _, _ = self._tie_index(
            np.atleast_2d(np.asarray(model.channel_map)).astype(int)
        )
        time_index, _, _ = self._tie_index(
            np.atleast_2d(np.asarray(model.time_map)).astype(int)
        )

        probabilities = []
        for index in chosen:
            shared_channel = np.asarray(
                channel_draws.isel(pooled=index).values, dtype=np.float64
            )
            shared_scale = np.asarray(
                scale_draws.isel(pooled=index).values, dtype=np.float64
            )
            # expand the shared values back out to one entry per group
            channel_pars = shared_channel[channel_index]
            scale = shared_scale[time_index]
            time_pars = np.stack([np.full(scale.shape, shape), scale], axis=-1)
            probabilities.append(
                model.event_probabilities(
                    pattern_data, channel_pars, time_pars, groups
                ).values
            )
        return np.stack(probabilities)

    @staticmethod
    def _tie_index(codes):
        """Give every distinct parameter in a map its own index.

        Codes are compared down each column, not across the whole map: groups
        carrying the same code at a given event share that event's parameter,
        while the same code at a different event is a different parameter. This
        matches how the maps are read when parameters are tied during EM.

        Returns
        -------
        index : (n_groups, n_positions) ndarray
            Where each group and position reads its value from.
        n_distinct : int
        first_seen : list of (group, position)
            Where each distinct parameter first appears.
        """
        codes = np.asarray(codes)
        index = np.zeros(codes.shape, dtype=int)
        first_seen = []
        for position in range(codes.shape[1]):
            column = codes[:, position]
            for code in np.unique(column):
                shared_by = column == code
                index[shared_by, position] = len(first_seen)
                first_seen.append((int(np.argmax(shared_by)), position))
        return index, len(first_seen), first_seen

    def _initial_values(self, model, initial_channel_pars, initial_time_pars,
                        pymc_model=None):
        """Give every chain the best of the model's starting points.

        The starting points are laid out per group, while the variables are per
        distinct code, so each shared value is read from the first position
        carrying that code.

        Several starting points mean "search from each and keep the best" to an
        estimator that returns one answer. A sampler pools its chains into a
        single posterior instead, so starting them in different basins mixes
        draws from the worse ones into the result rather than discarding them.
        The starting points are therefore scored and the best one is used
        throughout, which is the search EM performs, and leaves the chains
        agreeing or disagreeing about one basin.
        """
        channel = np.asarray(initial_channel_pars, dtype=np.float64)
        time = np.asarray(initial_time_pars, dtype=np.float64)
        n_available = min(len(channel), len(time))

        channel_map = np.atleast_2d(np.asarray(model.channel_map)).astype(int)
        time_map = np.atleast_2d(np.asarray(model.time_map)).astype(int)
        _, _, channel_at = self._tie_index(channel_map)
        _, _, time_at = self._tie_index(time_map)

        candidates = []
        for source in range(n_available):
            channel_start = np.stack(
                [channel[source][group][event] for group, event in channel_at]
            )
            scale_start = np.array(
                [time[source][group][stage][1] for group, stage in time_at]
            )
            candidates.append((channel_start, scale_start))

        best = 0
        if len(candidates) > 1:
            observed = self._observed_logp(pymc_model)
            scores = [self._evaluate_logp(observed, *candidate)
                      for candidate in candidates]
            if all(score is not None for score in scores):
                best = int(np.argmax(scores))

        channel_start, scale_start = candidates[best]
        # a copy per chain: the same array in every chain would let anything
        # that writes in place change the others' starting point
        return [
            {
                "channel_pars": channel_start.copy(),
                "log_scale": np.log(np.clip(scale_start, 1e-6, None)),
            }
            for _ in range(self.chains)
        ]

    @staticmethod
    def _log_density(idata, pymc_model):
        """Draw-wise log density, with the backend's sign convention removed.

        The numpyro sampler records potential energy in ``sample_stats["lp"]``,
        the default backend records log density. The sign is settled by
        evaluating the model's log density at one draw.
        """
        recorded = np.asarray(idata.sample_stats["lp"].values)
        if pymc_model is None:
            return recorded
        try:
            logp = pymc_model.compile_logp()
            point = {
                variable.name: np.asarray(idata.posterior[variable.name].values[0, 0])
                for variable in pymc_model.value_vars
            }
            reference = float(logp(point))
        except Exception:  # noqa: BLE001 - diagnostics must not break a good fit
            return recorded
        if abs(reference - recorded[0, 0]) > abs(reference + recorded[0, 0]):
            return -recorded
        return recorded

    @staticmethod
    def _observed_logp(pymc_model):
        """Compile the summed log-likelihood, without the priors."""
        if pymc_model is None:
            return None
        try:
            return pymc_model.compile_fn(
                pymc_model.logp(vars=pymc_model.observed_RVs, sum=True)
            )
        except Exception:  # noqa: BLE001 - a good fit must survive this failing
            return None

    @staticmethod
    def _evaluate_logp(observed, channel, scale):
        """Evaluate a compiled log-likelihood at one parameter vector.

        ``EstimationResult.likelihood`` has to mean the same thing whichever
        estimator produced it, because the search strategies compare it across
        models. EM gives the likelihood at the parameters it returns, so the
        sampler is asked for the same thing at its posterior mean rather than
        for a maximum over draws, which belongs to no returned parameter vector
        and grows with the number of draws.

        Returns None when the value cannot be computed, leaving the caller to
        fall back rather than report a different quantity under the same name.
        """
        if observed is None:
            return None
        try:
            return float(observed({
                "channel_pars": np.asarray(channel, dtype=float),
                "log_scale": np.log(
                    np.clip(np.asarray(scale, dtype=float), 1e-12, None)
                ),
            }))
        except Exception:  # noqa: BLE001 - a good fit must survive this failing
            return None

    def _summarise(self, idata, model, pymc_model=None, em_result=None):
        """Posterior mean as the point estimate, with diagnostics alongside."""
        import arviz as az  # noqa: PLC0415

        posterior = idata.posterior
        shared_channel = posterior["channel_pars"].mean(dim=("chain", "draw")).values
        shared_scale = posterior["scale"].mean(dim=("chain", "draw")).values
        shape = float(model.distribution.shape)

        # expand the shared values back out to one entry per group
        channel_map = np.atleast_2d(np.asarray(model.channel_map)).astype(int)
        time_map = np.atleast_2d(np.asarray(model.time_map)).astype(int)
        channel_index, _, _ = self._tie_index(channel_map)
        time_index, _, _ = self._tie_index(time_map)
        channel_mean = shared_channel[channel_index]
        scale_mean = shared_scale[time_index]
        time_mean = np.stack(
            [np.full(scale_mean.shape, shape), scale_mean], axis=-1
        )

        variables = ["channel_pars", "scale"]
        summary = az.summary(idata, var_names=variables)
        # not from az.summary, which rounds to two decimals
        rhat = az.rhat(idata, var_names=variables)
        ess = az.ess(idata, var_names=variables)
        max_rhat = float(max(float(rhat[name].max()) for name in variables))
        min_ess = float(min(float(ess[name].min()) for name in variables))
        divergences = int(idata.sample_stats["diverging"].sum())

        # chains far apart in log probability are in different modes, not
        # merely mixing slowly
        log_probability = self._log_density(idata, pymc_model)
        chain_levels = log_probability.mean(axis=1)
        within_chain = float(log_probability.std(axis=1).mean())
        level_spread = float(chain_levels.max() - chain_levels.min())
        separated_modes = bool(
            log_probability.shape[0] > 1 and level_spread > 5 * max(within_chain, 1e-12)
        )

        shared_channel_sd = posterior["channel_pars"].std(dim=("chain", "draw")).values
        shared_scale_sd = posterior["scale"].std(dim=("chain", "draw")).values

        max_log_density = float(np.max(log_probability))
        log_likelihood = self._evaluate_logp(
            self._observed_logp(pymc_model), shared_channel, shared_scale
        )
        if log_likelihood is None:
            log_likelihood = max_log_density

        return EstimationResult(
            channel_pars=channel_mean,
            time_pars=time_mean,
            likelihood=log_likelihood,
            # a single chain has no r_hat, so convergence rests on the rest
            converged=bool(
                (np.isnan(max_rhat) or max_rhat < 1.01) and divergences == 0
            ),
            n_iterations=int(self.draws * self.chains),
            diagnostics={
                "idata": idata,
                "max_rhat": max_rhat,
                "min_ess": min_ess,
                "divergences": divergences,
                "summary": summary,
                "separated_modes": separated_modes,
                "chain_lp_spread": level_spread,
                "chain_lp": chain_levels,
                "max_log_density": max_log_density,
                # what the sampler was started from, so the two can be compared
                "em_likelihood": None if em_result is None else em_result.likelihood,
            },
            uncertainty={
                "channel_pars_sd": shared_channel_sd[channel_index],
                "scale_sd": shared_scale_sd[time_index],
            },
        )
