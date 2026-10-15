"""Tests for the differentiable likelihood and the sampling estimator.

The likelihood is checked against reference values captured from the numpy
implementation in float64, stage by stage rather than only on the final number,
so a disagreement says where it is. Regenerate the reference with
``python tests/gen_data/generate_reference.py``.
"""

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import hmp
from hmp.models import EventModel
from hmp.patterndata import PatternData
from test_io import init_data

REFERENCE = Path(__file__).parent / "gen_data" / "likelihood_reference.npz"
SHIFT = 1  # hmp.distributions.Gamma.shift

# Reference values are float64 throughout, so agreement is demanded at a
# tolerance set by double precision rather than by whatever happens to pass.
EXACT = 1e-10


@pytest.fixture(scope="module")
def reference():
    if not REFERENCE.exists():
        pytest.skip(f"{REFERENCE.name} missing; run tests/gen_data/generate_reference.py")
    return np.load(REFERENCE, allow_pickle=False)


@pytest.fixture(scope="module")
def layout(reference):
    """Return the data and trial layout shared by every reference case."""
    durations = reference["durations"]
    stacked = np.vstack(
        [reference["cross_corr"][s:e + 1]
         for s, e in zip(reference["starts"], reference["ends"])]
    )
    return {
        "cross_corr": stacked,
        "trial_starts": np.concatenate([[0], np.cumsum(durations)[:-1]]),
        "durations": durations,
        "max_duration": int(durations.max()),
    }


def case_names():
    """List the reference cases to parametrise over."""
    if not REFERENCE.exists():
        return []
    return list(np.load(REFERENCE, allow_pickle=False)["cases"])


def scaled_error(actual, expected):
    """Return the error relative to the scale of the array.

    These arrays span hundreds of orders of magnitude, so an elementwise
    relative error on entries that are legitimately near zero means nothing.
    """
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    scale = max(float(np.max(np.abs(expected))), 1e-300)
    return float(np.max(np.abs(actual - expected)) / scale)


class TestLikelihoodMatchesReference:
    """The JAX likelihood must reproduce the numpy one it was ported from."""

    @pytest.mark.parametrize("case", case_names())
    def test_stages_match(self, case, reference, layout):
        from hmp.estimators import jax_likelihood as jl

        channel_pars = reference[f"{case}__channel_pars"]
        time_pars = reference[f"{case}__time_pars"]
        locations = reference[f"{case}__locations_samples"]

        computed_gains = jl.gains(layout["cross_corr"], channel_pars)
        assert scaled_error(computed_gains, reference[f"{case}__gains"]) < EXACT

        pmf = jl.stage_pmf(time_pars, locations, layout["max_duration"], SHIFT)
        assert scaled_error(pmf, reference[f"{case}__pmf"]) < EXACT

        per_trial = jl.trial_log_likelihood(
            layout["cross_corr"], channel_pars, time_pars, layout["trial_starts"],
            layout["durations"], locations, layout["max_duration"], SHIFT,
        )
        assert scaled_error(per_trial, reference[f"{case}__trial_likelihood"]) < EXACT

    @pytest.mark.parametrize("case", case_names())
    def test_total_matches(self, case, reference, layout):
        from hmp.estimators import jax_likelihood as jl

        total = jl.log_likelihood(
            layout["cross_corr"], reference[f"{case}__channel_pars"],
            reference[f"{case}__time_pars"], layout["trial_starts"],
            layout["durations"], reference[f"{case}__locations_samples"],
            layout["max_duration"], SHIFT,
        )
        assert scaled_error(total, reference[f"{case}__likelihood"]) < EXACT


class TestGradients:
    """A wrong gradient can hide behind correct values, so check it separately.

    The density goes through log(0) at the origin, where jnp.where masks the
    value but not the gradient, so this is a live failure mode rather than a
    formality.
    """

    @pytest.mark.parametrize("case", case_names())
    def test_matches_finite_differences(self, case, reference, layout):
        from jax.test_util import check_grads

        from hmp.estimators import jax_likelihood as jl

        channel_pars = jnp.asarray(reference[f"{case}__channel_pars"])
        time_pars = jnp.asarray(reference[f"{case}__time_pars"])
        locations = jnp.asarray(reference[f"{case}__locations_samples"])

        def total(channel, time):
            return jl.log_likelihood(
                layout["cross_corr"], channel, time, layout["trial_starts"],
                layout["durations"], locations, layout["max_duration"], SHIFT,
            )

        grad_channel, grad_time = jax.grad(total, argnums=(0, 1))(channel_pars, time_pars)
        assert np.isfinite(np.asarray(grad_channel)).all()
        assert np.isfinite(np.asarray(grad_time)).all()

        check_grads(total, (channel_pars, time_pars), order=1, modes=["rev"],
                    atol=2e-4, rtol=2e-4)


@pytest.fixture(scope="module")
def fitted_setup():
    """Return float64 data and an EM fit, for the bridge and sampler tests."""
    _, _, epoch_data, _, _, n_events = init_data()
    hmp_data = hmp.basedata.default(
        epoch_data, n_comp=3, center=True, duration_id="response_time"
    )
    pattern_data = PatternData.from_basedata(hmp_data, dtype=np.float64)
    fitted = EventModel(n_events=n_events)
    fitted.fit(pattern_data, verbose=False)
    return pattern_data, fitted, n_events


class TestPyTensorBridge:
    """The Op has to agree with the function it wraps, on both backends."""

    def test_backends_agree_with_numpy(self, fitted_setup):
        pytest.importorskip("pytensor")
        import pytensor
        import pytensor.tensor as pt

        from hmp.estimators.pytensor_op import build_op

        pattern_data, fitted, n_events = fitted_setup
        model = EventModel(n_events=n_events)
        model.n_dims = pattern_data.cross_corr.shape[1]

        # float64 parameters: EM stores channel_pars as float32, which caps
        # agreement at about 1e-7 for reasons unrelated to this implementation
        channel_pars = np.asarray(fitted.channel_pars, dtype=np.float64)
        time_pars = np.asarray(fitted.time_pars, dtype=np.float64)
        expected = model.log_likelihood(pattern_data, channel_pars, time_pars)

        op = build_op(pattern_data, model)
        channel_input, time_input = pt.dmatrix("channel"), pt.dmatrix("time")
        out = op(channel_input, time_input)

        default = float(pytensor.function([channel_input, time_input], out)(
            channel_pars[0], time_pars[0]))
        as_jax = float(pytensor.function([channel_input, time_input], out, mode="JAX")(
            channel_pars[0], time_pars[0]))

        assert abs(default - expected) / abs(expected) < EXACT
        assert default == as_jax

    def test_gradient_available_on_both_backends(self, fitted_setup):
        pytest.importorskip("pytensor")
        import pytensor
        import pytensor.tensor as pt

        from hmp.estimators.pytensor_op import build_op

        pattern_data, fitted, n_events = fitted_setup
        model = EventModel(n_events=n_events)
        model.n_dims = pattern_data.cross_corr.shape[1]
        op = build_op(pattern_data, model)

        channel_input, time_input = pt.dmatrix("channel"), pt.dmatrix("time")
        gradients = pt.grad(op(channel_input, time_input), [channel_input, time_input])
        channel_pars = np.asarray(fitted.channel_pars[0], dtype=np.float64)
        time_pars = np.asarray(fitted.time_pars[0], dtype=np.float64)

        results = []
        for mode in (None, "JAX"):
            grad_channel, grad_time = pytensor.function(
                [channel_input, time_input], gradients, mode=mode
            )(channel_pars, time_pars)
            assert np.isfinite(grad_channel).all()
            assert np.isfinite(grad_time).all()
            results.append((grad_channel, grad_time))

        assert np.allclose(results[0][0], results[1][0], atol=0, rtol=EXACT)
        assert np.allclose(results[0][1], results[1][1], atol=0, rtol=EXACT)


class TestMCMCEstimator:
    """The PyMC model has to be wired to the likelihood we verified."""

    def test_model_logp_is_likelihood_plus_priors(self, fitted_setup):
        pytest.importorskip("pymc")
        from scipy import stats

        from hmp.estimators.mcmc import MCMCEstimator
        from hmp.estimators.pytensor_op import build_op

        pattern_data, fitted, n_events = fitted_setup
        model = EventModel(n_events=n_events)
        model.n_dims = pattern_data.cross_corr.shape[1]

        estimator = MCMCEstimator()
        pymc_model = estimator.build_model(model, pattern_data)

        channel_pars = np.asarray(fitted.channel_pars[0], dtype=np.float64)
        scale = np.asarray(fitted.time_pars[0], dtype=np.float64)[:, 1]
        total = float(pymc_model.compile_logp()(
            {"channel_pars": channel_pars, "log_scale": np.log(scale)}
        ))

        op = build_op(pattern_data, model)
        shape = float(model.distribution.shape)
        time_pars = np.column_stack([np.full(scale.shape, shape), scale])
        likelihood = float(op._call_jax(channel_pars, time_pars))

        channel_sd = float(np.std(np.asarray(pattern_data.cross_corr))) * \
            MCMCEstimator.CHANNEL_PRIOR_WIDTH
        priors = (
            stats.norm.logpdf(channel_pars, 0.0, channel_sd).sum()
            + stats.norm.logpdf(np.log(scale), np.log(estimator._even_scale), 1.0).sum()
        )
        assert abs(total - (likelihood + priors)) / abs(likelihood + priors) < EXACT

    def test_log_density_removes_the_backend_sign_convention(self, fitted_setup):
        """A negated ``lp`` has to come back positive, and a correct one unchanged.

        numpyro records potential energy where the default backend records log
        density. Every diagnostic computed from it is invariant under negation,
        so nothing else in the suite can tell whether the correction fired.
        """
        pytest.importorskip("pymc")
        import arviz as az

        from hmp.estimators.mcmc import MCMCEstimator

        pattern_data, fitted, n_events = fitted_setup
        model = EventModel(n_events=n_events)
        model.n_dims = pattern_data.cross_corr.shape[1]

        estimator = MCMCEstimator()
        pymc_model = estimator.build_model(model, pattern_data)

        channel_pars = np.asarray(fitted.channel_pars[0], dtype=np.float64)
        log_scale = np.log(np.asarray(fitted.time_pars[0], dtype=np.float64)[:, 1])
        truth = float(pymc_model.compile_logp()(
            {"channel_pars": channel_pars, "log_scale": log_scale}
        ))

        def density(recorded):
            idata = az.from_dict(
                posterior={
                    "channel_pars": channel_pars[None, None, ...],
                    "log_scale": log_scale[None, None, ...],
                },
                sample_stats={"lp": np.array([[recorded]])},
            )
            return MCMCEstimator._log_density(idata, pymc_model)[0, 0]

        assert abs(density(truth) - truth) < abs(truth) * EXACT
        assert abs(density(-truth) - truth) < abs(truth) * EXACT

    def test_likelihood_is_taken_at_the_returned_parameters(self, fitted_setup):
        """The field has to hold what EM puts there, not a maximum over draws.

        The search strategies compare ``EstimationResult.likelihood`` across
        models, so it has to be the likelihood at the parameters that come back
        with it. A maximum over draws belongs to no returned parameter vector
        and grows as draws are added, which would make more sampling look like
        a better model.
        """
        pytest.importorskip("pymc")
        pytest.importorskip("numpyro")

        from hmp.estimators.mcmc import MCMCEstimator
        from hmp.estimators.pytensor_op import build_op

        pattern_data, _, n_events = fitted_setup
        model = EventModel(n_events=n_events)
        estimator = MCMCEstimator(draws=100, tune=100, chains=2, jitter=False,
                                  random_seed=0, progressbar=False)
        model.fit(pattern_data, estimator=estimator, verbose=False)
        result = model.estimation_result

        op = build_op(pattern_data, model)
        channel = np.asarray(result.channel_pars[0], dtype=np.float64)
        time_pars = np.asarray(result.time_pars[0], dtype=np.float64)
        at_estimate = float(op._call_jax(channel, time_pars))

        assert abs(result.likelihood - at_estimate) / abs(at_estimate) < 1e-6

        idata = result.diagnostics["idata"]
        per_draw = sum(np.asarray(idata.log_likelihood[name].values).sum(axis=-1)
                       for name in idata.log_likelihood.data_vars)
        assert result.likelihood != float(np.max(per_draw))

    def test_fixed_parameters_are_refused(self, fitted_setup):
        """EM holds these at their starting values; sampling cannot, so it says so."""
        pytest.importorskip("pymc")

        from hmp.estimators.mcmc import MCMCEstimator

        pattern_data, _, n_events = fitted_setup
        model = EventModel(n_events=n_events, fixed_time_pars=[0])
        model.n_dims = pattern_data.cross_corr.shape[1]

        with pytest.raises(NotImplementedError, match="fixed_"):
            MCMCEstimator().build_model(model, pattern_data)

    def test_probabilities_refuse_another_model(self, fitted_setup):
        """One estimator serves many submodels, so the stored posterior is checked.

        The search strategies reuse a single estimator for every submodel, which
        leaves the last fit's posterior in place. Pairing it with a different
        model would silently describe the wrong one.
        """
        pytest.importorskip("pymc")
        pytest.importorskip("numpyro")

        from hmp.estimators.mcmc import MCMCEstimator

        pattern_data, _, n_events = fitted_setup
        estimator = MCMCEstimator(draws=50, tune=50, chains=2, jitter=False,
                                  random_seed=0, progressbar=False)
        fitted = EventModel(n_events=n_events)
        fitted.fit(pattern_data, estimator=estimator, verbose=False)

        other = EventModel(n_events=n_events)
        with pytest.raises(ValueError, match="different model"):
            estimator.posterior_event_probabilities(other, pattern_data, n_draws=2)

    def test_em_seeding_starts_where_em_finished(self, fitted_setup):
        """``init_from="em"`` has to run EM and sample from its solution.

        A maximizer gets into a sensible region of this likelihood more
        reliably than a sampler started anywhere else, so the two are used for
        what each is good at. The EM likelihood is recorded alongside so the
        posterior can be compared against the point it was started from.
        """
        pytest.importorskip("pymc")
        pytest.importorskip("numpyro")

        from hmp.estimators.mcmc import MCMCEstimator

        pattern_data, _, n_events = fitted_setup
        model = EventModel(n_events=n_events)
        estimator = MCMCEstimator(draws=50, tune=50, chains=2, jitter=False,
                                  init_from="em", random_seed=0, progressbar=False)
        model.fit(pattern_data, estimator=estimator, verbose=False)
        result = model.estimation_result

        seeded_from = result.diagnostics["em_likelihood"]
        assert seeded_from is not None, "the EM solution it started from is not recorded"
        assert np.isfinite(seeded_from)

        plain = EventModel(n_events=n_events)
        plain.fit(pattern_data, estimator=MCMCEstimator(
            draws=50, tune=50, chains=2, jitter=False, random_seed=0,
            progressbar=False), verbose=False)
        assert plain.estimation_result.diagnostics["em_likelihood"] is None

    def test_selection_prefers_the_smaller_model_when_the_gap_is_noise(self):
        """A larger model has to beat the smaller by more than the error on the gap.

        Two models are compared where one is a copy of the other, so the
        difference between them is zero and entirely within its own error. The
        smaller has to win, which is what stops the rule from behaving like the
        likelihood it replaces.
        """
        pytest.importorskip("arviz")
        import arviz as az

        from hmp.models.base import select_by_loo

        rng = np.random.default_rng(0)
        draws = rng.normal(size=(2, 60, 25))

        class _Fitted:
            def __init__(self, idata):
                self.estimation_result = type(
                    "R", (), {"diagnostics": {"idata": idata}}
                )()

        idata = az.from_dict(
            posterior={"x": draws[..., :1]},
            log_likelihood={"obs": draws},
        )
        same = az.from_dict(
            posterior={"x": draws[..., :1]},
            log_likelihood={"obs": draws.copy()},
        )
        chosen, comparison = select_by_loo({3: _Fitted(idata), 4: _Fitted(same)})

        assert chosen == 3, f"took the larger model on a zero difference\n{comparison}"

    def test_selection_needs_pointwise_likelihoods(self):
        """EM reports one number per fit, so the rule has to refuse rather than guess."""
        pytest.importorskip("arviz")

        from hmp.models.base import select_by_loo

        class _EMFitted:
            estimation_result = type("R", (), {"diagnostics": {}})()

        with pytest.raises(ValueError, match="per-trial log-likelihoods"):
            select_by_loo({3: _EMFitted(), 4: _EMFitted()})

    def test_samples_and_reports_diagnostics(self, fitted_setup):
        pytest.importorskip("pymc")
        pytest.importorskip("numpyro")

        from hmp.estimators.base import EstimationResult
        from hmp.estimators.mcmc import MCMCEstimator

        pattern_data, _, n_events = fitted_setup
        model = EventModel(n_events=n_events)
        model.n_dims = pattern_data.cross_corr.shape[1]
        _, groups, _ = model.group_constructor(pattern_data.durations, verbose=False)
        channel_pars, time_pars = model._format_parameters(
            None, None, groups, 1, pattern_data.durations, pattern_data.sfreq
        )

        estimator = MCMCEstimator(draws=100, tune=100, chains=2, random_seed=0)
        result = estimator.fit(model, pattern_data, channel_pars, time_pars, groups=groups)

        assert isinstance(result, EstimationResult)
        assert estimator.supports_uncertainty()
        assert result.channel_pars.shape == (1, n_events, model.n_dims)
        assert result.time_pars.shape == (1, n_events + 1, 2)
        # the shape of the duration distribution is held, not sampled
        assert np.allclose(result.time_pars[0][:, 0], model.distribution.shape)
        for key in ("idata", "max_rhat", "min_ess", "divergences",
                    "separated_modes", "chain_lp_spread"):
            assert key in result.diagnostics
        assert set(result.uncertainty) == {"channel_pars_sd", "scale_sd"}

        # the reported r_hat has to be the diagnostic itself, not the value from
        # az.summary, which rounds to two decimals and so cannot be compared
        # against a 1.01 threshold
        import arviz as az

        variables = ["channel_pars", "scale"]
        rhat = az.rhat(result.diagnostics["idata"], var_names=variables)
        recomputed = max(float(rhat[name].max()) for name in variables)
        assert result.diagnostics["max_rhat"] == pytest.approx(recomputed, abs=1e-12)

    def test_default_backend_samples(self, fitted_setup):
        """The default backend must sample rather than hang.

        It runs chains in forked processes, and forking after JAX has been used
        deadlocks instead of raising, so this path once hung indefinitely with
        no error. Chains are run sequentially there to avoid it.
        """
        pytest.importorskip("pymc")

        from hmp.estimators.mcmc import MCMCEstimator

        pattern_data, _, n_events = fitted_setup
        model = EventModel(n_events=n_events)
        model.n_dims = pattern_data.cross_corr.shape[1]
        _, groups, _ = model.group_constructor(pattern_data.durations, verbose=False)
        channel_pars, time_pars = model._format_parameters(
            None, None, groups, 1, pattern_data.durations, pattern_data.sfreq
        )

        estimator = MCMCEstimator(draws=25, tune=25, chains=2, random_seed=0,
                                  nuts_sampler="pymc")
        result = estimator.fit(
            model, pattern_data, channel_pars, time_pars, groups=groups
        )

        assert result.diagnostics["idata"].posterior.sizes["chain"] == 2
        assert np.isfinite(result.channel_pars).all()
        assert np.isfinite(result.time_pars).all()

    def test_posterior_event_probabilities(self, fitted_setup):
        """Each draw implies its own by-trial event probabilities."""
        pytest.importorskip("pymc")
        pytest.importorskip("numpyro")

        from hmp.estimators.mcmc import MCMCEstimator

        pattern_data, _, n_events = fitted_setup
        model = EventModel(n_events=n_events)
        model.n_dims = pattern_data.cross_corr.shape[1]
        _, groups, _ = model.group_constructor(pattern_data.durations, verbose=False)
        channel_pars, time_pars = model._format_parameters(
            None, None, groups, 1, pattern_data.durations, pattern_data.sfreq
        )

        estimator = MCMCEstimator(draws=100, tune=100, chains=2, random_seed=0)

        with pytest.raises(ValueError, match="fit"):
            estimator.posterior_event_probabilities(model, pattern_data)

        estimator.fit(model, pattern_data, channel_pars, time_pars, groups=groups)
        probabilities = estimator.posterior_event_probabilities(
            model, pattern_data, n_draws=10, groups=groups, random_seed=0
        )

        n_trials = len(pattern_data.durations)
        assert probabilities.shape[0] == 10
        assert probabilities.shape[1] == n_trials
        assert probabilities.shape[3] == n_events
        assert np.isfinite(probabilities).all()
        assert (probabilities >= 0).all()
        # each draw, trial and event is a distribution over samples
        assert np.allclose(probabilities.sum(axis=2), 1.0, atol=1e-10)

class TestParameterTying:
    """Codes in the maps are compared down each column, not across the map.

    Groups carrying the same code at a given event share that event's parameter,
    while the same code at a different event is a different parameter. Reading
    them as global identifiers would tie every event together whenever the map
    is all zeros, which is the default.
    """

    def test_default_map_gives_one_parameter_per_event(self):
        from hmp.estimators.mcmc import MCMCEstimator

        model = EventModel(n_events=3)
        index, n_distinct, _ = MCMCEstimator._tie_index(
            np.atleast_2d(np.asarray(model.channel_map)).astype(int)
        )
        assert n_distinct == 3
        assert index.tolist() == [[0, 1, 2]]

    def test_shared_prefix_across_groups(self):
        from hmp.estimators.mcmc import MCMCEstimator

        # first two events shared by all three groups, the rest per group
        channel_map = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
        index, n_distinct, first_seen = MCMCEstimator._tie_index(channel_map)

        assert index[:, 0].tolist() == [0, 0, 0]
        assert index[:, 1].tolist() == [1, 1, 1]
        assert len(set(index[:, 2].tolist())) == 3
        assert n_distinct == 5
        assert len(first_seen) == n_distinct

    def test_grouped_likelihood_matches_numpy(self, fitted_setup):
        """The summed per-group Ops must equal the numpy grouped likelihood.

        Each group is scored on its own trials, so each carries its own
        max_duration, and the pmf is normalised over that. Getting that wrong
        would still sample, just from the wrong posterior.
        """
        pytest.importorskip("pymc")
        from scipy import stats

        from hmp.estimators.mcmc import MCMCEstimator

        pattern_data, _, n_events = fitted_setup
        n_dims = pattern_data.cross_corr.shape[1]
        channel_map = np.array([[0, 0, 0], [0, 1, 1]])
        time_map = np.array([[0, 0, 0, 0], [0, 1, 1, 1]])
        model = EventModel(
            n_events=n_events, channel_map=channel_map, time_map=time_map,
            grouping_dict={"condition": ["a", "b"]},
        )
        model.n_dims = n_dims
        groups = np.arange(len(pattern_data.durations)) % 2

        estimator = MCMCEstimator()
        pymc_model = estimator.build_model(model, pattern_data, groups)
        channel_index, n_channel, _ = estimator._tie_index(channel_map)
        time_index, n_time, _ = estimator._tie_index(time_map)
        shape = float(model.distribution.shape)

        rng = np.random.default_rng(0)
        for _ in range(3):
            shared_channel = rng.normal(0, 1.0, (n_channel, n_dims))
            shared_scale = np.exp(rng.normal(np.log(5.0), 0.3, n_time))

            total = float(pymc_model.compile_logp()(
                {"channel_pars": shared_channel, "log_scale": np.log(shared_scale)}
            ))
            channel_sd = float(np.std(np.asarray(pattern_data.cross_corr))) * \
            MCMCEstimator.CHANNEL_PRIOR_WIDTH
            priors = (
                stats.norm.logpdf(shared_channel, 0.0, channel_sd).sum()
                + stats.norm.logpdf(
                    np.log(shared_scale), np.log(estimator._even_scale), 1.0).sum()
            )

            scale = shared_scale[time_index]
            expected = model.log_likelihood(
                pattern_data,
                shared_channel[channel_index],
                np.stack([np.full(scale.shape, shape), scale], axis=-1),
                groups,
            )
            assert abs((total - priors) - expected) / abs(expected) < EXACT

    def test_grouped_model_samples(self, fitted_setup):
        pytest.importorskip("pymc")
        pytest.importorskip("numpyro")

        from hmp.estimators.mcmc import MCMCEstimator

        pattern_data, _, n_events = fitted_setup
        n_dims = pattern_data.cross_corr.shape[1]
        # two groups sharing the first event, differing on the rest
        channel_map = np.array([[0, 0, 0], [0, 1, 1]])
        time_map = np.array([[0, 0, 0, 0], [0, 1, 1, 1]])
        model = EventModel(
            n_events=n_events, channel_map=channel_map, time_map=time_map,
            grouping_dict={"condition": ["a", "b"]},
        )
        model.n_dims = n_dims

        groups = np.arange(len(pattern_data.durations)) % 2
        channel_pars, time_pars = model._format_parameters(
            None, None, groups, 2, pattern_data.durations, pattern_data.sfreq
        )

        estimator = MCMCEstimator(draws=100, tune=100, chains=2, random_seed=0)
        result = estimator.fit(
            model, pattern_data, channel_pars, time_pars, groups=groups
        )

        assert result.channel_pars.shape == (2, n_events, n_dims)
        assert result.time_pars.shape == (2, n_events + 1, 2)
        # the shared first event must come out identical in both groups
        assert np.array_equal(result.channel_pars[0, 0], result.channel_pars[1, 0])
        assert result.time_pars[0][0, 1] == result.time_pars[1][0, 1]
        # the unshared ones must not
        assert not np.array_equal(result.channel_pars[0, 1], result.channel_pars[1, 1])

    def test_groups_omitting_events_are_refused(self, fitted_setup):
        pytest.importorskip("pymc")
        from hmp.estimators.mcmc import MCMCEstimator

        pattern_data, _, n_events = fitted_setup
        model = EventModel(
            n_events=n_events,
            channel_map=np.array([[0, 0, 0], [0, 0, -1]]),
            time_map=np.array([[0, 0, 0, 0], [0, 0, 0, 0]]),
            grouping_dict={"condition": ["a", "b"]},
        )
        model.n_dims = pattern_data.cross_corr.shape[1]

        with pytest.raises(NotImplementedError, match="omit events"):
            MCMCEstimator().build_model(model, pattern_data)


class TestPosteriorAgreesWithEM:
    """On data with enough trials the posterior should sit on the EM solution.

    This is the check that the sampler is estimating the same thing EM does.
    Absolute recovery of the simulated parameters is a question about HMP rather
    than about this estimator: EM shows the same small upward offset on the
    scales, so agreeing with EM is the claim that belongs here.
    """

    def test_posterior_concentrates_on_the_em_solution(self):
        pytest.importorskip("pymc")
        pytest.importorskip("numpyro")
        from test_io import init_data_large

        from hmp.estimators.mcmc import MCMCEstimator

        _, epoch_data, _, _, n_events = init_data_large()
        hmp_data = hmp.basedata.default(
            epoch_data, n_comp=3, center=True, duration_id="response_time"
        )
        pattern_data = PatternData.from_basedata(hmp_data, dtype=np.float64)

        em = EventModel(n_events=n_events)
        em.fit(pattern_data, verbose=False)

        model = EventModel(n_events=n_events)
        model.n_dims = pattern_data.cross_corr.shape[1]
        _, groups, _ = model.group_constructor(pattern_data.durations, verbose=False)
        channel_pars, time_pars = model._format_parameters(
            None, None, groups, 1, pattern_data.durations, pattern_data.sfreq
        )

        estimator = MCMCEstimator(draws=500, tune=500, chains=2, random_seed=0)
        result = estimator.fit(
            model, pattern_data, channel_pars, time_pars, groups=groups
        )

        # 1.05 rather than the 1.01 wanted for reported results: this runs few
        # draws to stay quick, which makes r_hat itself noisier
        assert result.diagnostics["divergences"] == 0
        assert result.diagnostics["max_rhat"] < 1.05

        em_scales = np.asarray(em.time_pars[0], dtype=np.float64)[:, 1]
        posterior_scales = result.time_pars[0][:, 1]
        posterior_sd = result.uncertainty["scale_sd"]

        # every scale within one posterior standard deviation of EM's
        assert np.all(np.abs(posterior_scales - em_scales) < posterior_sd)


class TestMultimodality:
    """A misspecified number of events makes the posterior multimodal.

    The events can attach to different features of the data, so chains settle at
    different log probabilities. r_hat reports the disagreement but not the
    reason, and more draws do not fix this one, so the estimator reports whether
    the chains are separated rather than merely mixing slowly.

    Jitter is asked for here rather than left at its default. Chains that start
    at the same point tend to reach the same mode, so a fit can look settled
    while the posterior has several; spreading the starting points is what gives
    the chains a chance to disagree, which is the whole of what this diagnostic
    reads.
    """

    def test_separated_modes_flagged_when_events_misspecified(self):
        pytest.importorskip("pymc")
        pytest.importorskip("numpyro")
        from test_io import init_data_large

        from hmp.estimators.mcmc import MCMCEstimator

        _, epoch_data, _, _, n_events = init_data_large()
        hmp_data = hmp.basedata.default(
            epoch_data, n_comp=3, center=True, duration_id="response_time"
        )
        pattern_data = PatternData.from_basedata(hmp_data, dtype=np.float64)

        outcomes = {}
        for candidate in (1, n_events):
            model = EventModel(n_events=candidate)
            model.n_dims = pattern_data.cross_corr.shape[1]
            _, groups, _ = model.group_constructor(
                pattern_data.durations, verbose=False
            )
            channel_pars, time_pars = model._format_parameters(
                None, None, groups, 1, pattern_data.durations, pattern_data.sfreq
            )
            estimator = MCMCEstimator(draws=300, tune=300, chains=2, jitter=True,
                                      random_seed=0)
            outcomes[candidate] = estimator.fit(
                model, pattern_data, channel_pars, time_pars, groups=groups
            )

        correct = outcomes[n_events]
        misspecified = outcomes[1]

        # the right number of events gives one mode and a usable fit
        assert not correct.diagnostics["separated_modes"]
        assert correct.diagnostics["max_rhat"] < 1.05

        # the wrong number does not, and says so rather than only failing r_hat
        assert misspecified.diagnostics["separated_modes"]
        assert (misspecified.diagnostics["chain_lp_spread"]
                > correct.diagnostics["chain_lp_spread"])


class TestParameterPrecision:
    """Why comparisons against the numpy likelihood have to use float64 parameters."""

    def test_float32_parameters_cap_agreement(self, fitted_setup):
        """EM stores channel_pars as float32, which limits agreement to about 1e-7.

        estim_probs evaluates ``channel_pars ** 2 / 2`` in the dtype of the
        array it is given, so float32 parameters make the whole likelihood
        float32-accurate even when the data is float64. This is a property of
        how the parameters are stored, not of either implementation.
        """
        from hmp.estimators.pytensor_op import build_op

        pattern_data, fitted, n_events = fitted_setup
        model = EventModel(n_events=n_events)
        model.n_dims = pattern_data.cross_corr.shape[1]
        op = build_op(pattern_data, model)

        assert fitted.channel_pars.dtype == np.float32

        as_float64 = np.asarray(fitted.channel_pars, dtype=np.float64)
        time_pars = np.asarray(fitted.time_pars, dtype=np.float64)
        reference = float(op._call_jax(as_float64[0], time_pars[0]))

        from_stored = model.log_likelihood(
            pattern_data, fitted.channel_pars, fitted.time_pars
        )
        from_float64 = model.log_likelihood(pattern_data, as_float64, time_pars)

        assert abs(reference - from_float64) / abs(from_float64) < EXACT
        assert abs(reference - from_stored) / abs(from_stored) > EXACT


class TestEMFixedPoint:
    """EM does not stop at a stationary point of the likelihood.

    At its solution the channel contributions have converged but the scales have
    not, and stepping along the gradient still improves the likelihood. That
    matters here because it means the posterior mode and the EM solution should
    not be expected to coincide, so agreement between them is not a valid check
    on a sampler.

    Why EM stops there is not established. There is a separate inconsistency
    recorded below, between the mean ``mean_to_scale`` assumes and the mean of
    the pmf the likelihood uses, but the size of that gap does not track the
    residual gradient across stages, so it is not the explanation.
    """

    def test_scale_update_assumes_a_mean_the_pmf_does_not_have(self, fitted_setup):
        pattern_data, _, n_events = fitted_setup
        model = EventModel(n_events=n_events)
        durations = np.asarray(pattern_data.ends) - np.asarray(pattern_data.starts) + 1
        max_duration = int(durations.max())
        support = np.arange(max_duration)

        # a scale putting appreciable mass beyond the support
        scale = max_duration / 4
        pmf = model.distribution_pdf(model.distribution.shape, scale, max_duration)

        assumed = float(model.distribution.scale_to_mean(scale))
        actual = float(np.sum(support * pmf))
        assert abs(actual - assumed) / assumed > 0.05

    def test_likelihood_still_improves_at_the_em_solution(self, fitted_setup):
        from hmp.estimators.pytensor_op import build_op

        pattern_data, fitted, n_events = fitted_setup
        model = EventModel(n_events=n_events)
        model.n_dims = pattern_data.cross_corr.shape[1]
        op = build_op(pattern_data, model)

        channel_pars = np.asarray(fitted.channel_pars[0], dtype=np.float64)
        time_pars = np.asarray(fitted.time_pars[0], dtype=np.float64)
        grad_channel, grad_time = jax.grad(op._call_jax, argnums=(0, 1))(
            channel_pars, time_pars
        )
        grad_channel = np.asarray(grad_channel)
        grad_time = np.asarray(grad_time)

        # the channel contributions do reach a stationary point
        assert np.linalg.norm(grad_channel) < 1e-2
        # the scales do not, by orders of magnitude
        assert np.linalg.norm(grad_time[:, 1]) > 10 * np.linalg.norm(grad_channel)

        before = float(op._call_jax(channel_pars, time_pars))
        stepped = time_pars.copy()
        stepped[:, 1] = time_pars[:, 1] + 1e-2 * grad_time[:, 1]
        after = float(op._call_jax(channel_pars + 1e-2 * grad_channel, stepped))
        assert after > before


class TestOpIdentity:
    """PyTensor merges Ops it considers equal, so equality has to track the data.

    ``__props__`` drives ``__eq__``, ``__hash__`` and graph merging. A key built
    only from shapes makes two Ops over different trials interchangeable, and the
    merged graph then scores one group's data twice and drops the other's. That
    is a wrong posterior rather than an error, so it is pinned here.
    """

    @staticmethod
    def _op(seed, n_samples=40, max_duration=20):
        from hmp.estimators.pytensor_op import HMPTrialLogLikelihood

        rng = np.random.default_rng(seed)
        half = n_samples // 2
        return HMPTrialLogLikelihood(
            cross_corr=rng.normal(size=(n_samples, 5)),
            trial_starts=np.array([0, half]),
            durations=np.array([half, half]),
            locations_samples=np.zeros(3),
            max_duration=max_duration,
            shift=0,
        )

    def test_same_data_is_equal(self):
        assert self._op(0) == self._op(0)
        assert hash(self._op(0)) == hash(self._op(0))

    def test_different_data_is_not_equal(self):
        """Same shapes and durations, different values: must not be interchangeable."""
        first, second = self._op(0), self._op(1)
        assert first._key != second._key
        assert first != second
        assert hash(first) != hash(second)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("cross_corr", np.random.default_rng(99).normal(size=(40, 5))),
            ("trial_starts", np.array([1, 21])),
            ("durations", np.array([19, 20])),
            ("locations_samples", np.array([0.0, 1.0, 0.0])),
            ("max_duration", 21),
            ("shift", 1),
        ],
    )
    def test_every_component_enters_the_key(self, field, value):
        """Changing any one of them must make the Ops distinguishable.

        The bug was a key built from shapes rather than values. Covering one
        component is not enough: whichever is left out lets two Ops carrying
        different data compare equal, and PyTensor then merges them.
        """
        from hmp.estimators.pytensor_op import HMPTrialLogLikelihood

        base = self._op(0)
        fields = {
            "cross_corr": base.cross_corr, "trial_starts": base.trial_starts,
            "durations": base.durations,
            "locations_samples": base.locations_samples,
            "max_duration": base.max_duration, "shift": base.shift,
        }
        fields[field] = value
        other = HMPTrialLogLikelihood(**fields)

        assert base._key != other._key, f"{field} does not reach the key"
        assert base != other
        assert hash(base) != hash(other)

    def test_graph_keeps_both_ops(self):
        """Two Ops on identical inputs must contribute their own likelihoods."""
        import pytensor
        import pytensor.tensor as pt

        first, second = self._op(0), self._op(1)
        channel_pars = np.random.default_rng(7).normal(size=(2, 5)) * 0.8
        time_pars = np.column_stack([np.full(3, 2.0), np.full(3, 3.0)])

        alone = [float(np.sum(np.asarray(op._call_jax(channel_pars, time_pars))))
                 for op in (first, second)]
        assert abs(alone[0] - alone[1]) > 1e-6, "the two Ops must differ to be a test"

        channel = pt.dmatrix("channel")
        time = pt.dmatrix("time")
        combined = pytensor.function(
            [channel, time],
            pt.sum(first(channel, time)) + pt.sum(second(channel, time)),
        )
        assert abs(float(combined(channel_pars, time_pars)) - sum(alone)) < 1e-6


class TestFullyTiedGroups:
    """Groups sharing every parameter still have to be scored on their own data.

    With a fully tied map both groups receive the same parameter expressions, so
    the only thing distinguishing their contributions is the Op itself.
    """

    def test_tied_groups_match_numpy(self, fitted_setup):
        pytest.importorskip("pymc")
        from scipy import stats

        from hmp.estimators.mcmc import MCMCEstimator

        pattern_data, _, n_events = fitted_setup
        n_dims = pattern_data.cross_corr.shape[1]
        channel_map = np.zeros((2, n_events), dtype=int)
        time_map = np.zeros((2, n_events + 1), dtype=int)
        model = EventModel(
            n_events=n_events, channel_map=channel_map, time_map=time_map,
            grouping_dict={"condition": ["a", "b"]},
        )
        model.n_dims = n_dims
        groups = np.arange(len(pattern_data.durations)) % 2

        estimator = MCMCEstimator()
        pymc_model = estimator.build_model(model, pattern_data, groups)
        channel_index, n_channel, _ = estimator._tie_index(channel_map)
        time_index, n_time, _ = estimator._tie_index(time_map)
        shape = float(model.distribution.shape)

        rng = np.random.default_rng(0)
        shared_channel = rng.normal(0, 1.0, (n_channel, n_dims))
        shared_scale = np.exp(rng.normal(np.log(5.0), 0.3, n_time))

        total = float(pymc_model.compile_logp()(
            {"channel_pars": shared_channel, "log_scale": np.log(shared_scale)}
        ))
        channel_sd = float(np.std(np.asarray(pattern_data.cross_corr))) * \
            MCMCEstimator.CHANNEL_PRIOR_WIDTH
        priors = (
            stats.norm.logpdf(shared_channel, 0.0, channel_sd).sum()
            + stats.norm.logpdf(
                np.log(shared_scale), np.log(estimator._even_scale), 1.0).sum()
        )

        scale = shared_scale[time_index]
        expected = model.log_likelihood(
            pattern_data,
            shared_channel[channel_index],
            np.stack([np.full(scale.shape, shape), scale], axis=-1),
            groups,
        )
        assert abs((total - priors) - expected) / abs(expected) < EXACT


if __name__ == "__main__":
    pytest.main([__file__])
