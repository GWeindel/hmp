"""Tests for the estimation interface."""

import numpy as np
import pytest
from test_io import init_data

import hmp
from hmp.estimators import BaseEstimator, EMEstimator, EstimationResult
from hmp.models import EventModel
from hmp.patterndata import PatternData


@pytest.fixture(scope="module")
def projected():
    """Read and project the noiseless recording once for the whole module."""
    _, _, epoch_data, _, _, n_events = init_data()
    hmp_data = hmp.basedata.default(
        epoch_data, n_comp=3, center=True, duration_id="response_time"
    )
    return hmp_data.select_coord("b", "subject"), n_events


@pytest.fixture
def pdata(projected):
    """Build fresh PatternData per test from the shared projection."""
    base_data, n_events = projected
    return PatternData.from_basedata(base_data), n_events


@pytest.fixture
def pdata_f64(projected_all):
    """All trials, in float64, for assertions about exact arithmetic.

    The data is float32 by default, which caps agreement between two
    differently accumulated sums at around 1e-7 however many trials there are,
    so float64 keeps precision from being the thing under test. It also has to
    be all the trials rather than one participant: a single participant here is
    two identical noiseless trials, and adding a float to itself is exact, so
    the comparison would hold in float32 too and could not detect the float64
    path regressing.
    """
    base_data, n_events = projected_all
    return PatternData.from_basedata(base_data, dtype=np.float64), n_events


@pytest.fixture(scope="module")
def projected_all():
    """Both participants, needed to exercise grouped models."""
    _, _, epoch_data, _, _, n_events = init_data()
    hmp_data = hmp.basedata.default(
        epoch_data, n_comp=3, center=True, duration_id="response_time"
    )
    return hmp_data, n_events


@pytest.fixture
def pdata_all(projected_all):
    """Build fresh PatternData over all trials from the shared projection."""
    base_data, n_events = projected_all
    return PatternData.from_basedata(base_data), n_events


class MockEstimator(BaseEstimator):
    """Minimal estimator used to check that injection is honored."""

    def __init__(self):
        super().__init__()
        self.fit_called = False

    def fit(self, model, pattern_data, initial_channel_pars, initial_time_pars,  # noqa: ARG002
            groups=None, cpus=1):  # noqa: ARG002
        """Return the starting point untouched, without running any estimation."""
        self.fit_called = True
        self.fitted = True
        return EstimationResult(
            channel_pars=initial_channel_pars[0],
            time_pars=initial_time_pars[0],
            likelihood=100.0,
            converged=True,
            n_iterations=0,
            diagnostics={
                "traces": np.array([100.0]),
                "traces_group": np.array([[100.0]]),
                "time_pars_dev": initial_time_pars[:1],
            },
        )


class TestEstimationResult:
    """The container passed back from every estimator."""

    def test_creation(self):
        result = EstimationResult(
            channel_pars=np.array([[1, 2], [3, 4]]),
            time_pars=np.array([[0.5, 1.0]]),
            likelihood=100.0,
            converged=True,
            n_iterations=10,
            diagnostics={"traces": np.array([1.0])},
        )
        assert result.likelihood == 100.0
        assert result.converged is True
        assert result.n_iterations == 10
        assert result.uncertainty is None

    def test_uncertainty_is_optional(self):
        """Bayesian estimators may attach uncertainty; EM leaves it unset."""
        result = EstimationResult(
            channel_pars=np.array([[1, 2]]),
            time_pars=np.array([[0.5, 1.0]]),
            likelihood=50.0,
            converged=False,
            n_iterations=5,
            uncertainty={"channel_std": np.array([[0.1, 0.2]])},
        )
        assert "channel_std" in result.uncertainty
        assert result.diagnostics == {}


class TestBaseEstimator:
    """The abstract interface all estimation methods implement."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseEstimator()

    def test_subclass_interface(self):
        estimator = MockEstimator()
        assert not estimator.is_fitted
        assert estimator.get_method_name() == "MockEstimator"
        assert not estimator.supports_uncertainty()


class TestEMEstimator:
    """EMEstimator construction."""

    def test_defaults(self):
        em = EMEstimator()
        assert em.tolerance == 1e-4
        assert em.max_iteration == 1e3
        assert em.min_iteration == 1
        assert em.n_cor == 30
        assert not em.is_fitted
        assert em.get_method_name() == "EMEstimator"
        assert not em.supports_uncertainty()

    def test_custom(self):
        em = EMEstimator(tolerance=1e-3, max_iteration=500, min_iteration=5, n_cor=10)
        assert em.tolerance == 1e-3
        assert em.max_iteration == 500
        assert em.min_iteration == 5
        assert em.n_cor == 10


class TestEstimatorInjection:
    """EventModel.fit delegates the estimation to a swappable estimator."""

    def test_injected_em_matches_default(self, pdata):
        """Passing an equivalent EMEstimator must not change the fit at all.

        This is the guard on the refactor: routing the estimation through
        EMEstimator has to reproduce the in-model implementation exactly.
        """
        pdata_b, n_events = pdata

        default_model = EventModel(n_events=n_events)
        default_model.fit(pdata_b, verbose=False)

        injected_model = EventModel(n_events=n_events)
        injected_model.fit(pdata_b, verbose=False, estimator=EMEstimator())

        assert injected_model.lkhs == default_model.lkhs
        assert np.array_equal(injected_model.time_pars, default_model.time_pars)
        assert np.array_equal(injected_model.channel_pars, default_model.channel_pars)

    def test_estimator_settings_change_the_fit(self, pdata):
        """A coarser estimator must actually stop earlier.

        Without this, `test_injected_em_matches_default` would also pass if the
        estimator argument were silently ignored.
        """
        pdata_b, n_events = pdata

        fine = EventModel(n_events=n_events)
        fine.fit(pdata_b, verbose=False)

        coarse = EventModel(n_events=n_events)
        coarse.fit(pdata_b, verbose=False,
                   estimator=EMEstimator(tolerance=1e-1, max_iteration=2))

        assert len(coarse.traces) < len(fine.traces)

    def test_custom_estimator_is_used(self, pdata):
        """A non-EM estimator drives the fit and populates the model."""
        pdata_b, n_events = pdata

        model = EventModel(n_events=n_events)
        mock = MockEstimator()
        result = model.fit(pdata_b, verbose=False, estimator=mock)

        assert mock.fit_called
        assert mock.is_fitted
        assert model._fitted
        assert result.likelihood == 100.0
        assert model.lkhs == result.likelihood
        assert np.array_equal(model.channel_pars, result.channel_pars)
        assert np.array_equal(model.time_pars, result.time_pars)

    def test_result_is_returned_and_stored(self, pdata):
        pdata_b, n_events = pdata

        model = EventModel(n_events=n_events)
        result = model.fit(pdata_b, verbose=False)

        assert isinstance(result, EstimationResult)
        assert model.estimation_result is result
        assert result.converged
        assert result.n_iterations >= 1

    def test_diagnostics_structure(self, pdata):
        pdata_b, n_events = pdata

        model = EventModel(n_events=n_events)
        result = model.fit(pdata_b, verbose=False)

        for key in ("traces", "traces_group", "time_pars_dev", "lkhs"):
            assert key in result.diagnostics

        traces = result.diagnostics["traces"]
        assert isinstance(traces, np.ndarray)
        assert len(traces) == result.n_iterations + 1
        assert result.diagnostics["time_pars_dev"].shape[1:] == result.time_pars.shape


class TestBackwardCompatibility:
    """Existing entry points keep working through the estimator."""

    def test_fit_transform(self, pdata):
        pdata_b, n_events = pdata

        model = EventModel(n_events=n_events)
        lkh, estimates = model.fit_transform(pdata_b, verbose=False)

        assert isinstance(lkh, (float, np.floating, np.ndarray))
        assert hasattr(estimates, "dims")

    def test_multiple_starting_points(self, pdata):
        """Selection across starting points is the estimator's responsibility.

        Note `_format_parameters` builds ``starting_points + 1`` time parameter
        sets but only ``starting_points`` channel parameter sets, so pairing
        them drops the last proposal. That is pre-existing model behaviour,
        kept as-is here; the estimator simply evaluates every pair it is given.
        """
        pdata_b, n_events = pdata

        model = EventModel(n_events=n_events, starting_points=3)
        result = model.fit(pdata_b, verbose=False)

        assert len(result.diagnostics["lkhs"]) == model.starting_points
        assert result.likelihood == np.max(result.diagnostics["lkhs"])


class TestLikelihoodSurface:
    """The public contract an estimator scores parameters through."""

    def test_per_trial_sums_to_total(self, pdata_f64):
        """The per-trial values must be the total, split up, not merely close to it."""
        pdata_b, n_events = pdata_f64
        model = EventModel(n_events=n_events)
        model.fit(pdata_b, verbose=False)

        total = model.log_likelihood(pdata_b, model.channel_pars, model.time_pars)
        per_trial = model.log_likelihood(
            pdata_b, model.channel_pars, model.time_pars, per_trial=True
        )

        assert len(per_trial) == len(pdata_b.durations)
        assert np.isclose(per_trial.sum(), total, atol=0, rtol=1e-12)

    def test_matches_the_reported_fit(self, pdata):
        """The public surface reproduces the likelihood the fit itself reported."""
        pdata_b, n_events = pdata
        model = EventModel(n_events=n_events)
        model.fit(pdata_b, verbose=False)

        assert model.log_likelihood(
            pdata_b, model.channel_pars, model.time_pars
        ) == model.lkhs

    def test_evaluable_without_fitting(self, pdata):
        """Samplers score arbitrary parameters, so neither entry point may need a fit."""
        pdata_b, n_events = pdata
        fitted = EventModel(n_events=n_events)
        fitted.fit(pdata_b, verbose=False)

        fresh = EventModel(n_events=n_events)
        eventprobs = fresh.event_probabilities(
            pdata_b, fitted.channel_pars, fitted.time_pars
        )

        assert eventprobs.dims == ("trial", "sample", "event")
        assert np.isclose(
            eventprobs.trial_lkh.sum(), eventprobs.likelihood, atol=0, rtol=1e-12
        )

    def test_grouped_per_trial_keeps_trial_order(self, pdata_all):
        """Per-trial values must land on the trial they belong to, not the group.

        Each group is scored on its own subset, so the results have to be scattered
        back into the original trial order. Relabelling the groups and swapping the
        parameters to match describes the same model, so every trial must keep its
        value. A scatter that wrote results in group order would sum correctly but
        fail here.
        """
        pdata_a, n_events = pdata_all
        n_trials = len(pdata_a.durations)

        reference = EventModel(n_events=n_events)
        reference.fit(pdata_a, verbose=False)

        grouped = EventModel(
            n_events=n_events,
            channel_map=np.zeros((2, n_events)),
            time_map=np.zeros((2, n_events + 1)),
            grouping_dict={"half": ["first", "second"]},
        )
        first = np.tile(reference.channel_pars, (2, 1, 1))
        second = np.tile(reference.time_pars, (2, 1, 1))
        second[1, :, 1] *= 1.05  # make the two groups genuinely differ

        alternating = np.tile([0, 1], n_trials // 2)
        as_labelled = grouped.log_likelihood(
            pdata_a, first, second, groups=alternating, per_trial=True
        )
        as_relabelled = grouped.log_likelihood(
            pdata_a, first[::-1], second[::-1], groups=1 - alternating, per_trial=True
        )

        assert np.allclose(as_labelled, as_relabelled, atol=0, rtol=1e-10)


class TestParallelExecution:
    """Spreading starting points over cpus must not change the answer."""

    @staticmethod
    def _starting_points(model, pattern_data, n_events, n_points=2):
        """Build explicit starting points.

        `gen_random_stages` draws from an unseeded `default_rng`, so letting the
        model generate its own would give the two runs different starting points
        and make the comparison meaningless.
        """
        n_dims = pattern_data.cross_corr.shape[1]
        mean_duration = float(pattern_data.durations.mean())
        channel_pars = np.zeros((n_points, 1, n_events, n_dims))
        time_pars = np.zeros((n_points, 1, n_events + 1, 2))
        for point in range(n_points):
            time_pars[point, 0, :, 0] = model.distribution.shape
            time_pars[point, 0, :, 1] = model.distribution.mean_to_scale(
                mean_duration / (n_events + 1) * (1 + 0.1 * point)
            )
        return channel_pars, time_pars

    def test_multiprocessing_matches_serial(self, pdata):
        pdata_b, n_events = pdata
        channel_pars, time_pars = self._starting_points(
            EventModel(n_events=n_events), pdata_b, n_events
        )

        serial = EventModel(n_events=n_events)
        serial.fit(pdata_b, channel_pars=channel_pars.copy(),
                   time_pars=time_pars.copy(), verbose=False, cpus=1)

        parallel = EventModel(n_events=n_events)
        parallel.fit(pdata_b, channel_pars=channel_pars.copy(),
                     time_pars=time_pars.copy(), verbose=False, cpus=2)

        assert serial.lkhs == parallel.lkhs
        assert np.array_equal(serial.time_pars, parallel.time_pars)
        assert np.array_equal(serial.channel_pars, parallel.channel_pars)


if __name__ == "__main__":
    pytest.main([__file__])
