"""PyTensor wrapper around the JAX likelihood, so it can be used inside PyMC.

The Op carries the data and static shapes; the parameter arrays are its inputs.
Registering with ``jax_funcify`` lets the graph compile to JAX for
``nuts_sampler="numpyro"``; ``perform`` keeps it working under the default
backend.
"""

import jax
import numpy as np
import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.link.jax.dispatch import jax_funcify

from hmp.distributions import Gamma
from hmp.estimators import jax_likelihood as jl


class HMPLogLikelihood(Op):
    """Summed log-likelihood of an HMP model as a PyTensor Op.

    Parameters
    ----------
    cross_corr : (n_samples, n_channels) ndarray
        Data cross-correlated with the pattern, stacked over trials.
    trial_starts, durations : (n_trials,) ndarray
        Where each trial begins in the stacked data, and how long it is.
    locations_samples : (n_stages,) ndarray
        Samples censored at the start of each stage.
    max_duration : int
    shift : int
    """

    __props__ = ("_key",)

    def __init__(  # noqa: PLR0913, PLR0917
        self, cross_corr, trial_starts, durations, locations_samples,
        max_duration, shift,
    ):
        self.cross_corr = np.asarray(cross_corr, dtype=np.float64)
        self.trial_starts = np.asarray(trial_starts, dtype=np.int64)
        self.durations = np.asarray(durations, dtype=np.int64)
        self.locations_samples = np.asarray(locations_samples, dtype=np.float64)
        self.max_duration = int(max_duration)
        self.shift = int(shift)
        # PyTensor treats Ops with equal __props__ as interchangeable and will
        # merge them, so the key has to identify the data. Shapes alone collide
        # between groups of equal size and the wrong group's likelihood is used.
        self._jitted = self._build_jitted()
        self._key = (
            self.max_duration,
            self.shift,
            hash(self.cross_corr.tobytes()),
            hash(self.trial_starts.tobytes()),
            hash(self.durations.tobytes()),
            hash(self.locations_samples.tobytes()),
        )

    def _call_jax(self, channel_pars, time_pars):
        return self._jitted(channel_pars, time_pars)

    def _build_jitted(self):
        """Compile the forward pass once; `perform` would otherwise retrace it.

        Measured at 59 ms unjitted against 28 ms jitted on a 40 trial model.
        """
        cross_corr = self.cross_corr
        trial_starts = self.trial_starts
        durations = self.durations
        locations = self.locations_samples
        max_duration = self.max_duration
        shift = self.shift

        def call(channel_pars, time_pars):
            return jl.log_likelihood(cross_corr, channel_pars, time_pars,
                                     trial_starts, durations, locations,
                                     max_duration, shift)

        return jax.jit(call)

    def make_node(self, channel_pars, time_pars):
        """Declare a scalar output; PyTensor requires perform to match this exactly."""
        channel_pars = pt.as_tensor_variable(channel_pars)
        time_pars = pt.as_tensor_variable(time_pars)
        return Apply(self, [channel_pars, time_pars], [pt.dscalar()])

    def perform(self, node, inputs, outputs):
        """Evaluate under the default backend."""
        channel_pars, time_pars = inputs
        outputs[0][0] = np.asarray(
            self._call_jax(channel_pars, time_pars), dtype=node.outputs[0].dtype
        )

    def grad(self, inputs, output_gradients):
        """Differentiate under the default backend.

        Compiling the graph to JAX makes this unnecessary, since JAX
        differentiates the whole thing. It is here so the Op is also usable with
        the default backend, which find_MAP and the default NUTS need.
        """
        channel_pars, time_pars = inputs
        grad_channel, grad_time = HMPLogLikelihoodGrad(self)(channel_pars, time_pars)
        return [output_gradients[0] * grad_channel, output_gradients[0] * grad_time]


class HMPLogLikelihoodGrad(Op):
    """Gradient of :class:`HMPLogLikelihood` with respect to both parameter arrays."""

    __props__ = ("_key",)

    def __init__(self, likelihood_op):
        self.likelihood_op = likelihood_op
        self._key = likelihood_op._key
        self._grad = jax.jit(jax.grad(likelihood_op._call_jax, argnums=(0, 1)))

    def make_node(self, channel_pars, time_pars):
        channel_pars = pt.as_tensor_variable(channel_pars)
        time_pars = pt.as_tensor_variable(time_pars)
        return Apply(
            self, [channel_pars, time_pars],
            [channel_pars.type(), time_pars.type()],
        )

    def perform(self, node, inputs, outputs):
        grad_channel, grad_time = self._grad(*inputs)
        outputs[0][0] = np.asarray(grad_channel, dtype=node.outputs[0].dtype)
        outputs[1][0] = np.asarray(grad_time, dtype=node.outputs[1].dtype)


@jax_funcify.register(HMPLogLikelihood)
def _hmp_log_likelihood_jax(op, **kwargs):  # noqa: ARG001
    """Return the unjitted function so the surrounding graph can be jitted as one."""
    def log_likelihood(channel_pars, time_pars):
        return op._call_jax(channel_pars, time_pars)

    return log_likelihood


@jax_funcify.register(HMPLogLikelihoodGrad)
def _hmp_log_likelihood_grad_jax(op, **kwargs):  # noqa: ARG001
    def log_likelihood_grad(channel_pars, time_pars):
        return jax.grad(op.likelihood_op._call_jax, argnums=(0, 1))(
            channel_pars, time_pars
        )

    return log_likelihood_grad


def build_op(pattern_data, model, subset=None):
    """Build the Op for a model and its data, taking the layout from the model.

    Parameters
    ----------
    pattern_data : PatternData
        Should be float64, otherwise the likelihood is only good to float32.
    model : EventModel
        Supplies the censoring locations and the duration distribution.
    subset : ndarray of bool, optional
        Trials to include. Defaults to all of them.

    Returns
    -------
    HMPLogLikelihood

    Raises
    ------
    NotImplementedError
        For any duration distribution other than the gamma, which is the only
        density :mod:`hmp.estimators.jax_likelihood` implements. Only the shape
        is read from the model, so another family would be fitted as a gamma.
    """
    if not isinstance(model.distribution, Gamma):
        raise NotImplementedError(
            "sampling implements the gamma duration distribution only, not "
            f"{type(model.distribution).__name__}"
        )
    starts = np.asarray(pattern_data.starts)
    ends = np.asarray(pattern_data.ends)
    if subset is not None:
        starts, ends = starts[subset], ends[subset]

    durations = ends - starts + 1
    stacked = np.vstack(
        [np.asarray(pattern_data.cross_corr)[s:e + 1] for s, e in zip(starts, ends)]
    )
    trial_starts = np.concatenate([[0], np.cumsum(durations)[:-1]])

    locations = model._time_to_samples(model.locations, pattern_data.sfreq).astype(float)
    locations[1:-1] -= model.distribution.shift

    return HMPLogLikelihood(
        cross_corr=stacked,
        trial_starts=trial_starts,
        durations=durations,
        locations_samples=locations,
        max_duration=int(durations.max()),
        shift=int(model.distribution.shift),
    )


class HMPTrialLogLikelihood(HMPLogLikelihood):
    """Per-trial log-likelihoods, rather than their sum.

    Needed for leave-one-out cross validation and model comparison, which
    require one value per observation.
    """

    def _build_jitted(self):
        """Compile the per-trial counterpart once, for the same reason."""
        cross_corr = self.cross_corr
        trial_starts = self.trial_starts
        durations = self.durations
        locations = self.locations_samples
        max_duration = self.max_duration
        shift = self.shift

        def call(channel_pars, time_pars):
            return jl.trial_log_likelihood(cross_corr, channel_pars, time_pars,
                                           trial_starts, durations, locations,
                                           max_duration, shift)

        return jax.jit(call)

    def make_node(self, channel_pars, time_pars):
        """Declare one output per trial."""
        channel_pars = pt.as_tensor_variable(channel_pars)
        time_pars = pt.as_tensor_variable(time_pars)
        return Apply(self, [channel_pars, time_pars], [pt.dvector()])

    def grad(self, inputs, output_gradients):
        """Vector-Jacobian product against the incoming cotangent."""
        channel_pars, time_pars = inputs
        grad_op = HMPTrialLogLikelihoodGrad(self)
        grad_channel, grad_time = grad_op(channel_pars, time_pars,
                                          output_gradients[0])
        return [grad_channel, grad_time]


class HMPTrialLogLikelihoodGrad(Op):
    """Vector-Jacobian product of :class:`HMPTrialLogLikelihood`."""

    __props__ = ("_key",)

    def __init__(self, likelihood_op):
        self.likelihood_op = likelihood_op
        self._key = likelihood_op._key

        def vjp(channel_pars, time_pars, cotangent):
            _, pullback = jax.vjp(
                likelihood_op._call_jax, channel_pars, time_pars
            )
            return pullback(cotangent)

        self._vjp = jax.jit(vjp)

    def make_node(self, channel_pars, time_pars, cotangent):
        channel_pars = pt.as_tensor_variable(channel_pars)
        time_pars = pt.as_tensor_variable(time_pars)
        cotangent = pt.as_tensor_variable(cotangent)
        return Apply(
            self, [channel_pars, time_pars, cotangent],
            [channel_pars.type(), time_pars.type()],
        )

    def perform(self, node, inputs, outputs):
        grad_channel, grad_time = self._vjp(*inputs)
        outputs[0][0] = np.asarray(grad_channel, dtype=node.outputs[0].dtype)
        outputs[1][0] = np.asarray(grad_time, dtype=node.outputs[1].dtype)


@jax_funcify.register(HMPTrialLogLikelihood)
def _hmp_trial_log_likelihood_jax(op, **kwargs):  # noqa: ARG001
    def trial_log_likelihood(channel_pars, time_pars):
        return op._call_jax(channel_pars, time_pars)

    return trial_log_likelihood


@jax_funcify.register(HMPTrialLogLikelihoodGrad)
def _hmp_trial_log_likelihood_grad_jax(op, **kwargs):  # noqa: ARG001
    def trial_log_likelihood_grad(channel_pars, time_pars, cotangent):
        _, pullback = jax.vjp(
            op.likelihood_op._call_jax, channel_pars, time_pars
        )
        return pullback(cotangent)

    return trial_log_likelihood_grad


def build_trial_op(pattern_data, model, subset=None):
    """Build the per-trial counterpart of :func:`build_op`, sharing its layout."""
    scalar_op = build_op(pattern_data, model, subset=subset)
    return HMPTrialLogLikelihood(
        cross_corr=scalar_op.cross_corr,
        trial_starts=scalar_op.trial_starts,
        durations=scalar_op.durations,
        locations_samples=scalar_op.locations_samples,
        max_duration=scalar_op.max_duration,
        shift=scalar_op.shift,
    )
