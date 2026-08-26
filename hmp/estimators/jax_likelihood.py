"""JAX implementation of the HMP likelihood.

Mirrors :meth:`hmp.models.event.EventModel.estim_probs` in a differentiable
form. Equation numbers refer to Weindel, van Maanen and Borst (2024).

Only the likelihood is reproduced, not the normalised event probabilities of
Eq 8, which are downstream of anything a sampler needs.

The reference values in ``tests/gen_data/likelihood_reference.npz`` are what
this has to reproduce.
"""

import jax
import jax.numpy as jnp
from jax.scipy.stats import gamma as jax_gamma

# double precision: needed to reproduce the numpy implementation exactly
jax.config.update("jax_enable_x64", True)


def gains(cross_corr, channel_pars):
    """Eq 2, the match between channels and each event.

    Parameters
    ----------
    cross_corr : (n_samples, n_channels) array
        Data cross-correlated with the pattern, stacked over trials.
    channel_pars : (n_events, n_channels) array
        Channel contributions to each event.

    Returns
    -------
    (n_samples, n_events) array
    """
    return jnp.exp(cross_corr @ channel_pars.T - 0.5 * jnp.sum(channel_pars**2, axis=1))


def stage_pmf(time_pars, locations_samples, max_duration, shift):
    """Eq 3, the discretised and normalised duration distribution per stage.

    The distribution is shifted, then normalised, and only then censored by the
    location, so the censored pmf does not sum to one.

    Parameters
    ----------
    time_pars : (n_stages, 2) array
        Shape and scale of each stage.
    locations_samples : (n_stages,) array
        Samples censored at the start of each stage.
    max_duration : int
    shift : int

    Returns
    -------
    (max_duration, n_stages) array
    """
    support = jnp.arange(max_duration, dtype=jnp.float64)

    def one_stage(pars, location):
        # never evaluate the density at zero: jnp.where masks the value but not
        # the gradient, and log(0) would give a NaN gradient
        censored = support < shift
        safe_support = jnp.where(censored, 1.0, support)
        pdf = jax_gamma.pdf(safe_support, pars[0], scale=pars[1])
        pdf = jnp.where(censored, 0.0, pdf)

        # likewise guard the denominator rather than repair the quotient
        total = jnp.sum(pdf)
        positive = total > 0
        pdf = jnp.where(positive, pdf / jnp.where(positive, total, 1.0), 0.0)

        return jnp.where(support < location, 0.0, pdf)

    return jax.vmap(one_stage)(time_pars, locations_samples).T


def _gather_trials(values, trial_starts, durations, max_duration, reverse):
    """Lay the stacked per-sample values out as (max_duration, n_trials, n_events).

    Trials are padded to the longest one. When ``reverse`` the samples of each
    trial are reversed within its own length and the events are reversed too,
    which is what lets the backward pass reuse the forward recursion.
    """
    offsets = jnp.arange(max_duration)
    if reverse:
        indices = trial_starts[:, None] + (durations[:, None] - 1 - offsets[None, :])
    else:
        indices = trial_starts[:, None] + offsets[None, :]
    within = offsets[None, :] < durations[:, None]
    indices = jnp.clip(indices, 0, values.shape[0] - 1)

    laid_out = jnp.where(within[:, :, None], values[indices], 0.0)
    laid_out = jnp.transpose(laid_out, (1, 0, 2))
    return laid_out[:, :, ::-1] if reverse else laid_out


def _convolve_per_trial(sequences, kernel, max_duration):
    """Convolve each trial's sequence with a shared kernel, truncated to length."""
    return jax.vmap(
        lambda sequence: jnp.convolve(sequence, kernel)[:max_duration],
        in_axes=1,
        out_axes=1,
    )(sequences)


def _unreverse_samples(values, durations, max_duration):
    """Undo the per-trial sample reversal, leaving the padding untouched."""
    offsets = jnp.arange(max_duration)
    indices = jnp.clip(durations[None, :] - 1 - offsets[:, None], 0, max_duration - 1)
    within = offsets[:, None] < durations[None, :]
    reversed_ = jnp.take_along_axis(values, indices[:, :, None], axis=0)
    return jnp.where(within[:, :, None], reversed_, values)


def trial_log_likelihood(  # noqa: PLR0913, PLR0917
    cross_corr,
    channel_pars,
    time_pars,
    trial_starts,
    durations,
    locations_samples,
    max_duration,
    shift,
):
    """Log-likelihood of each trial, Eq 5 to 7 and Eq 9 without the outer sum.

    Returns
    -------
    (n_trials,) array
    """
    n_events = channel_pars.shape[0]

    per_sample = gains(cross_corr, channel_pars)
    probs = _gather_trials(per_sample, trial_starts, durations, max_duration, False)
    probs_b = _gather_trials(per_sample, trial_starts, durations, max_duration, True)

    pmf = stage_pmf(time_pars, locations_samples, max_duration, shift)
    pmf_b = pmf[:, ::-1]

    # Eq 5 and 6, unrolled: the number of events is small and known
    forward = [pmf[:, 0][:, None] * probs[:, :, 0]]
    backward = [jnp.broadcast_to(pmf_b[:, 0][:, None], probs.shape[:2])]
    for event in range(1, n_events):
        carried = backward[event - 1] * probs_b[:, :, event - 1]
        forward.append(
            _convolve_per_trial(forward[event - 1], pmf[:, event], max_duration)
            * probs[:, :, event]
        )
        backward.append(_convolve_per_trial(carried, pmf_b[:, event], max_duration))

    forward = jnp.stack(forward, axis=-1)
    backward = jnp.stack(backward, axis=-1)[:, :, ::-1]
    backward = _unreverse_samples(backward, durations, max_duration)

    # Eq 7, then Eq 9 per trial. Summing over samples avoids taking log of zero.
    eventprobs = jnp.clip(forward * backward, 0.0, None)
    return jnp.log(jnp.sum(eventprobs[:, :, 0], axis=0))


def log_likelihood(  # noqa: PLR0913, PLR0917
    cross_corr,
    channel_pars,
    time_pars,
    trial_starts,
    durations,
    locations_samples,
    max_duration,
    shift,
):
    """Return the summed log-likelihood, Eq 9."""
    return jnp.sum(
        trial_log_likelihood(
            cross_corr, channel_pars, time_pars, trial_starts, durations,
            locations_samples, max_duration, shift,
        )
    )
