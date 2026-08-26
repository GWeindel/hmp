"""Parameter estimation methods for HMP models."""

from .base import BaseEstimator, EstimationResult
from .em import EMEstimator

__all__ = [
    "BaseEstimator",
    "EstimationResult",
    "EMEstimator",
    "MCMCEstimator",
]


def __getattr__(name):
    # imported on request: the sampler needs pymc and jax, which are an extra
    if name == "MCMCEstimator":
        from .mcmc import MCMCEstimator  # noqa: PLC0415

        return MCMCEstimator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
