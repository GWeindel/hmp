"""Base classes for parameter estimation in HMP models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from hmp.patterndata import PatternData


@dataclass
class EstimationResult:
    """Results from parameter estimation.

    Parameters
    ----------
    channel_pars : np.ndarray
        Estimated channel parameters
    time_pars : np.ndarray
        Estimated time distribution parameters
    likelihood : float
        Final log-likelihood value
    converged : bool
        Whether estimation converged
    n_iterations : int
        Number of iterations performed
    diagnostics : dict
        Estimation-specific diagnostic information
    uncertainty : dict, optional
        Parameter uncertainty measures (for Bayesian methods)
    """

    channel_pars: np.ndarray
    time_pars: np.ndarray
    likelihood: float
    converged: bool
    n_iterations: int
    diagnostics: dict[str, Any] = field(default_factory=dict)
    uncertainty: Optional[dict[str, Any]] = None


class BaseEstimator(ABC):
    """Abstract base class for parameter estimation methods.

    This class defines the interface that all parameter estimation methods
    must implement to work with HMP models.
    """

    def __init__(self):
        """Initialize the estimator."""
        self.fitted = False

    @abstractmethod
    def fit(
        self,
        model,
        pattern_data: PatternData,
        initial_channel_pars: np.ndarray,
        initial_time_pars: np.ndarray,
        groups: np.ndarray = None,
        cpus: int = 1,
    ) -> EstimationResult:
        """Estimate model parameters.

        Parameters
        ----------
        model : BaseModel
            Model providing the likelihood and the expectation step.
        pattern_data : PatternData
            Preprocessed data cross-correlated with the pattern of the model.
        initial_channel_pars : np.ndarray
            Initial channel parameter values, one per starting point.
        initial_time_pars : np.ndarray
            Initial time distribution parameter values, one per starting point.
        groups : np.ndarray, optional
            Array indicating the groups for grouping modeling. Default is None.
        cpus : int, optional
            Number of cores to use in multiprocessing functions. Default is 1.

        Returns
        -------
        EstimationResult
            Results of parameter estimation
        """
        pass

    @property
    def is_fitted(self) -> bool:
        """Whether the estimator has been fitted."""
        return self.fitted

    def get_method_name(self) -> str:
        """Get the name of the estimation method."""
        return self.__class__.__name__

    def supports_uncertainty(self) -> bool:
        """Whether this estimator provides uncertainty estimates."""
        return False
