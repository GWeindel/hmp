"""Classes for several probability distributions (Gamma, Lognormal, Wald, Weibull)."""

import numpy as np
from scipy.special import gamma as gamma_func
from scipy.stats import gamma, invgauss, lognorm, weibull_min


class Gamma():
    """
    Define a gamma distribution.

    This class represents a gamma distribution with a specified shape parameter.

    Parameters
    ----------
    shape : float, optional
        The shape parameter of the gamma distribution (default is 2).

    Attributes
    ----------
    shape : float
        The shape parameter of the gamma distribution.
    pdf : function
        The probability density function from `scipy.stats.gamma`.

    Methods
    -------
    scale_to_mean(scale: float) -> float
        Compute the mean of the distribution given a scale parameter.
    mean_to_scale(mean: float) -> float
        Compute the scale parameter of the distribution given a mean.
    """

    def __init__(self, shape=2):
        self.shape = shape
        self.pdf = gamma.pdf

    def scale_to_mean(self, scale: float) -> float:
        """
        Compute the mean associated with a given scale and shape parameters.

        Parameters
        ----------
        scale : float
            The scale parameter of the distribution.

        Returns
        -------
        float
            The calculated mean value.
        """
        return scale * self.shape

    def mean_to_scale(self, mean: float) -> float:
        """
        Compute the scale associated with a given mean and shape.

        Parameters
        ----------
        mean : float
            The mean value of the distribution.

        Returns
        -------
        float
            The calculated scale parameter.
        """
        return mean / self.shape


class Lognorm():
    """
    Define a Lognormal distribution.

    Parameters
    ----------
    shape : float
        The shape parameter of the lognormal distribution.

    Attributes
    ----------
    shape : float
        The shape parameter of the distribution.
    pdf : function
        The probability density function from `scipy.stats.lognorm`.

    Methods
    -------
    scale_to_mean(scale: float) -> float
        Compute the mean of the distribution given a scale parameter.
    mean_to_scale(mean: float) -> float
        Compute the scale parameter of the distribution given a mean.
    """

    def __init__(self, shape):
        self.shape = shape
        self.pdf = lognorm.pdf

    def scale_to_mean(self, scale: float) -> float:
        """
        Compute the mean associated with a given scale and shape parameters.

        Parameters
        ----------
        scale : float
            The scale parameter of the distribution.

        Returns
        -------
        float
            The calculated mean value.
        """
        return np.exp(scale + self.shape**2 / 2)

    def mean_to_scale(self, mean: float) -> float:
        """
        Compute the scale associated with a given mean and shape.

        Parameters
        ----------
        mean : float
            The mean value of the distribution.

        Returns
        -------
        float
            The calculated scale parameter.
        """
        return np.exp(np.log(mean) - (self.shape**2 / 2))

class Wald():
    """
    Define a Wald distribution (aka inverse Gaussian).

    Parameters
    ----------
    shape : float
        The shape parameter of the Wald distribution.

    Attributes
    ----------
    shape : float
        The shape parameter of the distribution.
    pdf : function
        The probability density function from `scipy.stats.invgauss`.

    Methods
    -------
    scale_to_mean(scale: float) -> float
        Compute the mean of the distribution given a scale parameter.
    mean_to_scale(mean: float) -> float
        Compute the scale parameter of the distribution given a mean.
    """

    def __init__(self, shape):
        self.shape = shape
        self.pdf = invgauss.pdf

    def scale_to_mean(self, scale: float) -> float:
        """
        Compute the mean associated with a given scale and shape parameters.

        Parameters
        ----------
        scale : float
            The scale parameter of the distribution.

        Returns
        -------
        float
            The calculated mean value.
        """
        return scale * self.shape

    def mean_to_scale(self, mean: float) -> float:
        """
        Compute the scale associated with a given mean and shape.

        Parameters
        ----------
        mean : float
            The mean value of the distribution.

        Returns
        -------
        float
            The calculated scale parameter.
        """
        return mean / self.shape

class Weibull():
    """
    Define a Weibull distribution.

    Parameters
    ----------
    shape : float
        The shape parameter of the Wald distribution.

    Attributes
    ----------
    shape : float
        The shape parameter of the distribution.
    pdf : function
        The probability density function from `scipy.stats.weibull_min`.

    Methods
    -------
    scale_to_mean(scale: float) -> float
        Compute the mean of the distribution given a scale parameter.
    mean_to_scale(mean: float) -> float
        Compute the scale parameter of the distribution given a mean.
    """

    def __init__(self, shape):
        self.shape = shape
        self.pdf = weibull_min.pdf
        self.gamma_func = gamma_func

    def scale_to_mean(self, scale: float) -> float:
        """
        Compute the mean associated with a given scale and shape parameters.

        Parameters
        ----------
        scale : float
            The scale parameter of the distribution.

        Returns
        -------
        float
            The calculated mean value.
        """
        return scale * self.gamma_func(1 + 1 / self.shape)

    def mean_to_scale(self, mean: float) -> float:
        """
        Compute the scale associated with a given mean and shape.

        Parameters
        ----------
        mean : float
            The mean value of the distribution.

        Returns
        -------
        float
            The calculated scale parameter.
        """
        return mean / self.gamma_func(1 + 1 / self.shape)

