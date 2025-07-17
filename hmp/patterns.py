"""Classes for generating and representing templates for HMP event detection.

including a half-sine wave template (`HalfSine`) and an arbitrary waveform template (`Arbitrary`).

Classes
-------
HalfSine
    Generates a normalized half-sine wave template for use in signal processing or event detection.
Arbitrary
    Allows the use of any arbitrary pattern as a template.

Both classes provide methods to create expected templates based on sampling frequency and other
parameters,
and store relevant metadata such as template width and censoring location for model fitting
procedures.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class HalfSine:
    """
    Represents a half-sine wave template.

    Attributes
    ----------
    sfreq : float
        Sampling frequency in Hz.
    width_samples : int
        Number of samples in the half-sine wave.
    location : int
        How much samples should be censored in the EM() step of model fitting.
    template : np.ndarray
        The half-sine wave template.
    """

    sfreq: float
    width_samples: int
    location: int
    template: np.ndarray

    @classmethod
    def create_expected(cls, sfreq: float, width: float = 50,
                        location: float | None = None) -> "HalfSine":
        """
        Create a HalfSine instance with the expected parameters.

        Parameters
        ----------
        sfreq : float
            Sampling frequency in Hz.
        width : float, optional
            Width of the half-sine wave in milliseconds, by default 50.
        location : float, optional
            How much samples should be censored in the EM() step of model fitting.

        Returns
        -------
        HalfSine
            An instance of the HalfSine class.
        """
        steps = 1000 / sfreq
        width_samples = int(np.round(width / steps))
        if location is None:
            location = int(width / steps)
        else:
            location = int(np.rint(location))
        template = cls._create_template(width_samples, steps, width)
        return cls(sfreq, width_samples, location, template)

    @staticmethod
    def _create_template(width_samples: int, steps: float, width: float) -> np.ndarray:
        """
        Compute the event shape as a half-sine wave.

        Parameters
        ----------
        width_samples : int
            Number of samples in the half-sine wave.
        steps : float
            Time step in milliseconds between samples.
        width : float
            Width of the half-sine wave in milliseconds.

        Returns
        -------
        np.ndarray
            The normalized half-sine wave template.
        """
        event_idx = np.arange(width_samples) * steps + steps / 2
        event_frequency = 1000 / (width * 2)  # Event frequency for half-sine
        template = np.sin(2 * np.pi * event_idx / 1000 * event_frequency)
        template = template / np.sum(template**2)  # Weight normalized
        return template

@dataclass
class Arbitrary:
    """
    Represents an arbitrary template.

    Attributes
    ----------
    sfreq : float
        Sampling frequency in Hz.
    width_samples : int
        Number of samples in the template.
    location : int
        How much samples should be censored in the EM() step of model fitting.
    template : np.ndarray
        The arbitrary template.
    """

    sfreq: float
    width_samples: int
    location: int
    template: np.ndarray

    @classmethod
    def create_expected(cls, sfreq: float, template: np.ndarray,
                        location: float | None = None) -> "Arbitrary":
        """
        Create an Arbitrary instance with the expected parameters.

        Parameters
        ----------
        sfreq : float
            Sampling frequency in Hz.
        template : np.ndarray
            The arbitrary waveform template.
        location : float, optional
            How much samples should be censored in the EM() step of model fitting.

        Returns
        -------
        Arbitrary
            An instance of the Arbitrary class.
        """
        width_samples = len(template)
        if location is None:
            location = width_samples
        else:
            location = int(np.rint(location))
        return cls(sfreq, width_samples, location, template)
