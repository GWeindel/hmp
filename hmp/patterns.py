"""Classes for generating and representing templates for HMP event detection.

Main class Pattern and including a half-sine wave template (`HalfSine`) and an arbitrary waveform template (`Arbitrary`).

Classes
-------
Pattern - Main class
    HalfSine
        Generates a normalized half-sine wave template for use in 
        signal processing or event detection.
    Arbitrary
        Allows the use of any arbitrary pattern as a template.

Both classes provide methods to create expected templates based on sampling frequency and other
parameters,
and store relevant metadata such as template width and censoring location for model fitting
procedures.
"""
from abc import ABC, abstractmethod
from warnings import warn
import numpy as np


class Pattern(ABC):
    """
    Parameters
    ----------
    width_ms: float
        Width of pattern in ms. 
    sfreq : float
        Sampling frequency in Hz.
    width_samples : int
        Number of samples in the half-sine wave. 
    template : np.ndarray
        The pattern template.
    """

    def __init__(
        self,
        width_ms: float,
        sfreq: float = None,
        template: np.ndarray = None
    ):
        self.width_ms = width_ms
        self.sfreq = sfreq

        if sfreq is not None:
            steps = 1000 / sfreq
            self.width_samples = int(np.rint(self.width_ms / steps))
        else:
            self.width_samples = None

        self.template = template

    @abstractmethod
    def create_template(self):
        """
        This methods needs to fill sfreq, width_samples, location, and template.
        This requires parameters from the data that you fit, might not be availble at
        the moment the pattern itself is defined.
        """
        ...

class HalfSine(Pattern):
    """
    Create a HalfSine instance with the expected parameters.

    Parameters
    ----------

    width_ms : float, optional
        Width of the half-sine wave in milliseconds, by default 50 ms (10H).
        Controls for the precision of the estimate. Shorter values will
        model narrower half-sines (i.e. higher frequencies), higher values
        will model wider events (i.e. lower frequencies)
        Default is 50 ms
    sfreq : float
        Sampling frequency of the modelled signal in Hz. If None, required to be filled
        by calling create_template later.

    """

    def __init__(
        self,
        width_ms: float = 50,
        sfreq: float = None
    ):
        super().__init__(width_ms, sfreq, template=None)

        if sfreq is not None:
            self.create_template(sfreq)

    def create_template(self, sfreq: float):
        """
        Create a HalfSine template with the expected parameters.

        Parameters
        ----------
        sfreq : float
            Sampling frequency of the modelled signal in Hz.
        
        Returns
        -------
        HalfSine
            An instance of the HalfSine class.
        """
        self.sfreq = sfreq
        steps = 1000 / sfreq
        self.width_samples = int(np.rint(self.width_ms / steps))
        if self.width_samples < 5:
            warn('Using a pattern defined by less than 5 points is not recommended')
        if self.width_samples < 2:
            raise ValueError("Cannot use pattern with only one data point")
       
        self.template = self._create_template(steps)

    def _create_template(self,steps: float) -> np.ndarray:
        """
        Compute the event shape as a half-sine wave.

        Parameters
        ----------
        steps : float
            Time step in milliseconds between samples.
      
        Returns
        -------
        np.ndarray
            The normalized half-sine wave template.
        """
        event_idx = np.arange(self.width_samples) * steps + steps / 2
        event_frequency = 1000 / (self.width_samples * steps * 2)  # Event frequency for half-sine
        template = np.sin(2 * np.pi * event_idx / 1000 * event_frequency)
        template = template / np.sum(template**2)  # Weight normalized
        return template