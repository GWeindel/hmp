"""Classes for generating and representing templates for HMP event detection.

Main class Pattern and including a half-sine wave template (`HalfSine`)

Classes
-------
Pattern - Main class
    HalfSine
        Generates a normalized half-sine wave template for use in
        signal processing or event detection.

"""
from abc import ABC

import numpy as np


class Pattern(ABC):
    """General class to be passed to models.

    Parameters
    ----------
    template : np.ndarray
        The pattern template.
    width : int, optional
        Length of the pattern (at 1000Hz).
    """

    def __init__(
        self,
        template: np.ndarray,
        width: int | None = None,
    ):
        if width is None:
            width = len(template)
        self.width = width
        self.template = template

class HalfSine(Pattern):
    """
    Create a HalfSine instance with the expected parameters.

    Parameters
    ----------
    width : float, optional
        Width of the half-sine wave in milliseconds, by default 50 ms (10Hz).
        Controls for the precision of the estimate. Shorter values will
        model narrower half-sines (i.e. higher frequencies), higher values
        will model wider events (i.e. lower frequencies)
    """

    def __init__(
        self,
        width: float = 50,
    ):
        template = self.create_template(width)
        super().__init__(template=template, width=width)

    @staticmethod
    def create_template(width: float):
        """Create a HalfSine template with the expected parameters."""
        event_idx = np.arange(width) + .5
        event_frequency = 1000 / (width * 2)  # Event frequency for half-sine
        template = np.sin(2 * np.pi * event_idx / 1000 * event_frequency)
        return template
