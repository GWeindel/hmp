import numpy as np
from typing import Any
from dataclasses import dataclass

@dataclass
class HalfSine():
    sfreq: float
    width_samples: int
    location: int
    template: Any

    @classmethod
    def create_expected(cls, sfreq, width=50,  location=None):

        steps = 1000 / sfreq

        width_samples = int(np.round(width / steps))
        if location is None:
            location = int(width / steps)
        else:
            location = int(np.rint(location))
        template = cls._create_template(width_samples, steps, width)
        return cls(sfreq, width_samples, location, template)


    @staticmethod
    def _create_template(width_samples, steps, width):
        """Compute the event shape.

        Computes the template of a half-sine (event) with given frequency f and sampling frequency.

        Equations in section 2.4 in the 2024 paper
        """
        event_idx = np.arange(width_samples) * steps + steps / 2
        # gives event frequency given that events are defined as half-sines
        event_frequency = 1000 / (width * 2)

        # event morph based on a half sine with given event width and sampling frequency
        template = np.sin(2 * np.pi * event_idx / 1000 * event_frequency)
        template = template / np.sum(template**2)  # Weight normalized
        return template

@dataclass
class Arbitrary():
    sfreq: float
    width_samples: int
    location: int
    template: Any

    @classmethod
    def create_expected(cls, sfreq, template, width=50,  location=None):

        steps = 1000 / sfreq
        width_samples = len(template)

        if location is None:
            location = width_samples
        else:
            location = int(np.rint(location))
        return cls(sfreq, width_samples, location, template)
