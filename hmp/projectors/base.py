"""Classes for projecting data into new component space.

Classes
-------
PCA
    Project channels into principal component space based on the covariance matrix among electrodes
Custom
    Apply a user-defined linear combination of original channels to a new set of virtual channels
Identity
    Returns the channels in the same space
"""

from abc import ABC, abstractmethod


class Projector(ABC):
    """Base class projection.

    Any class should contain fit and transform method
    """

    @abstractmethod
    def fit(self):
        ...

    @abstractmethod
    def transform(self):
        ...

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
