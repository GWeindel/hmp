"""Return original data."""

import numpy as np
from xarray import DataArray

from hmp.projectors.base import Projector


class Identity(Projector):
    """Returns identity projection.

    Attributes
    ----------
    weights : xr.DataArray
        DataArray with dimensions [channel, component] and coordinates
        channel (Fp1, CPz, ..) and component (0, 1, ..).
        Default = None

    """

    weights: DataArray

    def __init__(self):
        self.weights = None

    def fit(self,
            data: DataArray,
            verbose: bool = True):
        """Create weights as identity matrix."""
        if verbose is True:
            print('No projection applied, returning Identity')
        self.weights = DataArray(
                    np.identity(len(data.channel)),
                    dims=("channel", "component"),
                    coords={"channel": data.channel,
                    "component": np.arange(len(data.channel))})

    def transform(self,
                  data: DataArray,) -> DataArray:
        data = data @ self.weights.astype(data.dtype)
        return data
