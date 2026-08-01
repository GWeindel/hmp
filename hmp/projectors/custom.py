"""Apply given weights for custom linear combination."""
from xarray import DataArray

from hmp.projectors.base import Projector


class Custom(Projector):
    """Project data using custom weights.

    Attributes
    ----------
    weights : xr.DataArray
        DataArray with dimensions [channel, component] and coordinates
        channel (Fp1, CPz, ..) and component (0, 1, ..).
        Default = None

    """

    weights: DataArray

    def __init__(self, weights):
        self.weights = weights

    def fit(self,
            _data: DataArray,
            verbose: bool = True):
        """Pass weights."""
        if verbose is True:
            print("Using weights at initialization for "
                  "custom projection on the data")

    def transform(self,
                  data: DataArray,) -> DataArray:
        data = data @ self.weights.astype(data.dtype)
        return data
