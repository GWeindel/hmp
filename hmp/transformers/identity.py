"""Transforms epoched data using a the ProjIdentity matrix for HMP analysis.

Returns electrode values after performing transformation steps without the projection.
For consistency the channel dimension is renamed 'component'
"""

from typing import Optional
from warnings import warn

import numpy as np
import xarray as xr

from hmp.transformers.base import BaseTransformer


class ProjIdentity(BaseTransformer):
    """Transforms epoched data using a the ProjIdentity matrix for HMP analysis.

    Returns electrode values after performing transformation steps without the projection.
    For consistency the channel dimension is renamed 'component'

    Parameters
    ----------
    epoch_data : xr.Dataset
        Input EEG data with dimensions [participant, epoch, sample, channel], from `io` module
    interval_id: str
        Name of the variable that contains the trial intervals in the epoch_data used for cropping.
    offset_end : float
        Time offset after interval end for cropping.
    offset_start : float
        Time offset before interval start for cropping. Negative number extends epoch before start.
    min_duration : float, optional
        Minimum duration threshold for keeping epochs.
    max_duration : float, optional
        Maximum duration threshold for keeping epochs.
    reject_threshold : float, optional
        Threshold for rejecting noisy epochs.
    common_variance : bool
        Whether to standardize variance across trials.
    subject_zscore: bool
        Z-score each component for each participant
    whiten : bool
        Return the components with unit-variance
    center : bool
        Whether to center the data across the last dimension before projection
    verbose : bool
        Whether to print rejection/cropping details.
    """

    def __init__(#noqa: PLR0913
        self,
        epoch_data: xr.Dataset,
        interval_id: str = 'rt',
        offset_end: float = 0,
        offset_start: float = 0,
        min_duration: float = 0,
        max_duration: float = float('Inf'),
        reject_threshold: Optional[float] = None,
        center: bool = False,
        whiten: bool = True,
        common_variance: bool = False,
        subject_zscore: bool = False,
        verbose: bool = True,
    ):
        super().__init__(
            interval_id=interval_id,
            offset_end=offset_end,
            offset_start=offset_start,
            min_duration=min_duration,
            max_duration=max_duration,
            reject_threshold=reject_threshold,
            verbose=verbose,
            common_variance=common_variance,
            subject_zscore=subject_zscore,
            whiten=whiten,
            center=center,
        )
        warn('Identity projection might pose problems of dimensionality'
             ' and collinearity of channels. Thus rendering HMP estimation'
             ' difficult, use with care!')

        # Preprocessing
        data = self.common_preprocess(epoch_data)

        # Projection
        weights = xr.DataArray(
            np.identity(len(epoch_data.channel)),
            dims=("channel", "component"),
            coords={"channel": epoch_data.channel, "component": np.arange(len(epoch_data.channel))}
        )
        # Final formatting
        self.data_format(
            data, weights
        )
