"""Applying a PCA on data."""

from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from xarray import DataArray

from hmp.projectors.base import Projector


class PCA(Projector):
    """Project data into PC space using the covariance matrix.

    Attributes
    ----------
    method_pca: str, optional
        Perform PCA ('pca') or SVD decomposition ('svd') for PCA
        Default = 'svd'
    n_comp: int, optional
        Nr of components retained if > 1, otherwise (0 < n_comp < 1) nr of components
        explaining at least n_comp% variance are retained.  If None, user input requested.
        Default = None
    weights : DataArray
        DataArray with dimensions [channel, component] and coordinates
        channel (Fp1, CPz, ..) and component (0, 1, ..).
        Default = None

    """

    def __init__(self, method_pca="svd", n_comp=None, verbose: bool = True):
        self.method_pca = method_pca
        self.n_comp = n_comp
        self.weights = None
        self.verbose = verbose

    def fit(self,
            data: DataArray):
        """Estimate PCA weights."""
        #Compute covariance
        recordings = set(data.recording.values)
        group_cov = np.zeros((len(recordings), data.sizes["channel"],
                              data.sizes["channel"]), dtype=np.float64)
        for j, recording in enumerate(recordings):
            part_data = data.where(data.recording == recording, drop=True)
            group_cov[j] = self._compute_covariance(part_data)
        vcov_mat = np.mean(group_cov, axis=0)

        #pca
        pca_weights, eigenvalues = self._pca(vcov_mat, data.sizes["channel"],
                                            data.coords["channel"].values)
        explained_variance_ratio = eigenvalues/np.sum(eigenvalues)

        #select components
        if self.n_comp is None:
            self.n_comp = self._user_input_n_comp(explained_variance_ratio)
        elif self.n_comp < 1: #return nr of components based on explained variance
            self.n_comp = np.where(np.cumsum(explained_variance_ratio) >= self.n_comp)[0][0] + 1
            if self.verbose:
                print(f"{self.n_comp} components retained, explaining "
                      f"{np.cumsum(explained_variance_ratio)[self.n_comp]:.5f}% variance")
        self.weights = pca_weights[:,:self.n_comp]

    def transform(self,
                  data: DataArray,) -> DataArray:
        data = data @ self.weights.astype(data.dtype)
        return data

    def _user_input_n_comp(self, explained_variance_ratio) -> int:
        """Request user input on number of retained PCA components."""
        _, ax = plt.subplots(1, 2, figsize=(0.2 * len(explained_variance_ratio), 4))
        ax[0].plot(explained_variance_ratio, ".-")
        ax[0].set_ylabel("Normalized explained variance")
        ax[0].set_xlabel("Component")
        ax[1].plot(np.cumsum(explained_variance_ratio), ".-")
        ax[1].set_ylabel("Cumulative normalized explained variance")
        ax[1].set_xlabel("Component")
        plt.tight_layout()
        plt.show()

        n_comp = int(
            input(
                f"How many PCs would you like to retain?"
                f"95 and 99% explained variance at components "
                f"{np.where(np.cumsum(explained_variance_ratio) >= 0.95)[0][0] + 1} and "
                f"{np.where(np.cumsum(explained_variance_ratio) >= 0.99)[0][0] + 1}; "
                f"components till {np.where(explained_variance_ratio >= 0.01)[0][-1] + 1} "
                f"explain at least 1% variance per component)?"
            )
        )
        return n_comp

    def _pca(self, vcov_mat: DataArray, n_comp: int, channel: DataArray):
        """
        Calculate PCA.

        vcov_mat : DataArray
            data to perform PCA on.
        n_comp : int
            nr of components to return
        channel: DataArray
            channel names
        """
        if self.method_pca == 'pca':
            eigvals, eigvecs = eigh(vcov_mat)
            ix = np.argsort(np.abs(eigvals))[::-1]
            evecs = eigvecs[:, ix]
            evecs = evecs[:, :n_comp]
            eigvals = eigvals[ix]
        elif self.method_pca == 'svd':
            _, s, evecs = np.linalg.svd(vcov_mat, full_matrices=False)
            eigvals = (s**2) / (vcov_mat.shape[0] - 1)
            evecs = evecs.T[:, :n_comp]

        # Rebuilding as xarray to ease computation
        coords = dict(channel=("channel", channel), component=("component", np.arange(n_comp)))
        pca_weights = DataArray(evecs, dims=("channel", "component"), coords=coords)
        return pca_weights, eigvals

    @staticmethod
    def _compute_covariance(data):
        """Compute covariance of data trial by trial."""
        vcov_mat = np.zeros((data.sizes["channel"], data.sizes["channel"]), dtype=np.float64)
        # Iteratively for memory efficiency
        count = 0
        for i in data.trial:
            x_i = np.squeeze(data.sel(trial=i).values)
            x_i = x_i[:, ~np.isnan(x_i[0, :])]
            if x_i.shape[1] > x_i.shape[0]:
                count += 1
                cov_i = (x_i @ x_i.T) / (x_i.shape[1]-1)
                # Regularization using MNE python's default
                sigma = np.mean(np.diag(cov_i))
                cov_i.flat[:: len(cov_i) + 1] += 0.1 * sigma
                vcov_mat += cov_i
        if count < len(data.trial)/10:
            warn(f"Less than 10% of the trials used to compute covariance for"
                 f"{np.unique(data.recording.values)}. Covariance matrix might be unreliable")
        return vcov_mat/count

