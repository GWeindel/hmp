"""Adapted from https://github.com/ANCPLabOldenburg/MCCA/blob/main/README.md.

Author/maintainer: Leo Michalke https://github.com/lmichalke


MIT License

Copyright (c) 2022 Applied Neurocognitive Psychology Lab University Oldenburg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import warnings

import numpy as np
from scipy.linalg import eigh, norm
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError


class MCCA:
    """Performs multiset canonical correlation analysis.

    with an optional regularization term based on spatial similarity of weight maps. The
    stronger the regularization, the more similar weight maps are forced to
    be across subjects. Note that the term 'weights' is used interchangeably
    with PCA / MCCA eigenvectors here.

    Parameters
    ----------
        n_components_pca (int): Number of PCA components to retain for each subject (default 50)

        n_components_mcca (int): Number of MCCA components to retain (default 10)

        cov (bool): If true (default), apply the PCA to the var-cov matrix

        r (int or float): Regularization strength (default 0)

        pca_only (bool): If true, skip MCCA calculation (default False)

    Attributes
    ----------
        mu (ndarray): Mean subtracted before PCA (subjects, sensors)

        sigma (ndarray): PCA standard deviation (subjects, PCs)

        pca_weights (ndarray): PCA weights that transform sensors to PCAs for each
                               subject (subjects, sensors, PCs)

        mcca_weights (ndarray): MCCA weights that transform PCAs to MCCAs for each subject.
                                None if pca_only is True. (subjects, PCs, CCs)
    """

    def __init__(self, n_components_pca, n_components_mcca, r=0, pca_only=False):
        if n_components_mcca > n_components_pca:
            import warnings

            warnings.warn(
                f"Warning........... number of MCCA components ({n_components_mcca}) cannot be "
                "greater than "
                f"number of PCA components ({n_components_pca}), setting them equal."
            )
            n_components_mcca = n_components_pca
        self.n_pcs = n_components_pca
        self.n_ccs = n_components_mcca
        self.r = r
        self.pca_only = pca_only
        self.mcca_weights, self.pca_weights, self.mu, self.sigma = None, None, None, None

    def obtain_mcca(self, X):
        """Apply individual-subject PCA and across-subjects MCCA.

        Parameters
        ----------
            X (ndarray): Input data in sensor space (subjects, samples, sensors)

        Returns
        -------
            scores (ndarray): Returns scores in PCA space if self.pca_only is true
            and MCCA scores otherwise.
        """
        n_subjects, n_samples, n_sensors = X.shape
        X_pca = np.zeros((n_subjects, n_samples, self.n_pcs))
        self.pca_weights = np.zeros((n_subjects, n_sensors, self.n_pcs))
        self.mu = np.zeros((n_subjects, n_sensors))
        self.sigma = np.zeros((n_subjects, self.n_pcs))
        lim = 0
        # obtain subject-specific PCAs
        for i in range(n_subjects):
            pca = PCA(n_components=self.n_pcs, svd_solver="full", copy=False)
            x_i = np.squeeze(X[i])  # time x sensors
            score = pca.fit_transform(x_i[~np.isnan(x_i[:, 0]), :])
            self.mu[i] = pca.mean_
            self.sigma[i] = np.sqrt(pca.explained_variance_)
            score /= self.sigma[i]
            lim_i = len(x_i[~np.isnan(x_i[:, 0])])
            lim = int(np.max([lim, lim_i]))
            self.pca_weights[i] = pca.components_.T
            X_pca[i, :lim_i, :] = score
        warnings.warn(f"MCCA is done on {lim} samples per subject")
        X_pca = X_pca[:, :lim, :]

        if self.pca_only:
            return X_pca
        else:
            return self._mcca(X_pca)

    def obtain_mcca_cov(self, X):
        """Apply individual-subject PCA on the variance-covariance matrix and across-subjects MCCA.

        Parameters
        ----------
        X (ndarray):
            Input data in sensor space for each trial (subjects, n_trials, samples, sensors)

        Returns
        -------
        scores (ndarray):
            Returns scores in PCA space if self.pca_only is true and MCCA scores otherwise.
        """
        n_subjects, n_trials, n_samples, n_sensors = X.shape
        X_pca = np.zeros((n_subjects, n_trials * n_samples, self.n_pcs))
        self.pca_weights = np.zeros((n_subjects, n_sensors, self.n_pcs))
        lim = 0
        self.mu = np.zeros((n_subjects, n_sensors))
        self.sigma = np.ones((n_subjects, self.n_pcs))
        # obtain subject-specific PCAs
        for i in range(n_subjects):
            pca = PCA(n_components=self.n_pcs, svd_solver="full", copy=False)
            x_i = np.squeeze(X[i])  # time x sensors
            av_vcov = np.mean(
                [
                    np.cov(x_i[trial, ~np.isnan(x_i[trial, :, 0]), :].T)
                    for trial in range(x_i.shape[0])
                    if ~np.isnan(x_i[trial, :, 0]).all()
                ],
                axis=0,
            )
            pca.fit(av_vcov)
            x_i = x_i.reshape(n_trials * n_samples, n_sensors)
            score = x_i[~np.isnan(x_i[:, 0]), :] @ pca.components_.T
            lim_i = len(x_i[~np.isnan(x_i[:, 0])])
            lim = int(np.max([lim, lim_i]))
            self.pca_weights[i] = pca.components_.T
            X_pca[i, :lim_i, :] = score
        warnings.warn(f"MCCA is done on {lim} out of {n_trials * n_samples} samples per subject")
        X_pca = X_pca[:, :lim, :]

        if self.pca_only:
            return X_pca
        else:
            return self._mcca(X_pca)

    def _mcca(self, pca_scores):
        """Perform multiset canonical correlation analysis with an optional.

        regularization term based on spatial similarity of weight maps. The
        stronger the regularization, the more similar weight maps are forced to
        be across subjects.

        Parameters
        ----------
            pca_scores (ndarray): Input data in PCA space (subjects, samples, PCs)

        Returns
        -------
            mcca_scores (ndarray): Input data in MCCA space (subjects, samples, CCs).
        """
        # R_kl is a block matrix containing all cross-covariances R_kl = X_k^T X_l between
        # subjects k, l, k != l
        # where X is the data in the subject-specific PCA space (PCA scores)
        # R_kk is a block diagonal matrix containing auto-correlations R_kk = X_k^T X_k in its
        # diagonal blocks
        R_kl, R_kk = _compute_cross_covariance(pca_scores)
        # Regularization
        if self.r != 0:
            # The regularization terms W_kl and W_kk are calculated the same way as
            # R_kl and R_kk above, but using
            # cross-covariance of PCA weights instead of PCA scores
            W_kl, W_kk = _compute_cross_covariance(self.pca_weights)
            # Add regularization term to R_kl and R_kk
            R_kl += self.r * W_kl
            R_kk += self.r * W_kk
        # Obtain MCCA solution by solving the generalized eigenvalue problem
        #                   R_kl h = p R_kk h
        # where h are the concatenated eigenvectors of all subjects and
        # p are the generalized eigenvalues (canonical correlations).
        # If PCA scores are whitened and no regularisation is used, R_kk is an identity matrix and
        # the generalized
        # eigenvalue problem is reduced to a regular eigenvalue problem
        p, h = eigh(R_kl, R_kk, subset_by_index=(R_kl.shape[0] - self.n_ccs, R_kl.shape[0] - 1))
        # eigh returns eigenvalues in ascending order. To pick the k largest from a
        # total of n eigenvalues,
        # we use subset_by_index=(n - k, n - 1).
        # Flip eigenvectors so that they are in descending order
        h = np.flip(h, axis=1)
        # Reshape h from (subjects * PCs, CCs) to (subjects, PCs, CCs)
        h = h.reshape((pca_scores.shape[0], self.n_pcs, self.n_ccs))
        # Normalize eigenvectors per subject
        self.mcca_weights = h / norm(h, ord=2, axis=(1, 2), keepdims=True)
        return np.matmul(pca_scores, self.mcca_weights)

    def transform_trials(self, X, subject=0):
        """Transform single trial data to MCCA space.

        Use of MCCA weights (obtained from averaged data) to transform single
        trial data from sensor space to MCCA space.

        Parameters
        ----------
            X (ndarray): Single trial data of one subject in sensor space
                         (trials, samples, sensors)
            subject (int): Index of the subject whose data is being transformed

        Returns
        -------
            X_mcca (ndarray): Transformed single trial data in MCCA space
                            (trials, samples, CCs)
        """
        if self.mcca_weights is None:
            raise NotFittedError("MCCA needs to be fitted before calling transform_trials")
        X -= self.mu[np.newaxis, np.newaxis, subject]  # centered
        X_pca = np.matmul(X, self.pca_weights[subject])
        X_pca /= self.sigma[np.newaxis, np.newaxis, subject]  # standardized
        X_mcca = np.matmul(X_pca, self.mcca_weights[subject])
        return X_mcca


def _compute_cross_covariance(X):
    """Compute cross-covariance of PCA scores or components between subjects.

    Parameters
    ----------
        X (ndarray): PCA scores (subjects, samples, PCs) or weights (subjects, sensors, PCs)

    Returns
    -------
    R_kl (ndarray):
        Block matrix containing all cross-covariances R_kl = X_k^T X_l between
        subjects k, l, k != l
        with shape (subjects * PCs, subjects * PCs)
    R_kk (ndarray):
        Block diagonal matrix containing auto-correlations R_kk = X_k^T X_k in its
        diagonal blocks
        with shape (subjects * PCs, subjects * PCs)
    """
    n_subjects, n_samples, n_pcs = X.shape
    R = np.cov(X.swapaxes(1, 2).reshape(n_subjects * n_pcs, n_samples))
    R_kk = R * np.kron(np.eye(n_subjects), np.ones((n_pcs, n_pcs)))
    R_kl = R - R_kk
    return R_kl, R_kk
