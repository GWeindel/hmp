""" Kfold crossvalidation class.

Class
-------
KFoldCV

"""

import numpy as np
import xarray as xr
from copy import deepcopy
import multiprocessing as mp

from hmp.patterndata import PatternData, select_indices
from hmp.utils import _define_random_state


class KFoldCV:
    """
    K-fold cross-validation for HMP EventModels.
    
    The class partitions PatternData into training and testing folds,
    fits an independent model on each training fold and stores
    out-of-sample log-likelihoods together with the fitted models.
    
    Attributes
    ----------
    llk : np.ndarray
        Test log-likelihood for each fold.
    train_models : tuple
        Models fitted on each training fold.
    folds : list[tuple[PatternData, PatternData]]
        Train/test splits used during cross-validation.

    Notes
    -----
    This is not a proper train/test split as the train and test data share
    the same projection from hmp.projectors
    """
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        seed: int = 1
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed
    
    def prepare(self, pattern_data):
        self.folds = list(
            get_folds(
                pattern_data,
                self.n_splits,
                self.shuffle,
                self.seed,
            )
        )

    def fit(self,
            model,
            pattern_data=None,
            channel_pars=None,
            time_pars=None,
            cpus_cv = 1,
            cpus_model = 1):

        """Fit a model independently on each cross-validation fold.
    
        The provided PatternData object is partitioned into training and
        testing folds according to the class settings. For each fold, a copy
        of the model is fitted on the training data and evaluated on the
        corresponding test data. The fitted models, fold definitions, and
        test log-likelihoods are stored as attributes of the instance.
    
        Parameters
        ----------
        model : EventModel
            HMP model to fit on each training fold.
        pattern_data : PatternData
            Data used to generate the cross-validation folds.
        cpus_cv : int
            Number of CPUs used to parallelize cross-validation across folds.
            If equal to 1, folds are processed sequentially.
        cpus_model : int
            Number of CPUs used during model fitting within each fold.
    
        Attributes
        ----------
        folds : list[tuple[PatternData, PatternData]]
            Training and testing datasets for each fold.
        train_models : tuple
            Models fitted on each training fold.
        test_loglikelihood : np.ndarray
            Test log-likelihood for each fold.
    
        Returns
        -------
        test_loglikelihood : np.ndarray
            Loglikelihood on the test folds
        train_models : tuple
            Model fitted on the train folds
        """
        self.base_model = model
        if not hasattr(self, "folds"):
            if pattern_data is None:
                raise ValueError(
                    "pattern_data must be provided if folds have not been prepared."
                )
            self.prepare(pattern_data)

        if cpus_cv == 1:
            results = [
                _per_fold(model, train_pd, test_pd, channel_pars, time_pars, cpus_model)
                for train_pd, test_pd in self.folds
            ]
        else:
            with mp.Pool(cpus_cv) as pool:
                results = pool.starmap(_per_fold,
                    [
                        (model, train_pd, test_pd, channel_pars, time_pars, cpus_model)
                        for train_pd, test_pd in self.folds
                    ],
                )
        llk, train_models = zip(*results)
        self.train_models = train_models
        self.test_loglikelihood = np.array(llk)
        return self.test_loglikelihood, self.train_models
    
    def get_eventprobs(self, source='train'):
        """Compute event probabilities for each cross-validation fold.
        
        The fitted model from each fold is applied either to the training
        data used during fitting or to the corresponding held-out test data.
        Event probabilities are concatenated along a new ``fold`` dimension.
        
        Parameters
        ----------
        source : {"train", "test"}
            Dataset on which to compute event probabilities.
        
            - ``"train"``: compute probabilities on the training data of
              each fold.
            - ``"test"``: compute probabilities on the held-out test data
              of each fold.
        
        Returns
        -------
        xr.DataArray
            Event probabilities with an additional ``fold`` dimension
            identifying the fitted model used to compute them.

        """
        if source not in {"train", "test"}:
            raise ValueError(
                "source must be either 'train' or 'test'"
            )
        if not hasattr(self, "train_models"):
            raise RuntimeError(
                "KFoldCV must be fitted before calling get_eventprobs()."
            )
        evprobs = xr.concat(
            [
                model.transform(
                    train_pd if source == "train" else test_pd
                )[1]
                for model, (train_pd, test_pd)
                in zip(self.train_models, self.folds)
            ],
            dim=xr.DataArray(
                np.arange(len(self.train_models)),
                dims="fold",
                name="fold",
            ),
            join='outer'
        ).sortby('trial')
    
        return evprobs



def get_folds(pattern_data: PatternData,
                 n_splits: int = 5,
                 shuffle: bool = True,
                 seed: int = 1):
    """Create kfold partition of trial data.

    By default the data is shuffled with a seed to avoid temporal
    dependencies in the partitioning. The default seed of 1 is used to
    ensure reproducibility.

    Parameters
    ----------
    pattern_data : PatternData
        The trial data to perform the split on.
    n_splits: int
        Number of split to perform on the data
    shuffle: bool
        If true shuffles the data before split
    seed: int
        Seed to use for shuffling

    Returns
    -------
    train_set: PatternData
        The trial data used for training the model
    test_set: PatternData
        The trial data used for testing the model
    """
    n_trials = len(pattern_data.durations)

    indices = np.arange(n_trials)
    
    if shuffle:
        random_state = _define_random_state(seed)
        random_state.shuffle(indices)

    fold_sizes = np.full(n_splits, n_trials // n_splits, dtype=int)
    fold_sizes[: n_trials % n_splits] += 1

    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        current = stop
        train_pd = select_indices(train_idx, pattern_data)
        test_pd = select_indices(test_idx, pattern_data)
        yield train_pd, test_pd
    

def _per_fold(model, train_pd, test_pd, channel_pars, time_pars, cpus):
    train_model = deepcopy(model)
    train_model.fit(
        train_pd,
        channel_pars=deepcopy(channel_pars),
        time_pars=deepcopy(time_pars),
        verbose=False,
        cpus=cpus
    )
    test_llk = train_model.transform(test_pd)[0]
    return test_llk, train_model