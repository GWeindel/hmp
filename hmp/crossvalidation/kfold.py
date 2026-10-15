"""Functions to perform several crossvalidation methods on different model fits."""

import numpy as np

from hmp.patterndata import PatternData
from hmp.utils import _define_random_state

def pseudo_cv_split(trial_data,
             test_frac: float = 1,
             seed: int | None = None):
    """Split the PatternData object into train and test sets.

    Parameters
    ----------
    trial_data : PatternData
        The trial data to perform the split on.
    test_frac: float
        Fraction of the data to use for the test of the model
    seed: int
        Seed to use for random number generation

    Returns
    -------
    train_set: PatternData
        The trial data used for training the model
    test_set: PatternData
        The trial data used for testing the model

    Notes
    -----
    This is not a proper train/test split as the train and test data share
    the same projection from hmp.projectors
    """
    random_state = _define_random_state(seed=seed)
    indices = np.arange(len(trial_data.durations))

    # Test set
    test_size = int(np.ceil(test_frac*len(indices)))
    test_rand_idx = random_state.choice(np.arange(indices), size=test_size, replace=False)
    test_trial_data = _recut_data(test_rand_idx, trial_data)

    # Train set
    train_rand_idx = np.setdiff1d(indices,  test_rand_idx)
    train_trial_data = _recut_data(train_rand_idx, trial_data)

    return train_trial_data, test_trial_data

def pseudo_kfold(trial_data: PatternData,
                 n_splits: int = 5,
                 seed: int = 1):
    """Create kfold partition of trial data.

    By default the data is shuffled with a seed to avoid temporal
    dependencies in the partitioning. The default seed of 1 is used to
    ensure reproducibility.

    Parameters
    ----------
    trial_data : PatternData
        The trial data to perform the split on.
    n_splits: float
        Number of split to perform on the data
    seed: int
        Seed to use for random number generation

    Returns
    -------
    train_set: PatternData
        The trial data used for training the model
    test_set: PatternData
        The trial data used for testing the model

    Notes
    -----
    This is not a proper train/test split as the train and test data share
    the same projection from hmp.projectors.
    """
    n_trials = len(trial_data.durations)

    indices = np.arange(n_trials)

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
        train_td = _recut_data(train_idx, trial_data)
        test_td = _recut_data(test_idx, trial_data)
        yield train_td, test_td
