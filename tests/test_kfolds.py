from hmp.models import EventModel
from hmp import basedata, projectors, patterndata
from hmp.crossvalidation.pseudo_kfold import KFoldCV, get_folds

import numpy as np

from test_io import init_data

_, _, epoch_data, info, sfreq, n_events = init_data()
base_data = basedata.default(epoch_data, n_comp=2)
pattern_data = patterndata.PatternData.from_basedata(base_data)
model = EventModel(n_events=3)

def test_kfold_fit():
    cv = KFoldCV(4)

    cv.fit(model, pattern_data, cpus_cv=1)

    assert len(cv.folds) == 4
    assert len(cv.train_models) == 4
    assert len(cv.test_loglikelihood) == 4

def test_kfold_partition_complete():
    cv = KFoldCV(4)
    cv.fit(model, pattern_data)
    folds = list(cv.folds)

    test_trials = np.concatenate([
        test_pd.durations.trial.values
        for _, test_pd in folds
    ])

    original_trials = np.sort(pattern_data.durations.trial.values)

    np.testing.assert_array_equal(
        np.sort(test_trials),
        original_trials,
    )

def test_fold_train_test_disjoint():
    for train_pd, test_pd in get_folds(pattern_data, n_splits=4):
        train_trials = set(train_pd.durations.trial.values)
        test_trials = set(test_pd.durations.trial.values)

        assert train_trials.isdisjoint(test_trials)

def test_fold_trial_count_preserved():
    n_trials = len(pattern_data.durations)

    for train_pd, test_pd in get_folds(pattern_data, n_splits=4):
        assert (
            len(train_pd.durations)
            + len(test_pd.durations)
            == n_trials
        )

def test_kfold_seed_reproducible():
    folds1 = list(get_folds(pattern_data, 4, True, 1))
    folds2 = list(get_folds(pattern_data, 4, True, 1))

    for (_, test1), (_, test2) in zip(folds1, folds2):
        np.testing.assert_array_equal(
            test1.durations.trial.values,
            test2.durations.trial.values,
        )

def test_kfold_mp_equivalent():
    model = EventModel(n_events=3)

    cv_serial = KFoldCV(4, seed=1)
    cv_serial.fit(model, pattern_data,
                  cpus_cv=1,
                  cpus_model=1)

    cv_parallel = KFoldCV(4, seed=1)
    cv_parallel.fit(model, pattern_data,
                    cpus_cv=2,
                    cpus_model=1)

    np.testing.assert_allclose(
        cv_serial.test_loglikelihood,
        cv_parallel.test_loglikelihood,
    )