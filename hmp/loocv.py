"""Module for leave-one-out crossvalidation evaluation."""

import itertools
import multiprocessing as mp

import numpy as np
import xarray as xr

import hmp


def loocv_calcs(data, init, participant, initial_fit, cpus=None, verbose=False):
    """Fit the model and compute likelihood of left out data.

    Fits model based on init settings and initial_fit parameters to data of
    n - 1 (participant) participants, and calculates likelihood on the left-out
    participant.

    Parameters
    ----------
    data : xarray.Dataset
        xarray data from transform_data()
    init : hmp object
        original hmp object used for the fit, all settings are copied to the left out models
    participant : str
        name of the participant to leave out and estimate likelihood on
    initial_fit : xarray.Dataset
        Fit of the model with the same number of events and all participants
    cpus : int
        Number of cpus to use to fit the models.
    verbose : bool

    Returns
    -------
    likelihood : float
        likelihood computed for the left-out participant
    """
    if verbose:
        print(f"\t\tCalculating fit for participant {participant}")
    if cpus is None:
        cpus = init.cpus

    participants_idx = data.participant.values

    # Extracting data with and without left out participant
    data_without_pp = hmp.utils.stack_data(
        data.sel(participant=participants_idx[participants_idx != participant], drop=False)
    )
    data_pp = hmp.utils.stack_data(data.sel(participant=participant, drop=False))

    # Building models
    model_without_pp = hmp.models.HMP(
        data_without_pp,
        sfreq=init.sfreq,
        event_width=init.event_width,
        cpus=cpus,
        shape=init.shape,
        template=init.template,
        location=init.location,
        distribution=init.distribution,
    )

    model_pp = hmp.models.HMP(
        data_pp,
        sfreq=init.sfreq,
        event_width=init.event_width,
        cpus=cpus,
        shape=init.shape,
        template=init.template,
        location=init.location,
        distribution=init.distribution,
    )

    # fit the HMP using previously estimated parameters as initial parameters,
    # and estimate likelihood
    dur_ratio = model_pp.mean_d / model_without_pp.mean_d
    if "level" in initial_fit.dims:
        n_eve = initial_fit.pars_map.shape[1] - 1
        locations = np.zeros(initial_fit.pars_map.shape, dtype=int)
        for c in range(len(initial_fit.pars_map)):
            indexes = np.arange(len(initial_fit.pars_map[c]))[
                initial_fit.pars_map[c, :] >= 0
            ]  # Actual estimated event
            if len(indexes[1:-1]) > 0:
                locations[c, indexes[1:-1]] = (
                    init.location
                )  # add location at every non first or last estimated event
        fit_without_pp = model_without_pp.fit_n(
            n_eve,
            magnitudes=initial_fit.magnitudes.values,
            parameters=initial_fit.parameters.values,
            mags_map=initial_fit.mags_map,
            pars_map=initial_fit.pars_map,
            level_dict=initial_fit.level_dict,
            verbose=False,
        )
        # calc lkh
        # adjust params to fit average duration of subject
        params = fit_without_pp.parameters.values
        params[:, :, 1] = params[:, :, 1] * dur_ratio
        levels_pp = initial_fit.sel(participant=participant)["levels"].values
        likelihood = model_pp._estim_probs_levels(
            fit_without_pp.magnitudes.values,
            params,
            locations,
            initial_fit.mags_map,
            initial_fit.pars_map,
            levels_pp,
            lkh_only=True,
        )
    else:
        n_eve = np.max(initial_fit.event.values) + 1
        locations = np.zeros(n_eve + 1, dtype=int)
        locations[1:-1] = init.location
        # fit model
        fit_without_pp = model_without_pp.fit_n(
            n_eve,
            initial_fit.magnitudes.dropna("event", how="all").values,
            initial_fit.parameters.dropna("stage").values,
            verbose=False,
        )
        # calc lkh
        # adjust params to fit average duration of subject
        params = fit_without_pp.parameters.dropna("stage").values
        params[:, 1] = params[:, 1] * dur_ratio
        likelihood = model_pp.estim_probs(
            fit_without_pp.magnitudes.dropna("event", how="all").values,
            params,
            locations,
            n_eve,
            None,
            True,
        )

    return likelihood


def loocv(init, data, estimate, cpus=1, verbose=True, print_warning=True):
    """Perform leave-one-out cross validation.

    Performs leave-one-out cross validation. For provided estimate(s), it will perform loocv by
    leaving out one participant, estimating a fit, and computing the likelihood of the data from
    the left out participant with the estimated parameters. This is repeated for all participants.

    Initial parameters for the models are based on estimate(s), hmp model settings on init.

    Estimate(s) can be provides as:
    - a single model estimate (from fit_n(..))
    - a set of fits from backward estimation (from backward_estimation(..))
    - a model fit with different levels
    - a list of one or more of the above
    Note that all need to share the same data and participants.

    IMPORTANT:  This loocv procedure is incorrect in the sense that an initial estimate is used
                to inform both the fit of the left-out participant and the other participants.
                This means that they are not fully independent, unless the initial estimate is
                based on the literature or another task. However, it does give a very good initial
                idea of the correct loocv procedure, and is relatively quick.

                To do this correctly, use loocv_backward, loocv_fit, or the general loocv_fun,
                which all three also calculate the initial estimate for every fold by applying
                backward estimation, the fit function, or your own function, respectively.

    Parameters
    ----------
    init : hmp model
        initialized hmp model
    data : xarray.Dataset
        xarray data from transform_data()
    estimate : hmp model estimate or list of estimates
        See above.
    cpus : int
        Nr of cpus to use. If 1, LOOCV is performed on a single CPU. Otherwise
        on the provided int or setting in init.
        We recommend using 1 CPU at this level on a laptop or normal PC. Only use multiple
        CPUs if you have *a lot* of memory available.
    verbose : bool
    print_warning : bool
        whether to plot the loocv 'incorrectness' warning

    Returns
    -------
    list of likelihood objects for provided model estimates
    """
    if verbose:
        if print_warning:
            print()
            print(
                "IMPORTANT:  This loocv procedure is incorrect in the sense that an "
                "initial estimate"
            )
            print(
                "is used to inform both the fit of the left-out participant and the other "
                "participants."
            )
            print("This means that they are not fully independent, unless the initial estimate is")
            print(
                "based on the literature or another task. However, it does give a very good initial"
            )
            print("idea of the correct loocv procedure and is relatively quick.")

            print("\nTo do loocv correctly, use loocv_backward or the general loocv_func,")
            print("which calculate the initial estimate for every fold by applying")
            print("backward estimation or your own function, respectively.")
            print()

    if cpus is None:
        cpus = init.cpus

    if cpus != 1:
        print("We recommend using cpus==1 unless you have *a lot* of memory and cpus available.")

    data = data.unstack()
    participants_idx = data.participant.values

    if not isinstance(estimate, list):
        models = [estimate]
    else:
        models = estimate

    n_models = len(models)
    if verbose:
        print(f"LOOCV started for {n_models} model(s)")
    # no mp here, but at participant level
    likelihoods = []
    for model in models:
        # option 1 and 2: single model and single model with levels
        if "n_events" not in model.dims:
            if verbose:
                if "level" in model.dims:
                    print(
                        f"\tLOOCV for multilevel model with "
                        f"{np.max(model.event).values + 1} event(s)"
                    )
                else:
                    print(
                        f"\tLOOCV for single model with {np.max(model.event).values + 1} event(s)"
                    )

            loocv = []
            if cpus == 1:  # not mp
                for participant in participants_idx:
                    loocv.append(loocv_calcs(data, init, participant, model, verbose=verbose))
            else:  # mp
                with mp.Pool(processes=cpus) as pool:
                    loocv = pool.starmap(
                        loocv_calcs,
                        zip(
                            itertools.repeat(data),
                            itertools.repeat(init),
                            participants_idx,
                            itertools.repeat(model),
                            itertools.repeat(1),
                            itertools.repeat(verbose),
                        ),
                    )

            likelihoods.append(
                xr.DataArray(
                    np.array(loocv).astype(np.float64),
                    dims="participant",
                    coords={"participant": participants_idx},
                    name="loo_loglikelihood",
                )
            )

        # option 3: backward
        if "n_events" in model.dims:
            if verbose:
                print(
                    f"\tLOOCV for backward estimation models with {model.n_events.values} event(s)"
                )

            loocv_back = []
            for n_eve in model.n_events.values:
                if verbose:
                    print(f"\t  Estimating backward estimation model with {n_eve} event(s)")
                loocv = []
                model_neve = model.sel(n_events=n_eve).dropna("event", how="all")
                if cpus == 1:  # not mp
                    for participant in participants_idx:
                        loocv.append(
                            loocv_calcs(data, init, participant, model_neve, verbose=verbose)
                        )
                else:  # mp
                    with mp.Pool(processes=cpus) as pool:
                        loocv = pool.starmap(
                            loocv_calcs,
                            zip(
                                itertools.repeat(data),
                                itertools.repeat(init),
                                participants_idx,
                                itertools.repeat(model_neve),
                                itertools.repeat(1),
                                itertools.repeat(verbose),
                            ),
                        )

                loocv_back.append(
                    xr.DataArray(
                        np.expand_dims(np.array(loocv).astype(np.float64), axis=0),
                        dims=("n_event", "participant"),
                        coords={"n_event": np.array([n_eve]), "participant": participants_idx},
                        name="loo_loglikelihood",
                    )
                )

            likelihoods.append(xr.concat(loocv_back, dim="n_event"))

    if n_models == 1:
        likelihoods = likelihoods[0]

    return likelihoods


def example_fit_n_func(hmp_model, n_events, magnitudes=None, parameters=None, verbose=False):
    """Fit function example.

    Example of simple function that can be used with loocv_func.
    This fits a model with n_events and potentially provided mags and params.
    Can be called, for example, as :
        loocv_func(hmp_model, hmp_data, example_fit_single_func, func_args=[2])
    """
    return hmp_model.fit_n(n_events, magnitudes=magnitudes, parameters=parameters, verbose=verbose)


def example_complex_fit_n_func(
    hmp_model, max_events=None, n_events=1, mags_map=None, pars_map=None, conds=None, verbose=False
):
    """Fit function, complex example.

    Example of a complex function that can be used with loocv_func.
    This function first performs backwards estimation up to max_events,
    and follows this with a condition-based model of n_events, informed
    by the selected backward model and the provided maps. It returns
    both models, so for both the likelihood will be estimated.
    Can be called, for example, as :
        pars_map = np.array([[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 2, 0],
                     [0, 0, 0, 0, 3, 0],
                     [0, 0, 0, 0, 4, 0]])
        conds = {'rep': np.arange(5)+1}
        loocv_func(hmp_model, hmp_data, example_complex_fit_n_func,
                   func_args=[7, 5, None, pars_map,conds])
    """
    # fit backward model up to max_events
    backward_model = hmp_model.backward_estimation(max_events)

    # select n_events model
    n_event_model = backward_model.sel(n_events=n_events).dropna("event", how="all")
    mags = n_event_model.magnitudes.dropna("event", how="all").data
    pars = n_event_model.parameters.dropna("stage").data

    # fit condition model
    cond_model = hmp_model.fit_n(
        magnitudes=mags,
        parameters=pars,
        mags_map=mags_map,
        pars_map=pars_map,
        level_dict=conds,
        verbose=verbose,
    )

    return [backward_model, cond_model]


def loocv_estimate_func(
    data, init, participant, func_estimate, func_args=None, cpus=None, verbose=False
):
    """Apply function estimation.

    Applies func_estimate with func_args to data of n - 1 (participant) participants.
    func_estimate should return an estimated hmp model; either a single model,
    a level model, or a backward estimation model. This model is then used
    to calculate the fit on the left out participant with loocv_loglikelihood.

    Parameters
    ----------
    data : xarray.Dataset
        xarray data from transform_data()
    init : hmp object
        original hmp object used for the fit, all settings are copied to the left out models
    participant : str
        name of the participant to leave out
    func_estimate : function that returns a hmp model estimate
        this can be backward_estimation, fit, or your own function.
        It should take an initialized hmp model as its first argument,
        other arguments are passed on from func_args.
        See also loocv_func(..)
    func_args : list
        List of arguments that need to be passed on to func_estimate.
        See also loocv_func(..)
    cpus : int
        number of cpus to use
    verbose : bool

    Returns
    -------
    hmp model
        estimated hmp_model with func_estimate on n-1 participants
    """
    if verbose:
        print(f"\tEstimating model for all participants except {participant}")
    if cpus is None:
        cpus = init.cpus

    participants_idx = data.participant.values

    # Extract data without left out participant
    data_without_pp = hmp.utils.stack_data(
        data.sel(participant=participants_idx[participants_idx != participant], drop=False)
    )

    # Building model
    model_without_pp = hmp.models.HMP(
        data_without_pp,
        sfreq=init.sfreq,
        event_width=init.event_width,
        cpus=cpus,
        shape=init.shape,
        template=init.template,
        location=init.location,
        distribution=init.distribution,
    )

    # Apply function and return
    estimates = func_estimate(model_without_pp, *func_args)
    if isinstance(estimates, list):
        for i in range(len(estimates)):
            estimates[i] = estimates[i].drop_vars(["eventprobs"])
    else:
        estimates = estimates.drop_vars(["eventprobs"])

    return estimates


def loocv_loglikelihood(data, init, participant, estimate, cpus=None, verbose=False):
    """Compute the log-likelihood of the fit.

    Calculate loglikelihood of fit on participant participant using parameters from estimate,
    either using single model or level based model.

    Parameters
    ----------
    data : xarray.Dataset
        xarray data from transform_data()
    init : hmp object
        original hmp object used for the fit, all settings are copied to the left out models
    participant : str
        name of the participant to leave out and estimate likelihood on
    estimate : xarray.Dataset
        estimate that has parameters to apply.
    cpus : int
        Number of cpus to use to fit the models.
    verbose : bool

    Returns
    -------
    likelihood : float
        likelihood computed for the left-out participant
    """
    if verbose:
        print(f"\tCalculating likelihood for participant {participant}")
    if cpus is None:
        cpus = init.cpus

    # Extracting data of left out participant
    data_pp = hmp.utils.stack_data(data.sel(participant=participant, drop=False))

    # Building model
    model_pp = hmp.models.HMP(
        data_pp,
        sfreq=init.sfreq,
        event_width=init.event_width,
        cpus=cpus,
        shape=init.shape,
        template=init.template,
        location=init.location,
        distribution=init.distribution,
    )

    # calc ratio average duration and subj duration
    nsubj = len(np.unique(data.participant.values))
    mean_d_without_subj = (init.mean_d * nsubj - model_pp.mean_d) / (nsubj - 1)
    dur_ratio = model_pp.mean_d / mean_d_without_subj
    locations = np.zeros(estimate.parameters.dropna("stage").values.shape[-2], dtype=int)
    locations[1:-1] = init.location
    if "level" in estimate.dims:
        locations = np.tile(locations, (estimate.parameters.shape[0], 1))  # Fix this
        from itertools import product

        # create levels for this participant based on estimate.levels_dict and model_pp
        # description of level for this participant, which is not available
        levels = estimate.levels_dict
        level_names = []
        level_levels = []
        level_trials = []
        for level in levels:
            level_names.append(list(level.keys())[0])
            level_levels.append(level[level_names[-1]])
            level_trials.append(model_pp.trial_coords[level_names[-1]].data.copy())

        level_levels = list(product(*level_levels))
        level_levels = np.array(level_levels, dtype=object)  # otherwise comparison below can fail

        # build level array with digit indicating the combined levels
        level_trials = np.vstack(level_trials).T
        levels = np.zeros((level_trials.shape[0])) * np.nan
        for i, level in enumerate(level_levels):
            levels[np.where((level_trials == level).all(axis=1))] = i
        levels = np.int8(levels)

        # adjust parameters based on average RT
        parameters = estimate.parameters.values
        parameters[:, :, 1] = parameters[:, :, 1] * dur_ratio

        likelihood = model_pp.estim_probs_levels(
            estimate.magnitudes.values,
            parameters,
            locations,
            estimate.mags_map,
            estimate.pars_map,
            levels,
            lkh_only=True,
        )
    else:
        n_eve = np.max(estimate.event.dropna("event").values) + 1
        print(n_eve)
        # adjust parameters based on average RT
        parameters = estimate.parameters.dropna("stage").values
        parameters[:, 1] = parameters[:, 1] * dur_ratio

        likelihood = model_pp.estim_probs(
            estimate.magnitudes.dropna("event", how="all").values,
            parameters,
            locations,
            n_eve,
            None,
            True,
        )

    return likelihood


def loocv_func(init, data, func_estimate, func_args=None, cpus=1, verbose=True):
    """Perform leave one-out cross validation.

    Performs leave-one-out cross validation using func_estimate to calculate the initial fit.
    It will perform loocv by leaving out one participant, applying 'func_estimate' to the
    data to estimate a fit, and computing the likelihood of the data from the left out
    participant with the estimated parameters. This is repeated for all participants. Hmp
    model settings are based on init.

    func_estimate is also allowed to return a list of estimates; for all provided estimates
    the likelihood of the left out participant will be calculated.

    For example of func_estimate, see these function above:
    example_fit_n_func(..)
    example_complex_fit_n_func(..)

    They can be called, for example, as
        loocv_func(hmp_model, hmp_data, example_fit_n_func, func_args=[1])

    Note that func_args is not named, so all arguments up to the one you want to use
    of func_estimate need to be provided.

    Parameters
    ----------
    init : hmp model
        initialized hmp model
    data : xarray.Dataset
        xarray data from transform_data()
    func_estimate : function that returns an hmp model estimate or a list
        of hmp model estimates. These can be the results of backward_estimation,
        fit_n, or your own function.
        It should take an initialized hmp model as its first argument,
        other arguments are passed on from func_args.
    func_args : list
        List of arguments that need to be passed on to func_estimate.
    cpus : int
        Nr of cpus to use. If 1, LOOCV is performed on a single CPU. Otherwise
        on the provided int or setting in init.
        We recommend using 1 CPU at this level on a laptop or normal PC. Only use multiple
        CPUs if you have *a lot* of memory available.
    verbose : bool

    Returns
    -------
    likelihood object containing likelihoods on left out participant
    estimates : list of all models without the left out participant
    """
    if cpus is None:
        cpus = init.cpus

    if cpus != 1:
        print("We recommend using cpus==1 unless you have *a lot* of memory and cpus available.")

    data = data.unstack()
    participants_idx = data.participant.values

    # first get estimates on n-1 subjects for all folds
    if verbose:
        print(f"Calculating estimates with func {func_estimate} and args {func_args}.")

    estimates = []
    if cpus == 1:  # not mp
        for participant in participants_idx:
            estimates.append(
                loocv_estimate_func(
                    data, init, participant, func_estimate, func_args=func_args, verbose=verbose
                )
            )
    else:  # mp
        with mp.Pool(processes=cpus) as pool:
            estimates = pool.starmap(
                loocv_estimate_func,
                zip(
                    itertools.repeat(data),
                    itertools.repeat(init),
                    participants_idx,
                    itertools.repeat(func_estimate),
                    itertools.repeat(func_args),
                    itertools.repeat(1),
                    itertools.repeat(verbose),
                ),
            )

    # if multiple estimates are returned per subject, rearrange data
    if isinstance(estimates[0], list):
        all_estimates = []
        for est_idx in range(len(estimates[0])):
            all_estimates.append([estimate[est_idx] for estimate in estimates])
    else:  # only one model estimate given
        all_estimates = [estimates]

    # second, calculate likelihood of left out subject for all folds
    print()

    all_likelihoods = []
    for estimates in all_estimates:
        # option 1 and 2: single model and single model with levels
        if "n_events" not in estimates[0].dims:
            if verbose:
                if "level" in estimates[0].dims:
                    print(
                        f"Calculating likelihood for multilevel model with "
                        f"{np.max(estimates[0].event).values + 1} event(s)"
                    )
                else:
                    print(
                        f"Calculating likelihood for single model with "
                        f"{np.max(estimates[0].event).values + 1} event(s)"
                    )

            loocv = []
            if cpus == 1:  # not mp
                for pidx, participant in enumerate(participants_idx):
                    loocv.append(
                        loocv_loglikelihood(
                            data, init, participant, estimates[pidx], verbose=verbose
                        )
                    )
            else:  # mp
                with mp.Pool(processes=cpus) as pool:
                    loocv = pool.starmap(
                        loocv_loglikelihood,
                        zip(
                            itertools.repeat(data),
                            itertools.repeat(init),
                            participants_idx,
                            estimates,
                            itertools.repeat(1),
                            itertools.repeat(verbose),
                        ),
                    )

            likelihoods = xr.DataArray(
                np.array(loocv).astype(np.float64),
                dims="participant",
                coords={"participant": participants_idx},
                name="loo_likelihood",
            )

        # option 3: backward
        if "n_events" in estimates[0].dims:
            # check max n_events (might differ by subject if fit function used)
            n_events_by_subject = [np.max(x.n_events.values) for x in estimates]
            max_n_events_over_subjects = np.min(n_events_by_subject)
            min_n_events = np.min(estimates[0].n_events.values)

            if verbose:
                print(
                    f"Calculating likelihood for backward estimation models with "
                    f"{max_n_events_over_subjects} to {min_n_events} event(s)"
                )

            loocv_back = []
            for n_eve in np.arange(max_n_events_over_subjects, min_n_events - 1, -1):
                if verbose:
                    print(
                        f"  Calculating likelihood for backward estimation model with {n_eve} "
                        "event(s)"
                    )
                loocv = []
                if cpus == 1:  # not mp
                    for pidx, participant in enumerate(participants_idx):
                        loocv.append(
                            loocv_loglikelihood(
                                data,
                                init,
                                participant,
                                estimates[pidx].sel(n_events=n_eve).dropna("event", how="all"),
                                verbose=verbose,
                            )
                        )
                else:  # mp
                    with mp.Pool(processes=cpus) as pool:
                        loocv = pool.starmap(
                            loocv_loglikelihood,
                            zip(
                                itertools.repeat(data),
                                itertools.repeat(init),
                                participants_idx,
                                [
                                    estimates[x].sel(n_events=n_eve).dropna("event", how="all")
                                    for x in range(len(participants_idx))
                                ],
                                itertools.repeat(1),
                                itertools.repeat(verbose),
                            ),
                        )

                loocv_back.append(
                    xr.DataArray(
                        np.expand_dims(np.array(loocv).astype(np.float64), axis=0),
                        dims=("n_event", "participant"),
                        coords={"n_event": np.array([n_eve]), "participant": participants_idx},
                        name="loo_likelihood",
                    )
                )

            likelihoods = xr.concat(loocv_back, dim="n_event")

        all_likelihoods.append(likelihoods)

    if len(all_likelihoods) == 1:
        all_likelihoods = all_likelihoods[0]
        all_estimates = all_estimates[0]

    return all_likelihoods, all_estimates


def backward_func(
    hmp_model,
    max_events=None,
    min_events=0,
    max_starting_points=1,
    tolerance=1e-4,
    max_iteration=1e3,
):
    """Wrap backward estimation helper.

    Helper function for loocv_backward. Calls backward_estimation on hmp_model with provided args.
    """
    return hmp_model.backward_estimation(
        max_events, min_events, None, max_starting_points, tolerance, True, max_iteration
    )


def fit_backward_func(hmp_model, by_sample=False, min_events=0, tolerance=1e-4, max_iteration=1e3):
    """Wrap fit backward function.

    Helper function for fit_loocv_backward. Calls fit function followed by backward_estimation on
    hmp_model with provided args.
    """
    # fit model
    fit_model = hmp_model.fit(by_sample=by_sample, tolerance=tolerance)

    # backward estimation based on fit
    backward_model = hmp_model.backward_estimation(
        max_fit=fit_model, min_events=min_events, tolerance=tolerance, max_iteration=max_iteration
    )

    return backward_model


def loocv_backward(
    init,
    data,
    max_events=None,
    min_events=0,
    max_starting_points=1,
    tolerance=1e-4,
    max_iteration=1e3,
    cpus=1,
    verbose=True,
):
    """Perform leave-one-out cross validation with backward estimation.

    Performs leave-one-out cross validation using backward_estimation to calculate the initial fit.
    It will perform loocv by leaving out one participant, applying 'backward_estimation' to the
    data to estimate a fit, and computing the likelihood of the data from the left out
    participant with the estimated parameters. This is repeated for all participants.

    Hmp model settings are based on init.

    Parameters
    ----------
    init : hmp model
        initialized hmp model
    data : xarray.Dataset
        xarray data from transform_data()
    max_events : int
        Maximum number of events to be estimated, by default the output of
        hmp.models.HMP.compute_max_events()
    min_events : int
        The minimum number of events to be estimated
    max_starting_points: int
        how many random starting points iteration to try for the model estimating the maximal number
        of events.
    tolerance: float
        Tolerance applied to the expectation maximization in the EM() function
    max_iteration: int
        Maximum number of iteration for the expectation maximization in the EM() function
    cpus : int
        Nr of cpus to use. If 1, LOOCV is performed on a single CPU. Otherwise
        on the provided int or setting in init.
        We recommend using 1 CPU at this level on a laptop or normal PC. Only use multiple
        CPUs if you have *a lot* of memory available.
    verbose : bool

    Returns
    -------
        likelihood object and fitten backward estimation models
    """
    return loocv_func(
        init,
        data,
        backward_func,
        func_args=[max_events, min_events, max_starting_points, tolerance, max_iteration],
        cpus=cpus,
        verbose=verbose,
    )


def loocv_fit_backward(
    init,
    data,
    by_sample=False,
    min_events=0,
    tolerance=1e-4,
    max_iteration=1e3,
    cpus=1,
    verbose=True,
):
    """Perform leave-one-out cross validation with fit function and backward estimation.

    Performs leave-one-out cross validation using the fit function followed by backward_estimation
    using the fit-ted model as the maximal model to calculate the initial fit.
    It will perform loocv by leaving out one participant, applying 'fit + backward_estimation'
    to the data to estimate a fit, and computing the likelihood of the data from the left out
    participant with the estimated parameters. This is repeated for all participants. Because the
    fit function might return models with different numbers of events, loglikelihood are calculated
    for models that exist for all folds.

    Hmp model settings are based on init.

    Parameters
    ----------
    init : hmp model
        initialized hmp model
    data : xarray.Dataset
        xarray data from transform_data()
    fix_prev : bool
        whether to fix previous estimates in fit function (default = False)
    by_sample : bool
        whether to explore data sample-by-sample in fit function
    min_events : int
        The minimum number of events to be estimated in backward_estimation
    tolerance: float
        Tolerance applied to the expectation maximization in both the fit function and
        the EM() function
    max_iteration: int
        Maximum number of iteration for the expectation maximization in the EM() function
    cpus : int
        Nr of cpus to use. If 1, LOOCV is performed on a single CPU. Otherwise
        on the provided int or setting in init.
        We recommend using 1 CPU at this level on a laptop or normal PC. Only use multiple
        CPUs if you have *a lot* of memory available.
    verbose : bool

    Returns
    -------
    likelihood object and fitten backward estimation models
    """
    return loocv_func(
        init,
        data,
        fit_backward_func,
        func_args=[by_sample, min_events, tolerance, max_iteration],
        cpus=cpus,
        verbose=verbose,
    )
