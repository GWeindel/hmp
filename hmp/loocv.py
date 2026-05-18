"""Leave-one out cross-validation class

Class
-------
LOOCV


"""

from typing import Any
import inspect

from hmp.trialdata import TrialData
import hmp

import numpy as np
import itertools
import multiprocessing as mp


class LOOCV():
    """LOOCV class. Can be initialized with a (fitted) model or a self-defined function.

    Parameters TO BE UPDATED
    ----------
    model : Any 
        either a (fitted) model or a self-defined function
    quick : bool, optional
        Toggle for quick LOOCV using parameters of fitted model. 
        Typically incorrect, see info message. Requires fitted model.
        Default = False.
    verbose: bool, optional
        Default = True
    print_warning : bool, optional
        print warning when using 'quick' LOOCV the loocv 'incorrectness' warning


    sfreq : float
        (optional) Sampling frequency of the signal if not provided, inferred from the epoch_data
    event_width : int
        Width of the pattern defining events in samples.
    shape: float
        shape of the probability distributions of the by-trial stage onset
        (one shape for all stages)
    location : int
        Minimum duration between events in samples. Default is the event_width.
    distribution : str
        Probability distribution for the by-trial onset of stages can be
        one of 'gamma','lognormal','wald', or 'weibull'
    """

    def __init__(
        self,
        model: Any,
        quick: bool = False,
        verbose: bool = True,
        print_warning: bool = True
    ):
        #instance of an hmp model
        if isinstance(model, hmp.models.base.BaseModel):
            self.model_class = type(model)
            self.model = model

            if quick:
                assert self.model._fitted, "For quick estimation, a fitted HMP model is required, but not provided."
                
                if verbose:
                    if print_warning:
                        print()
                        print("IMPORTANT: 'quick' set to true, requestion a faster LOOCV procedure.")
                        print()
                        print("Note that the quick procedure is typically incorrect in the sense that")
                        print("an initial estimate is used to inform both the fit of the left-out")
                        print("participant and the other participants.")
                        print()
                        print("This means that the fits are not independent, unless the initial")
                        print("estimate is based on the literature or another task. However, it")
                        print("typically provides a good initial estimate of the correct LOOCV") 
                        print("procedure and is relatively quick.")
                        print()
                        print("Set 'print_warning to False to suppress this info.")
        
        #assume it's a function that will return a fitted hmp model
        elif inspect.isfunction(model): 
            self.model_class = "function"
        
        #unknown
        else:
            raise ValueError(f"Unknown model definition {model}, aborting.")  

        self.quick = quick

        #EventModel
        #if self.model_class == hmp.models.EventModel:
        #    if self.parametrized:
        #        self.n_events = model.n_events


    def fit(
        self,
        trial_data: TrialData,
        cpus_cv: int = 1,
        cpus_model: int = 1,
        verbose = False):

        """Calculate LOOCV by first fitting n-1 models and then calculating the 
        loglikelihood of the nth subject, rotating over subjects.

        ...
        
        Parameters TO BE UPDATED
        ----------
        trial_data : TrialData
            Cross-correlated TrialData object of all participants.
        cpus_cv : int, optional
            Nr of cpus to use for cross validation. Defaul = 1.
            We recommend using 1 CPU for CV on a laptop or normal PC. 
            Only use multiple CPUs if you have *a lot* of memory available.
        cpus_model : int, optional
            Nr of cpus to use for model fitting if cpus_cv == 1,
            otherwise only as single cpu can be used for model
            fitting. cpus_cv takes precedence. Default = 1.
        verbose: bool, optional
            Default = True
       

        """

#process all paramters that can be provided to fit or function, probably give a an extra args parameter
#move map parameters to model definition

        estimates = []
        #simple EventModel - might be possible for all
        if self.model_class == hmp.models.EventModel:

            # Get participants here to be able to split for multithreading
            participants_idx = np.unique(trial_data.durations.participant.values)

            # Step 1, fit models on n-1 subjects for all folds
            if verbose:
                print(f"Fit models for n-1 subjects for all folds.")

            estimates = []
            if cpus_cv == 1:  # not mp at cv level
                for participant in participants_idx:
                    estimates.append(
                        loocv_estimate(
                            trial_data, 
                            participant, 
                            cpus=cpus_model, 
                            verbose=verbose))
            else:  # mp at cv level
                with mp.Pool(processes=cpus_cv) as pool:
                    estimates = pool.starmap(
                        loocv_estimate,
                        zip(
                            itertools.repeat(trial_data),
                            #itertools.repeat(self.model),
                            participants_idx,
                            #itertools.repeat(func_estimate),
                            #itertools.repeat(func_args),
                            itertools.repeat(1), #cpus_model has to be 1
                            itertools.repeat(verbose)
                        ),
                    )


            # Step 2, get loglikelihood from left out subjects


        else:
            print("don't know what to do yet :)")
            lkh_3ev = None

        return estimates
    

def loocv_estimate(trial_data, participant, cpus=1, verbose=True):
    """Apply loocv estimation using either the provided model or function.

    UPDATE
    Applies func_estimate with func_args to data of n - 1 (participant) participants.
    func_estimate should return an estimated hmp model; either a single model,
    a level model, or a backward estimation model. This model is then used
    to calculate the fit on the left out participant with loocv_loglikelihood.

    Parameters TO BE UPDATED
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

    participants_idx = trial_data.durations.participant.values

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



#below should go in general method
    # # if multiple estimates are returned per subject, rearrange data
    # if isinstance(estimates[0], list):
    #     all_estimates = []
    #     for est_idx in range(len(estimates[0])):
    #         all_estimates.append([estimate[est_idx] for estimate in estimates])
    # else:  # only one model estimate given
    #     all_estimates = [estimates]

    # # second, calculate likelihood of left out subject for all folds
    # print()

    # all_likelihoods = []
    # for estimates in all_estimates:
    #     # option 1 and 2: single model and single model with levels
    #     if "n_events" not in estimates[0].dims:
    #         if verbose:
    #             if "level" in estimates[0].dims:
    #                 print(
    #                     f"Calculating likelihood for multilevel model with "
    #                     f"{np.max(estimates[0].event).values + 1} event(s)"
    #                 )
    #             else:
    #                 print(
    #                     f"Calculating likelihood for single model with "
    #                     f"{np.max(estimates[0].event).values + 1} event(s)"
    #                 )

    #         loocv = []
    #         if cpus == 1:  # not mp
    #             for pidx, participant in enumerate(participants_idx):
    #                 loocv.append(
    #                     loocv_loglikelihood(
    #                         data, init, participant, estimates[pidx], verbose=verbose
    #                     )
    #                 )
    #         else:  # mp
    #             with mp.Pool(processes=cpus) as pool:
    #                 loocv = pool.starmap(
    #                     loocv_loglikelihood,
    #                     zip(
    #                         itertools.repeat(data),
    #                         itertools.repeat(init),
    #                         participants_idx,
    #                         estimates,
    #                         itertools.repeat(1),
    #                         itertools.repeat(verbose),
    #                     ),
    #                 )

    #         likelihoods = xr.DataArray(
    #             np.array(loocv).astype(np.float64),
    #             dims="participant",
    #             coords={"participant": participants_idx},
    #             name="loo_likelihood",
    #         )

    #     # option 3: backward
    #     if "n_events" in estimates[0].dims:
    #         # check max n_events (might differ by subject if fit function used)
    #         n_events_by_subject = [np.max(x.n_events.values) for x in estimates]
    #         max_n_events_over_subjects = np.min(n_events_by_subject)
    #         min_n_events = np.min(estimates[0].n_events.values)

    #         if verbose:
    #             print(
    #                 f"Calculating likelihood for backward estimation models with "
    #                 f"{max_n_events_over_subjects} to {min_n_events} event(s)"
    #             )

    #         loocv_back = []
    #         for n_eve in np.arange(max_n_events_over_subjects, min_n_events - 1, -1):
    #             if verbose:
    #                 print(
    #                     f"  Calculating likelihood for backward estimation model with {n_eve} "
    #                     "event(s)"
    #                 )
    #             loocv = []
    #             if cpus == 1:  # not mp
    #                 for pidx, participant in enumerate(participants_idx):
    #                     loocv.append(
    #                         loocv_loglikelihood(
    #                             data,
    #                             init,
    #                             participant,
    #                             estimates[pidx].sel(n_events=n_eve).dropna("event", how="all"),
    #                             verbose=verbose,
    #                         )
    #                     )
    #             else:  # mp
    #                 with mp.Pool(processes=cpus) as pool:
    #                     loocv = pool.starmap(
    #                         loocv_loglikelihood,
    #                         zip(
    #                             itertools.repeat(data),
    #                             itertools.repeat(init),
    #                             participants_idx,
    #                             [
    #                                 estimates[x].sel(n_events=n_eve).dropna("event", how="all")
    #                                 for x in range(len(participants_idx))
    #                             ],
    #                             itertools.repeat(1),
    #                             itertools.repeat(verbose),
    #                         ),
    #                     )

    #             loocv_back.append(
    #                 xr.DataArray(
    #                     np.expand_dims(np.array(loocv).astype(np.float64), axis=0),
    #                     dims=("n_event", "participant"),
    #                     coords={"n_event": np.array([n_eve]), "participant": participants_idx},
    #                     name="loo_likelihood",
    #                 )
    #             )

    #         likelihoods = xr.concat(loocv_back, dim="n_event")

    #     all_likelihoods.append(likelihoods)

    # if len(all_likelihoods) == 1:
    #     all_likelihoods = all_likelihoods[0]
    #     all_estimates = all_estimates[0]

    # return all_likelihoods, all_estimates



## add function to return average mdoel