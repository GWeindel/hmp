"""Leave-one out cross-validation class

Class
-------
LOOCV


"""

from typing import Any
import inspect

from hmp.patterndata import PatternData
import hmp

import numpy as np
import xarray as xr
import itertools
import multiprocessing as mp
import copy

class LOOCV():
    """LOOCV class. Can be initialized with a (fitted) model or a self-defined function.

    Parameters TO BE UPDATED
    ----------
    model : Any 
        either a (fitted) model or a self-defined function to be applied to provided data
    model_args :
    quick : bool, optional
        Toggle for quick LOOCV using parameters of fitted model. 
        Typically incorrect, see info message. Requires fitted model.
        Default = False.
    verbose: bool, optional
        Default = True
    print_warning : bool, optional
        print warning when using 'quick' LOOCV the loocv 'incorrectness' warning
    """

    def __init__(
        self,
        model: Any,
        quick: bool = False,
        verbose: bool = True,
        print_warning: bool = True
    ):

        self.model = model

        #instance of an hmp model
        if isinstance(model, hmp.models.base.BaseModel):
            self.model_class = type(model)

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
            if verbose and quick:
                print("NB: quick estimation is not possible in combination with a function.")
            self.model_class = "function"
        
        #unknown
        else:
            raise ValueError(f"Unknown model definition {model}, aborting.")  

        self.quick = quick


    def fit(
        self,
        data: Any,
        cpus_cv: int = 1,
        cpus_model: int = 1,
        verbose: bool = True):

        """Calculate LOOCV by first fitting n-1 models and then calculating the 
        loglikelihood of the nth subject, rotating over subjects.

        ...
        
        Parameters
        ----------
        data : Data to fit the model on. One of two options:
            1. data from BaseTransformer
            2. PatternData object.
            In case of option 1, data is cross-correlated with the pattern in self.model.pattern.
            If using a 'function' for LOOCV (see init), it is recommended to provide PatternData to reduce RAM requirements.
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

        # Only one parallel loop, cv takes precedence:
        if cpus_cv > 1:
            cpus_model = 1

        # Prep data
        if isinstance(data, PatternData): #easiest, just use as-is
            pass
        else: #assume transformed (is checked later)
            if self.model_class != "function":
                if self.model.pattern.sfreq is None:
                    self.model.pattern.create_template(data.sfreq)
                data = PatternData.from_transformer(data, self.model.pattern)
            else: #function: cannot create PatternData as template is unknown
                if verbose:
                    print("NOTE: 'function' provided for LOOCV without PatternData")
                    print("While this is possible, it roughly doubles RAM usage. If")
                    print("using a function, it is recommended to provide PatternData.")

        # Get participants here to be able to split for multithreading
        if isinstance(data, PatternData):
            self.participants_idx = np.unique(data.durations.participant.values)
        else:
            self.participants_idx = np.unique(data.data.participant.values)


        # Step 1, fit models on n-1 subjects for all folds
        if verbose:
            print(f"Fitting models for n-1 subjects for all folds.")

        estimates = []
        if cpus_cv == 1:  # not mp at cv level
            for participant in self.participants_idx:
                estimates.append(
                    self.loocv_estimate(
                        data, 
                        participant, 
                        cpus_model=cpus_model, 
                        verbose=verbose))
        else:  # mp at cv level
            with mp.Pool(processes=cpus_cv) as pool:
                estimates = pool.starmap(
                    self.loocv_estimate,
                    zip(
                        itertools.repeat(data),
                        self.participants_idx,
                        itertools.repeat(1), #cpus_model has to be 1
                        itertools.repeat(verbose)
                    ),
                )

        # if multiple estimates are returned per subject, rearrange data
        if isinstance(estimates[0], list):
            all_estimates = []
            for est_idx in range(len(estimates[0])):
                all_estimates.append([estimate[est_idx] for estimate in estimates])
        else:  # only one model estimate given per participant
            all_estimates = [estimates]


        # Step 2, get loglikelihood from left out subjects
        print()
        all_likelihoods = []

        for estimates in all_estimates:
            # option 1 and 2: single model and single model with levels. In fact, aren't they all eventmodels?
            if self.model_class == hmp.models.EventModel:
                if verbose:
                    mod_type = "multilevel" if estimates[0].time_pars.shape[0] > 1 else "single"
                    print(
                        f"Calculating likelihood for {mod_type} with "
                        f"{estimates[0].n_events} event(s)"
                    )

                loocv = []
                if cpus_cv == 1:  # no mp for cross validation
                    for pidx, participant in enumerate(self.participants_idx):
                        loocv.append(
                            self.loocv_loglikelihood(
                                data, 
                                participant, 
                                estimates[pidx], 
                                cpus=cpus_model,
                                verbose=verbose
                            )
                        )
                else:  # mp
                    with mp.Pool(processes=cpus_cv) as pool:
                        loocv = pool.starmap(
                            self.loocv_loglikelihood,
                            zip(
                                itertools.repeat(data),
                                self.participants_idx,
                                estimates,
                                itertools.repeat(1), #cpus
                                itertools.repeat(verbose),
                            ),
                        )

            likelihoods = xr.DataArray(
                np.array(loocv).astype(np.float64),
                dims="participant",
                coords={"participant": self.participants_idx},
                name="loo_likelihood",
            )
                
            all_likelihoods.append(likelihoods)


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





        #admin
        if len(all_likelihoods) == 1:
            all_likelihoods = all_likelihoods[0]
            all_estimates = all_estimates[0]
        
        return all_likelihoods, all_estimates
    

    def loocv_estimate(self, data, participant, cpus_model=1, verbose=True):
        """Apply loocv estimation using either the provided model or function, while
        leaving out participant 'participant'

        Parameters 
        ----------
        data : Data to fit the model on. One of two options:
            1. data from BaseTransformer
            2. PatternData object.
            In case of option 1, data is cross-correlated with the pattern in self.model.pattern.
            If using a 'function' for LOOCV (see init), it is recommended to provide PatternData to reduce RAM requirements.
        participant : str
            name of the participant to leave out
        cpus_model : int
            nr of cpus to use for model estimation (nr of cpus for cross validation specified 
            above)
        verbose : bool

        Returns
        -------
        hmp model
            estimated hmp_model(s) on n-1 participants
        """

        if verbose:
            print(f"\tEstimating model for all participants except {participant}")

        # Extract data without left-out participant
        if isinstance(data, PatternData):
            data = hmp.patterndata.remove_participant(data, participant)
        else:
            data = hmp.transformers.BaseTransformer.remove_participant(data, participant)

        # Fit model on data
        if self.model_class == "function":
            print("don't know what to do yet!")
        else:    
            estimated_model = copy.deepcopy(self.model)
            if self.quick:
                estimated_model.fit(data=data, channel_pars=estimated_model.channel_pars, time_pars=estimated_model.time_pars, cpus=cpus_model, verbose= False)
            else:
                estimated_model.fit(data=data, cpus=cpus_model, verbose= False)
            return estimated_model


    #estimates = []
    #simple EventModel - might be possible for all
    #if self.model_class == hmp.models.EventModel:
    #    pass

    # Apply function and return
    #estimates = func_estimate(model_without_pp, *func_args)
    #if isinstance(estimates, list):
    #    for i in range(len(estimates)):
    #        estimates[i] = estimates[i].drop_vars(["eventprobs"])
    #else:
    #    estimates = estimates.drop_vars(["eventprobs"])

    #return estimates

    def loocv_loglikelihood(self, data,participant, estimate, cpus_model=1, verbose=False):
        """Compute the log-likelihood of the fit.

        Calculate loglikelihood of fit on participant using parameters from estimate,
        either using single model or level based model.

        Parameters
        ----------
        data : Data to fit the model on. One of two options:
            1. data from BaseTransformer
            2. PatternData object.
            In case of option 1, data is cross-correlated with the pattern in self.model.pattern.
            If using a 'function' for LOOCV (see init), it is recommended to provide PatternData to reduce RAM requirements.
        participant : str
            name of the participant to estimate likelihood
        estimate : xarray.Dataset
            estimate that has parameters to apply.
        cpus_model : int
            Number of cpus to use to fit the models.
        verbose : bool

        Returns
        -------
        likelihood : float
            likelihood computed for the left-out participant
        """
        if verbose:
            print(f"\tCalculating likelihood for participant {participant}")

        # Extract data of left-out participant
        if isinstance(data, PatternData):
            data = hmp.patterndata.get_participants(data, participant)
        else:
            data = hmp.transformers.BaseTransformer.get_participants(data, participant)

        # Calculate loglikelihood of model applied on data of participant
        likelihood, _ = estimate.transform(data, cpus=cpus_model)

        
        # if "level" in estimate.dims:
        #     locations = np.tile(locations, (estimate.parameters.shape[0], 1))  # Fix this
        #     from itertools import product

        #     # create levels for this participant based on estimate.levels_dict and model_pp
        #     # description of level for this participant, which is not available
        #     levels = estimate.levels_dict
        #     level_names = []
        #     level_levels = []
        #     level_trials = []
        #     for level in levels:
        #         level_names.append(list(level.keys())[0])
        #         level_levels.append(level[level_names[-1]])
        #         level_trials.append(model_pp.trial_coords[level_names[-1]].data.copy())

        #     level_levels = list(product(*level_levels))
        #     level_levels = np.array(level_levels, dtype=object)  # otherwise comparison below can fail

        #     # build level array with digit indicating the combined levels
        #     level_trials = np.vstack(level_trials).T
        #     levels = np.zeros((level_trials.shape[0])) * np.nan
        #     for i, level in enumerate(level_levels):
        #         levels[np.where((level_trials == level).all(axis=1))] = i
        #     levels = np.int8(levels)

        #     # adjust parameters based on average RT
        #     parameters = estimate.parameters.values
        #     parameters[:, :, 1] = parameters[:, :, 1] * dur_ratio

        #     likelihood = model_pp.estim_probs_levels(
        #         estimate.magnitudes.values,
        #         parameters,
        #         locations,
        #         estimate.mags_map,
        #         estimate.pars_map,
        #         levels,
        #         lkh_only=True,
        #     )
        # else:
        #     n_eve = np.max(estimate.event.dropna("event").values) + 1
        #     print(n_eve)
        #     # adjust parameters based on average RT
        #     parameters = estimate.parameters.dropna("stage").values
        #     parameters[:, 1] = parameters[:, 1] * dur_ratio

        #     likelihood = model_pp.estim_probs(
        #         estimate.magnitudes.dropna("event", how="all").values,
        #         parameters,
        #         locations,
        #         n_eve,
        #         None,
        #         True,
        #     )

        if len(likelihood) == 1:
            return likelihood[0]
        else:
            return likelihood






## add function to return average mdoel