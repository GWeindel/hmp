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

    Performs leave-one-out cross validation using either the provided HMP model or
    user-defined function to calculate the initial fit. It will perform loocv by 
    leaving out one participant, applying the provided model or function to the
    data to estimate a fit, and computing the likelihood of the data from the left out
    participant with the estimated parameters. This is repeated for all participants.

    If a self-defined function is used, it must accept data (see fit(..) below) as its
    first argument. Additional arguments can be provided through named func_kwargs. 
    The function is required to return a (list of) fitted model(s); for all provided
    models the likelihood of the left out participant will be calculated.

    The reason for using a function instead of a model is if a sequence of operations
    needs to be performed for each subject (e.g., fitting multiple models in sequence,
    see example_complex_func(..) below.

    For an example of potential functions, see at the bottom of this file:
    example_simple_func(..)
    example_complex_func(..)

    Parameters
    ----------
    model : Any 
        either a (fitted) HMP model or a self-defined function to be applied to
        provided data. If function, must accept data as its first argument and 
        return a (list of) fitted model(s). See explanation above.
    function_kwargs : dict, optional
        additional arguments to pass on to self-defined function.
        Default = None.
    quick : bool, optional
        Toggle for quick LOOCV using parameters of fitted model. 
        Typically incorrect, see info message. Requires fitted model.
        Not possible for function or CumulativeMethods, as params
        cannot be provided (unless through function_params).
        Default = False.
    pattern : hmp.patterns.Pattern
        The pattern and properties to use for cross-correlation. This can be used 
        when providing a function as model in combination with preprocessed data.
        If model or PatternData is provided, Pattern in model or PatternData takes 
        precedence in fit.
    pca_cv: bool, optional
        Whether PCA has to be applied to each fold separately, and weights
        used for left-out participant. Only way to do proper independent
        LOOCV. Requires epoched, non-preprocessed EEG data.
        Default = False
    pca_kwargs : dict, optional/required if pca = True
        arguments to be passed to preprocessors.ProjPCA, n_comp is required.
        Default = None.
    verbose: bool, optional
        Default = True
    print_warning : bool, optional
        print warning when using 'quick' LOOCV the loocv 'incorrectness' warning
    """

    def __init__(
        self,
        model: Any,
        function_kwargs: dict = None,
        quick: bool = False,
        pattern: hmp.patterns.Pattern = None,
        pca_cv: bool = False,
        pca_kwargs : dict = None,
        verbose: bool = True,
        print_warning: bool = True
    ):

        self.model = model
        self.function_kwargs = function_kwargs
        self.quick = quick
        self.pca_cv = pca_cv
        self.pca_kwargs = pca_kwargs

        #instance of an hmp model
        if isinstance(model, hmp.models.base.BaseModel):
            self.model_class = type(model)

            if quick:
                assert self.model._fitted, "For quick estimation, a fitted HMP model is required, but not provided."
                
                if verbose:
                    if print_warning:
                        print()
                        print("IMPORTANT: 'quick' set to true, requesting a faster LOOCV procedure.")
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
        
                if self.model_class == hmp.models.CumulativeMethod:
                    if verbose and quick:
                        print("NB: quick estimation is not possible with CumulativeMethod")
                        print("    Continuing slowly.")
                    self.quick = False

        #assume it's a function that will return a fitted hmp model
        elif inspect.isfunction(model): 
            if verbose and quick:
                print("NB: quick estimation is not possible in combination with a function.")
                print("    Continuing slowly.")
            self.quick = False
            self.model_class = "function"
        
        #unknown
        else:
            raise ValueError(f"Unknown model definition {model}, aborting.")  

        if pattern is None: #no pattern given
            if self.model_class != "function":
                self.pattern = self.model.pattern
            else:
                self.pattern = None
        else: #pattern provided
            if self.model_class != "function":
                if not np.array_equal(pattern.template, self.model.pattern.template):
                    print("Different pattern provided for loocv than in model.")
                    print("Proceeding with model Pattern.")
                self.pattern = self.model.pattern
            else:
                self.pattern = pattern

        if self.pca_cv:
            assert pca_kwargs is not None, "If pca == True, pca_kwargs have to be provided"
            assert 'n_comp' in pca_kwargs, "n_comp required for pca"

    def fit(
        self,
        data: Any,
        cpus_cv: int = 1,
        cpus_model: int = 1,
        verbose: bool = True):

        """Calculate LOOCV by first fitting n-1 models and then calculating the 
        loglikelihood of the nth subject, rotating over subjects.
        
        Parameters
        ----------
        data : Data to fit the model on. Three options:
            1. data from BasePreprocessor
            2. PatternData object.
            3. Epoched data from MNE
            In case of option 1, data is cross-correlated with the pattern in self.pattern.
            If using a 'function' for LOOCV (see init), it is recommended to provide PatternData 
            or pattern to reduce RAM requirements. If 3, pca_kwargs are required to allow PCA.
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
        if isinstance(data, xr.core.dataset.Dataset) and 'epoch' in data.coords:
            assert self.pca_kwargs is not None, "If epoched data is provided, pca_kwargs are required"
            assert 'n_comp' in self.pca_kwargs, "n_comp required for pca"

        if self.pca_cv:
            #check epoched data
            assert isinstance(data, xr.core.dataset.Dataset) and 'epoch' in data.coords,\
                "If PCA, data must be provided as raw, epoched data from MNE."
        else:
            if isinstance(data, xr.core.dataset.Dataset) and 'epoch' in data.coords:
                #warn and apply pca
                print("Epoched data provided but PCA cross validation not requested.")
                print("Continuing by performing PCA on all data, NOT in folds.")
                data = hmp.preprocessors.ProjPCA(data, **self.pca_kwargs)

        if isinstance(data, PatternData) or isinstance(data, xr.core.dataset.Dataset):
            pass
        elif isinstance(data, hmp.preprocessors.pca.ProjPCA):
            if self.pattern is not None:
                data = PatternData.from_preprocessor(data, self.pattern)
            else: # must be function: cannot create PatternData as template is unknown
                if verbose:
                    print("NOTE: 'function' provided for LOOCV without PatternData or Pattern")
                    print("While this is possible, it roughly doubles RAM usage. If")
                    print("using a function, it is recommended to provide PatternData or pattern.")

        #if Eliminative/Cumulative, set max_events based on all data
        if self.model_class in [hmp.models.EliminativeMethod, hmp.models.CumulativeMethod]:
            if self.model.max_events is None:
                self.model.max_events = \
                    self.model._compute_max_events(data, self.model.location)              

        # Get participants here to be able to split for multithreading
        if isinstance(data, PatternData):
            self.participants_idx = np.unique(data.durations.participant.values)
        elif isinstance(data, hmp.preprocessors.pca.ProjPCA):
            self.participants_idx = np.unique(data.data.participant.values)
        elif isinstance(data, xr.core.dataset.Dataset):
            self.participants_idx = np.unique(data.participant.values)

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
                    )
                )
        
        # if multiple estimates are returned per subject, rearrange data so
        # each item of all_estimates will contain one estimate for each subject.
        # So for eliminative, all_estimates[0] will have all n-stage subject
        # models.
        if isinstance(estimates[0], list): #from a function or quick elim
            all_estimates = []
            for est_idx in range(len(estimates[0])):
                #each estimate can also be eliminative or cumulative
                if isinstance(estimates[0][est_idx], hmp.models.EliminativeMethod):
                    for est_idx2 in range(estimates[0][est_idx].max_events):
                        all_estimates.append([estimate[est_idx].submodels[est_idx2+1] for estimate in estimates])
                elif isinstance(estimates[0][est_idx], hmp.models.CumulativeMethod):
                    all_estimates.append([estimate[est_idx].submodels[-1] for estimate in estimates])
                elif isinstance(estimates[0][est_idx], hmp.models.EventModel):                       
                    all_estimates.append([estimate[est_idx] for estimate in estimates])
        elif isinstance(estimates[0], hmp.models.EliminativeMethod):
            all_estimates = []
            for est_idx in range(estimates[0].max_events):
                all_estimates.append([estimate.submodels[est_idx+1] for estimate in estimates])
        elif isinstance(estimates[0], hmp.models.CumulativeMethod):
            all_estimates = [[estimate.submodels[-1] for estimate in estimates]]
        elif isinstance(estimates[0], hmp.models.EventModel):                       
            all_estimates = [estimates]

        # Step 2, get loglikelihood from left out subjects
        print()
        all_likelihoods = []
        for est, estimates in enumerate(all_estimates):
            if verbose:
                print("Calculating likelihoods.")

            loocv = []
            if cpus_cv == 1:  # no mp for cross validation
                for pidx, participant in enumerate(self.participants_idx):
                    loocv.append(
                        self.loocv_loglikelihood(
                            data, 
                            participant, 
                            estimates[pidx], 
                            cpus_model=cpus_model,
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
                        )
                    )

            #format results
            likelihoods = xr.DataArray(
                np.expand_dims(np.array(loocv).astype(np.float64), axis=0),
                dims=("n_event", "participant"),
                coords={"n_event": np.array([estimates[0].n_events]),
                    "participant": self.participants_idx},
                name="loo_likelihood"
            )
            likelihoods.attrs['model_class'] = self.model_class
            likelihoods.attrs['model'] = self.model
            likelihoods.attrs['n_events'] = np.array([estimate.n_events for estimate in estimates])
            if not np.all(likelihoods.n_events==likelihoods.n_events[0]):
                likelihoods['n_event'] = 'variable'
            likelihoods.attrs['quick'] = self.quick
            
            estimates_xr = likelihoods.copy()
            estimates_xr.data = np.expand_dims(np.array(estimates), axis=0)
            all_estimates[est] = estimates_xr
            all_likelihoods.append(likelihoods)

        #In case of eliminative, we can concat likelihoods and estimates
        if self.model_class == hmp.models.EliminativeMethod:
            all_likelihoods = xr.concat(all_likelihoods, dim="n_event")
            all_likelihoods.attrs['n_event'] = np.zeros(all_likelihoods.data.shape).astype(int)
            for ev in range(all_likelihoods.data.shape[0]):
                all_likelihoods.attrs['n_event'][ev,:] = [estimate.n_events for estimate in all_estimates[ev].data[0,:]]
            all_estimates = xr.concat(all_estimates, dim="n_event")
            all_estimates.attrs['n_event'] = all_likelihoods.attrs['n_event']
        elif len(all_likelihoods) == 1:
            all_likelihoods = all_likelihoods[0]
        if len(all_estimates) == 1:
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
        elif isinstance(data, hmp.preprocessors.pca.ProjPCA):
            data = hmp.transformers.BaseTransformer.remove_participant(data, participant)
        elif isinstance(data, xr.core.dataset.Dataset): #pca_cv
            data = data.drop_sel(participant = [participant])
            data = hmp.preprocessors.ProjPCA(data, **self.pca_kwargs)

        # Fit model on data
        if self.model_class == "function":
            estimated_model = self.model(data, **self.function_kwargs)
        else:
            if not self.quick: #same for Event and Eliminative
                estimated_model = copy.deepcopy(self.model)
                estimated_model.fit(data=data, cpus=cpus_model, verbose= False)
            else: #quick
                if self.model_class == hmp.models.EventModel: #use prev params
                    estimated_model = copy.deepcopy(self.model)
                    estimated_model.fit(data=data, channel_pars=estimated_model.channel_pars,\
                         time_pars=estimated_model.time_pars, cpus=cpus_model, verbose= False)
                elif self.model_class == hmp.models.EliminativeMethod: 
                #use prev params per model
                    estimated_model = []
                    for n_events in range(1, self.model.max_events + 1):
                        estimated_model.append(copy.deepcopy(self.model.submodels[n_events]))
                        estimated_model[-1].fit(data=data, \
                            channel_pars=estimated_model[-1].channel_pars, \
                            time_pars=estimated_model[-1].time_pars, cpus=cpus_model, \
                            verbose= False)

        #if pca_cv, attach pca weights to each model
        if self.pca_cv:
            if isinstance(estimated_model, hmp.models.EventModel):
                estimated_model.pca_weights = data.weights
            elif isinstance(estimated_model, hmp.models.CumulativeMethod):
                estimated_model.submodels[-1].pca_weights = data.weights
            elif isinstance(estimated_model, hmp.models.EliminativeMethod):
                for est_idx in range(estimated_model.max_events):
                    estimated_model.submodels[est_idx+1].pca_weights = data.weights
            elif isinstance(estimated_model, list): #from a function or quick elim
                for est_idx in range(len(estimated_model)):
                    #each estimate can also be eliminative or cumulative
                    if isinstance(estimated_model[est_idx], hmp.models.EliminativeMethod):
                        for est_idx2 in range(estimated_model[est_idx].max_events):
                            estimated_model[est_idx].submodels[est_idx2+1].pca_weights = data.weights
                    elif isinstance(estimated_model[est_idx], hmp.models.CumulativeMethod):
                        estimated_model[est_idx].submodels[-1].pca_weights = data.weights
                    elif isinstance(estimated_model[est_idx], hmp.models.EventModel):
                        estimated_model[est_idx].pca_weights = data.weights

        return estimated_model


    def loocv_loglikelihood(self, data, participant, estimate, cpus_model=1, verbose=False):
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
        elif isinstance(data, hmp.preprocessors.pca.ProjPCA):
            data = hmp.transformers.BaseTransformer.get_participants(data, participant)
        elif isinstance(data, xr.core.dataset.Dataset): #pca_cv
            data = data.sel(participant = [participant])
            base_kwargs = {k: self.pca_kwargs[k] for k in \
                            ['interval_id', 'offset_end', 'offset_start',
                              'min_duration', 'max_duration', 'reject_threshold',
                              'verbose','common_variance','subject_zscore',
                              'whiten','center'                             
                             ] & self.pca_kwargs.keys()}
            base_preprocess = hmp.preprocessors.BasePreprocessor(**base_kwargs)
            data = base_preprocess.common_preprocess(data)
            base_preprocess.data_format(data, estimate.pca_weights)
            data = base_preprocess

        # Calculate loglikelihood of model applied on data of participant
        likelihood, _ = estimate.transform(data, cpus=cpus_model)

        if len(likelihood) == 1:
            return likelihood[0]
        else:
            return likelihood




## add function to return average model.
# follow structures of estimates
# if variable nr of events, return average for each number




### Example loocv functions
def example_simple_func(data, n_events, channel_pars=None, time_pars=None, verbose=False):
    """Example of a simple function that estimates an n_event model.

    Note that this would normally not be done, as you could just provide
    this model directly to loocv(..). This is only an example of how a 
    function could be used, see below for a more realistic example.

    Example code
    -------------
    loocv_simple_func = hmp.loocv.LOOCV(example_simple_func, function_kwargs={'n_events' : 2})
    lkh_simple_func, estimates_simple_func = loocv_simple_func.fit(pattern_data, cpus_cv=4, cpus_model=1)
    """

    ev_model = hmp.models.EventModel(n_events=n_events)
    ev_model.fit(data, channel_pars=channel_pars, time_pars=time_pars, verbose=verbose)
    return ev_model

def example_complex_func(
    data, max_events=None, n_events=1, channel_map=None, time_map=None, grouping_dict=None):
    """Example of a complex function for LOOCV.

    This function first performs backwards estimation up to max_events,
    and follows this with a group-based model of n_events, informed
    by the selected backward model and the provided maps. It returns
    both models, so for both the likelihood will be estimated.

    Example code
    -------------
    channel_map = np.array([[0, 0, -1, 0],
                            [0, 0, 0, 0]])
    time_map = np.array([[0, 0, -1, 0, 0],
                     [0, 0, 0, 1, 0]])
    grouping_dict = {'cue': ['SP', 'AC']}
    loocv_complex_func = hmp.loocv.LOOCV(example_complex_func, \
                                     function_kwargs={'max_events' : 5, 
                                                      'channel_map' : channel_map, 
                                                      'time_map': time_map, 
                                                      'grouping_dict': grouping_dict})   
    lkh_complex_func, estimates_complex_func = loocv_complex_func.fit(pattern_data, \
                                                            cpus_cv=4, cpus_model=1)
    """

    # fit backward model up to max_events
    eliminative_model = hmp.models.EliminativeMethod(max_events=max_events)
    eliminative_model.fit(data)

    # fit group model
    group_model = hmp.models.EventModel(n_events=n_events,
                                        channel_map=channel_map,
                                        time_map=time_map, 
                                        grouping_dict=grouping_dict)
    group_model.fit(data=data,
                    channel_pars = eliminative_model.submodels[4].channel_pars,
                    time_pars = eliminative_model.submodels[4].time_pars)

    return [eliminative_model, group_model]