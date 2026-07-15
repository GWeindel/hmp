"""Leave-one out cross-validation class

Class
-------
LOOCV


"""

from collections.abc import Callable
import inspect

from hmp.patterns import Pattern
from hmp.patterndata import PatternData
from hmp.basedata import BaseData
from hmp.projectors import Custom, Identity, PCA
from hmp.models.base import BaseModel
from hmp.models.event import EventModel
from hmp.models.eliminative import EliminativeMethod
from hmp.models.cumulative import CumulativeMethod
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
    leaving out one subject or recording, applying the provided model or function to the
    data to compute a fit, and computing the likelihood of the data from the left out
    subject/recording with the estimated parameters. This is repeated for all subjects/recordings.

    If a self-defined function is used, it must accept data (see fit(..) below) as its
    first argument. Additional arguments can be provided through named func_kwargs. 
    The function is required to return a (list of) fitted model(s); for all provided
    models the likelihood of the left out subject/recording will be calculated.

    The reason for using a function instead of a model is if a sequence of operations
    needs to be performed for each subject/recording (e.g., fitting multiple models in sequence).
    For an example of potential functions, see at the bottom of this file:
    example_simple_func(..)
    example_complex_func(..)

    Parameters
    ----------
    model : BaseModel | Callable
        either a (fitted) HMP model or a self-defined function to be applied to
        provided data. If function, must accept data as its first argument and 
        return a (list of) fitted model(s). See explanation above.
    dimension : Str
        dimension to perform LOOCV over, either 'subject' or 'recording'
        Default = 'subject'.
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
        LOOCV. Requires un-projected BaseData object, with cropping and rejecting
        epochs already applied. LOOCV will call BaseData.pca_and_variance(**pca_kwargs).
        Default = False
    pca_kwargs : dict, optional/required if pca = True
        arguments to be passed to BaseData.pca_and_variance(), n_comp is required.
        Default = None.
    verbose: bool, optional
        Default = True
    print_warning : bool, optional
        print warning when using 'quick' LOOCV the loocv 'incorrectness' warning
    """

    def __init__(
        self,
        model: BaseModel | Callable,
        dimension: str = 'subject',
        function_kwargs: dict = None,
        quick: bool = False,
        pattern: Pattern = None,
        pca_cv: bool = False,
        pca_kwargs : dict = None,
        verbose: bool = True,
        print_warning: bool = True
    ):

        self.model = model
        self.dimension = dimension
        self.function_kwargs = function_kwargs
        self.quick = quick
        self.pca_cv = pca_cv
        self.pca_kwargs = pca_kwargs

        #instance of an hmp model
        if isinstance(model, BaseModel):
            self.model_class = type(model)

            if quick:
                assert self.model._fitted, \
                    "For quick estimation, a fitted HMP model is required, but not provided."
                
                if verbose:
                    if print_warning:
                        print()
                        print("IMPORTANT: 'quick' set to true, requesting a faster LOOCV procedure.")
                        print()
                        print("Note that the quick procedure is typically incorrect in the sense that")
                        print("an initial model fit is used to inform both the fit of the left-out")
                        print("participant and the other participants.")
                        print()
                        print("This means that the fits are not independent, unless the initial")
                        print("model fit is based on the literature or another task. However, it")
                        print("typically provides a good initial model fit of the correct LOOCV") 
                        print("procedure and is relatively quick.")
                        print()
                        print("Set 'print_warning to False to suppress this info.")
        
                if self.model_class == CumulativeMethod:
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

        if pattern is None: #no pattern given
            if self.model_class != "function":
                self.pattern = self.model.pattern
            else:
                self.pattern = None
        else: #pattern provided
            if self.model_class != "function":
                if not np.array_equal(pattern.template, self.model.pattern.template):
                    print("Different pattern provided for LOOCV than in model.")
                    print("Proceeding with model Pattern.")
                self.pattern = self.model.pattern
            else:
                self.pattern = pattern

        if self.pca_cv:
            assert pca_kwargs is not None, "If pca_cv == True, pca_kwargs have to be provided"
            assert 'n_comp' in pca_kwargs, "n_comp required for pca"

    def fit(
        self,
        data: BaseData | PatternData,
        cpus_cv: int = 1,
        cpus_model: int = 1,
        verbose: bool = True):

        """Calculate LOOCV by first fitting n-1 models and then calculating the 
        loglikelihood of the nth subject, rotating over subjects.
        
        Parameters
        ----------
        data : Data to fit the model on. Two options:
            1. BaseData object
            2. PatternData object
            In case of option 1, data is cross-correlated with the pattern in self.pattern.
            If using a 'function' for LOOCV (see init), it is recommended to provide PatternData 
            or pattern to reduce RAM requirements.
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
        if isinstance(data, BaseData) and (not hasattr(data,'projector') or isinstance(data.projector, Identity)): #PCA not applied
            assert self.pca_kwargs is not None, \
                "If non-projected data are provided, pca_kwargs are required"
            assert 'n_comp' in self.pca_kwargs, "n_comp required for pca"

        if self.pca_cv:
            assert isinstance(data, BaseData) and (not hasattr(data,'projector') or isinstance(data.projector, Identity)),\
                "If PCA cross-validation, data must be provided as non-projected BaseData."
        else:
            if isinstance(data, BaseData) and (not hasattr(data,'projector') or isinstance(data.projector, Identity)):
                #warn and apply pca
                print("Non-projected BaseData provided but PCA cross validation not requested.")
                print("Continuing by performing PCA on all data, NOT in folds.")
                data.pca_and_variance(**self.pca_kwargs)

        if isinstance(data, BaseData) and not (not hasattr(data,'projector') or isinstance(data.projector, Identity)):
            if self.pattern is not None:
                data = PatternData.from_basedata(data, self.pattern)
            else: # must be function: cannot create PatternData as template is unknown
                if verbose:
                    print("NOTE: 'function' provided for LOOCV without PatternData or Pattern")
                    print("While this is possible, it roughly doubles RAM usage. If")
                    print("using a function, it is recommended to provide PatternData or pattern.")

        #if Eliminative/Cumulative, set max_events based on all data
        if self.model_class in [EliminativeMethod, CumulativeMethod]:
            if self.model.max_events is None:
                self.model.max_events = \
                    self.model._compute_max_events(data, self.model.location)              

        # Get subjects/recording here to be able to split for multithreading
        if isinstance(data, PatternData):
            self.dimension_idx = np.unique(data.durations[self.dimension].values)
        elif isinstance(data, BaseData):
            self.dimension_idx = np.unique(data.data[self.dimension].values)

        # Step 1, fit models on n-1 subjects for all folds
        if verbose:
            print(f"Fitting models for n-1 subjects for all folds.")

        modelfits = []
        if cpus_cv == 1:  # not mp at cv level
            for dim in self.dimension_idx:
                modelfits.append(
                    self.loocv_modelfit(
                        data, 
                        dim, 
                        cpus_model=cpus_model, 
                        verbose=verbose))
        else:  # mp at cv level
            with mp.Pool(processes=cpus_cv) as pool:
                modelfits = pool.starmap(
                    self.loocv_modelfit,
                    zip(
                        itertools.repeat(data),
                        self.dimension_idx,
                        itertools.repeat(1), #cpus_model has to be 1
                        itertools.repeat(verbose)
                    )
                )
        
        # if multiple modelfits are returned per subject, rearrange data so
        # each item of all_modelfits will contain one modelfit for each subject.
        # So for eliminative, all_modelfits[0] will have all n-stage subject
        # models.
        if isinstance(modelfits[0], list): #from a function or quick elim
            all_modelfits = []
            for est_idx in range(len(modelfits[0])):
                #each modelfit can also be eliminative or cumulative
                if isinstance(modelfits[0][est_idx], EliminativeMethod):
                    for est_idx2 in range(modelfits[0][est_idx].max_events):
                        all_modelfits.append([modelfit[est_idx].submodels[est_idx2+1] for modelfit in modelfits])
                elif isinstance(modelfits[0][est_idx], CumulativeMethod):
                    all_modelfits.append([modelfit[est_idx].submodels[-1] for modelfit in modelfits])
                elif isinstance(modelfits[0][est_idx], EventModel):                       
                    all_modelfits.append([modelfit[est_idx] for modelfit in modelfits])
        elif isinstance(modelfits[0], EliminativeMethod):
            all_modelfits = []
            for est_idx in range(modelfits[0].max_events):
                all_modelfits.append([modelfit.submodels[est_idx+1] for modelfit in modelfits])
        elif isinstance(modelfits[0], CumulativeMethod):
            all_modelfits = [[modelfit.submodels[-1] for modelfit in modelfits]]
        elif isinstance(modelfits[0], EventModel):
            all_modelfits = [modelfits]

        # Step 2, get loglikelihood from left out subjects
        print()
        all_likelihoods = []
        for est, modelfits in enumerate(all_modelfits):
            if verbose:
                print("Calculating likelihoods.")

            loocv = []
            if cpus_cv == 1:  # no mp for cross validation
                for didx, dim in enumerate(self.dimension_idx):
                    loocv.append(
                        self.loocv_loglikelihood(
                            data, 
                            dim, 
                            modelfits[didx], 
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
                            self.dimension_idx,
                            modelfits,
                            itertools.repeat(1), #cpus
                            itertools.repeat(verbose),
                        )
                    )

            #format results
            likelihoods = xr.DataArray(
                np.expand_dims(np.array(loocv).astype(np.float64), axis=0),
                dims=("n_event", self.dimension),
                coords={"n_event": np.array([modelfits[0].n_events]),
                    self.dimension: self.dimension_idx},
                name="loo_likelihood"
            )
            likelihoods.attrs['model_class'] = self.model_class
            likelihoods.attrs['model'] = self.model
            likelihoods.attrs['dimension'] = self.dimension
            likelihoods.attrs['n_events'] = np.array([modelfit.n_events for modelfit in modelfits])
            if not np.all(likelihoods.n_events==likelihoods.n_events[0]):
                likelihoods['n_event'] = 'variable'
            likelihoods.attrs['quick'] = self.quick
            
            modelfits_xr = likelihoods.copy()
            modelfits_xr.data = np.expand_dims(np.array(modelfits), axis=0)
            all_modelfits[est] = modelfits_xr
            all_likelihoods.append(likelihoods)

        #In case of eliminative, we can concat likelihoods and modelfits
        if self.model_class == EliminativeMethod:
            all_likelihoods = xr.concat(all_likelihoods, dim="n_event")
            all_likelihoods.attrs['n_event'] = np.zeros(all_likelihoods.data.shape).astype(int)
            for ev in range(all_likelihoods.data.shape[0]):
                all_likelihoods.attrs['n_event'][ev,:] = [modelfit.n_events for modelfit in all_modelfits[ev].data[0,:]]
            all_modelfits = xr.concat(all_modelfits, dim="n_event")
            all_modelfits.attrs['n_event'] = all_likelihoods.attrs['n_event']
        elif len(all_likelihoods) == 1:
            all_likelihoods = all_likelihoods[0]
        if len(all_modelfits) == 1:
            all_modelfits = all_modelfits[0]
        
        #average_modelfit = get_average_modelfit_eventprobs(data, all_modelfits)
        return all_likelihoods, all_modelfits #, average_modelfit, average_eventprobs
    

    def loocv_modelfit(self, data, dim, cpus_model=1, verbose=True):
        """Apply loocv estimation using either the provided model or function, while
        leaving out subject or recording 'dimension'

        Parameters 
        ----------
        data : Data to fit the model on. One of two options:
            1. BaseData
            2. PatternData object.
            In case of option 1, data is cross-correlated with the pattern in self.model.pattern.
            If using a 'function' for LOOCV (see init), it is recommended to provide PatternData
            or pattern to reduce RAM requirements.
        dim : str
            name of the dimension (of subject or recording) to leave out
        cpus_model : int
            nr of cpus to use for model estimation (nr of cpus for cross validation specified 
            above)
        verbose : bool

        Returns
        -------
        hmp model
            fitted hmp_model(s) on n-1 subjects/recordings
        """

        if verbose:
            print(f"\tEstimating model for all {self.dimension}s except {dim}")

        # Extract data without left-out dim
        if isinstance(data, PatternData):
            data = hmp.patterndata.get_subset(data, variable=self.dimension, values=self.dimension_idx[self.dimension_idx != dim])
        elif isinstance(data, BaseData):
            data = data.select_coord(variable=self.dimension, value=dim, method=lambda dim, variable: ~dim.isin(variable))
            if not hasattr(data,'projector') or isinstance(data.projector, Identity):
                data.pca_and_variance(**self.pca_kwargs)

        # Fit model on data
        if self.model_class == "function":
            fitted_model = self.model(data, **self.function_kwargs)
        else:
            if not self.quick: #same for Event and Eliminative
                fitted_model = copy.deepcopy(self.model)
                fitted_model.fit(data=data, cpus=cpus_model, verbose= False)
            else: #quick
                if self.model_class == hmp.models.EventModel: #use prev params
                    fitted_model = copy.deepcopy(self.model)
                    fitted_model.fit(data=data, channel_pars=fitted_model.channel_pars,\
                         time_pars=fitted_model.time_pars, cpus=cpus_model, verbose= False)
                elif self.model_class == hmp.models.EliminativeMethod: 
                #use prev params per model
                    fitted_model = []
                    for n_events in range(1, self.model.max_events + 1):
                        fitted_model.append(copy.deepcopy(self.model.submodels[n_events]))
                        fitted_model[-1].fit(data=data, \
                            channel_pars=fitted_model[-1].channel_pars, \
                            time_pars=fitted_model[-1].time_pars, cpus=cpus_model, \
                            verbose= False)

        #if pca_cv, attach pca weights to each model
        if self.pca_cv:
            if isinstance(fitted_model, hmp.models.EventModel):
                fitted_model.pca_weights = data.projector.weights
            elif isinstance(fitted_model, hmp.models.CumulativeMethod):
                fitted_model.submodels[-1].pca_weights = data.projector.weights
            elif isinstance(fitted_model, hmp.models.EliminativeMethod):
                for est_idx in range(fitted_model.max_events):
                    fitted_model.submodels[est_idx+1].pca_weights = data.projector.weights
            elif isinstance(fitted_model, list): #from a function or quick elim
                for est_idx in range(len(fitted_model)):
                    #each fitte can also be eliminative or cumulative
                    if isinstance(fitted_model[est_idx], hmp.models.EliminativeMethod):
                        for est_idx2 in range(fitted_model[est_idx].max_events):
                            fitted_model[est_idx].submodels[est_idx2+1].pca_weights = data.projector.weights
                    elif isinstance(fitted_model[est_idx], hmp.models.CumulativeMethod):
                        fitted_model[est_idx].submodels[-1].pca_weights = data.projector.weights
                    elif isinstance(fitted_model[est_idx], hmp.models.EventModel):
                        fitted_model[est_idx].pca_weights = data.projector.weights

        return fitted_model


    def loocv_loglikelihood(self, data, dim, modelfit, cpus_model=1, verbose=False):
        """Compute the log-likelihood of the fit.

        Calculate loglikelihood of fit on dim using parameters from modelfit,
        either using single model or level based model.

        Parameters
        ----------
        data : Data to fit the model on. One of two options:
            1. BaseData
            2. PatternData object.
            In case of option 1, data is cross-correlated with the pattern in self.model.pattern.
            If using a 'function' for LOOCV (see init), it is recommended to provide PatternData 
            or pattern to reduce RAM requirements.
        dim : str
            name of the dimension (subject or recording) to compute likelihood
        modelfit : xarray.Dataset
            modelfit that has parameters to apply.
        cpus_model : int
            Number of cpus to use to fit the models.
        verbose : bool

        Returns
        -------
        likelihood : float
            likelihood computed for the left-out dim
        """
        if verbose:
            print(f"\tCalculating likelihood for {self.dimension} {dim}")

        # Extract data of left-out dim
        if isinstance(data, PatternData):
            data = hmp.patterndata.get_subset(data, variable=self.dimension, values=dim)
        elif isinstance(data, BaseData):            
            data = data.select_coord(variable=self.dimension, value=dim)

            if not hasattr(data,'projector') or isinstance(data.projector, Identity): #pca_cv
                whiten = self.pca_kwargs.get('whiten', True)
                common_variance = self.pca_kwargs.get('common_variance', True)
                recording_zscore = self.pca_kwargs.get('recording_zscore', True)
                data.project(Custom(weights=modelfit.pca_weights))
                data.apply_variance_ops(whiten=whiten, common_variance=common_variance,
                                        recording_zscore=recording_zscore)

        # Calculate loglikelihood of model applied on data of participant
        likelihood, _ = modelfit.transform(data, cpus=cpus_model)

        if len(likelihood) == 1:
            return likelihood[0]
        else:
            return likelihood

def get_average_fit_eventprobs(modelfits, data: PatternData):
    """
    Return average modelfit(s) and eventprobs from list of fitted models.

    Parameters
    ----------
    modelfits : xr.DataArray
        output from LOOCV.fit(..), can be a 1d array or a 2d array. In the
        latter case, an average is provided for every row. If there are 
        modelfits with different numbers of events in a row, the average
        per event number is returned.
    data : PatternData
        Data to transform given average modelfit.
    """
    all_averages = []
    for row in modelfits:
        row = np.squeeze(row)
        ev_nrs = np.unique(row.n_events)
        all_averages_row = []
        for ev_nr in ev_nrs:
            subrow = row[row.n_events==ev_nr]
            average_modelfit = copy.deepcopy(subrow[0].item())
            if len(subrow) > 1:
                for subject in subrow[1:]:
                    average_modelfit.time_pars += subject.item().time_pars
                    average_modelfit.channel_pars += subject.item().channel_pars
                average_modelfit.time_pars /= len(subrow)
                average_modelfit.channel_pars /= len(subrow)
            all_averages_row.append(average_modelfit)
        if len(all_averages_row) == 1:
            all_averages_row = all_averages_row[0]
        all_averages.append(all_averages_row)

    all_eventprobs = []
    if isinstance(all_averages[0], list):
        all_eventprobs_row = []
        for row in all_averages:
            for average in row:
                _, eventprobs = average.transform(data)
                all_eventprobs_row.append(eventprobs)
            all_eventprobs.append(all_eventprobs_row)
    else:
        for average in all_averages:
            _, eventprobs = average.transform(data)
            all_eventprobs.append(eventprobs)
    
    return all_averages, all_eventprobs

### Example loocv functions
def example_simple_func(data, n_events, channel_pars=None, time_pars=None, verbose=False):
    """Example of a simple function that fits an n_event model.

    Note that this would normally not be done, as you could just provide
    this model directly to loocv(..). This is only an example of how a 
    function could be used, see below for a more realistic example.

    Example code
    -------------
    loocv_simple_func = hmp.loocv.LOOCV(example_simple_func, function_kwargs={'n_events' : 2})
    lkh_simple_func, modelfit_simple_func = loocv_simple_func.fit(pattern_data, cpus_cv=4, cpus_model=1)
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
    both models, so for both the likelihood will be computed.

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
    lkh_complex_func, modelfits_complex_func = loocv_complex_func.fit(pattern_data, \
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
                    channel_pars = eliminative_model.submodels[n_events].channel_pars,
                    time_pars = eliminative_model.submodels[n_events].time_pars)

    return [eliminative_model, group_model]