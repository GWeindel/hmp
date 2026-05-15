"""Leave-one out cross-validation method

Class
-------
LOOCV


"""

from typing import Any
import inspect

import hmp
from hmp.trialdata import TrialData


class LOOCV():
    """LOOCV class. Can be initialized with a (fitted) model or a self-defined function.

    Parameters TO BE UPDATED
    ----------
    model : Any 
        either a (fitted) model or a self-defined function
   
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
        model: Any
    ):
        #instance of an hmp model
        if isinstance(model, hmp.models.base.BaseModel):
            self.model_class = type(model)
            self.model = model
        
        #assume it's a function that will return a fitted hmp model
        elif inspect.isfunction(model): 
            self.model_class = "function"
        
        #unknown
        else:
            raise ValueError(f"Unknown model definition {model}, aborting.")  

        #EventModel
        #if self.model_class == hmp.models.EventModel:
        #    if self.parametrized:
        #        self.n_events = model.n_events


    def fit(
        self,
        trial_data: TrialData,
        quick = False,
        cpus: int = 1,
        verbose = False):

        """fit LOOCV

        """

        #fit model
        if self.model_class == hmp.models.EventModel:
            modeltorun = self.model_class(n_events=self.model.n_events)
            lkh_3ev, _ = modeltorun.fit_transform(trial_data, cpus=cpus)
        else:
            print("don't know what to do yet :)")
            lkh_3ev = None

        return lkh_3ev
