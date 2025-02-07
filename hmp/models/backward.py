"""Models to estimate event probabilities."""

import gc
import itertools
import multiprocessing as mp
from itertools import cycle, product
from warnings import resetwarnings, warn

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pandas import MultiIndex
from scipy.signal import correlate
from scipy.stats import norm as norm_pval

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

default_colors = ["cornflowerblue", "indianred", "orange", "darkblue", "darkgreen", "gold", "brown"]
from hmp.models.base import BaseModel
from hmp.models.fixedn import FixedEventModel

class BackwardEstimationModel(BaseModel):
    def fit(
        self,
        max_events=None,
        min_events=0,
        base_fit=None,
        max_starting_points=1,
        tolerance=1e-4,
        maximization=True,
        max_iteration=1e3,
        cpus=1,
    ):
        """Perform the backward estimation.

        First read or estimate max_event solution then estimate max_event - 1 solution by
        iteratively removing one of the event and pick the one with the highest
        loglikelihood

        Parameters
        ----------
        max_events : int
            Maximum number of events to be estimated, by default the output of
            hmp.models.hmp.compute_max_events()
        min_events : int
            The minimum number of events to be estimated
        base_fit : xarray
            To avoid re-estimating the model with maximum number of events it can be provided
            with this arguments, defaults to None
        max_starting_points: int
            how many random starting points iteration to try for the model estimating the maximal
            number of events
        tolerance: float
            Tolerance applied to the expectation maximization in the EM() function
        maximization: bool
            If True (Default) perform the maximization phase in EM() otherwhise skip
        max_iteration: int
            Maximum number of iteration for the expectation maximization in the EM() function
        """
        fixed_n_model = FixedEventModel(self.trial_data, self.events, self.distribution)

        if max_events is None and base_fit is None:
            max_events = self.compute_max_events()
        if not base_fit:
            if max_starting_points > 0:
                print(
                    f"Estimating all solutions for maximal number of events ({max_events}) with 1 "
                    "pre-defined starting point and {max_starting_points - 1} starting points"
                )
            event_loo_results = [
                fixed_n_model.fit(max_events, starting_points=max_starting_points, verbose=False)
            ]
        else:
            event_loo_results = [base_fit]
        max_events = event_loo_results[0].event.max().values + 1

        for n_events in np.arange(max_events - 1, min_events, -1):
            # only take previous model forward when it's actually fitting ok
            if event_loo_results[-1].loglikelihood.values != -np.inf:
                print(f"Estimating all solutions for {n_events} events")

                pars_prev = event_loo_results[-1].dropna("stage").parameters.values
                mags_prev = event_loo_results[-1].dropna("event").magnitudes.values

                events_temp, pars_temp = [], []

                for event in np.arange(n_events + 1):  # creating all possible solutions
                    events_temp.append(mags_prev[np.arange(n_events + 1) != event,])

                    temp_pars = np.copy(pars_prev)
                    temp_pars[event, 1] = (
                        temp_pars[event, 1] + temp_pars[event + 1, 1]
                    )  # combine two stages into one
                    temp_pars = np.delete(temp_pars, event + 1, axis=0)
                    pars_temp.append(temp_pars)

                if cpus == 1:
                    event_loo_likelihood_temp = []
                    for i in range(len(events_temp)):
                        event_loo_likelihood_temp.append(
                            fixed_n_model.fit(
                                n_events,
                                events_temp[i],
                                pars_temp[i],
                                tolerance=tolerance,
                                max_iteration=max_iteration,
                                maximization=maximization,
                                verbose=False,
                            )
                        )
                else:
                    inputs = zip(
                        itertools.repeat(n_events),
                        events_temp,
                        pars_temp,
                        itertools.repeat([]),
                        itertools.repeat([]),
                        itertools.repeat(tolerance),
                        itertools.repeat(max_iteration),
                        itertools.repeat(maximization),
                        itertools.repeat(1),
                        itertools.repeat(1),
                        itertools.repeat(True),
                        itertools.repeat(False),
                        itertools.repeat(1),
                    )
                    with mp.Pool(processes=cpus) as pool:
                        event_loo_likelihood_temp = pool.starmap(fixed_n_model.fit, inputs)

                lkhs = [x.loglikelihood.values for x in event_loo_likelihood_temp]
                event_loo_results.append(event_loo_likelihood_temp[np.nanargmax(lkhs)])

                # remove event_loo_likelihood
                del event_loo_likelihood_temp
                # Force garbage collection
                gc.collect()

            else:
                print(
                    f"Previous model did not fit well. Estimating a neutral {n_events} event model."
                )
                event_loo_results.append(
                    self.fixed_n_model(
                        n_events,
                        tolerance=tolerance,
                        max_iteration=max_iteration,
                        maximization=maximization,
                    )
                )
        event_loo_results = xr.concat(event_loo_results, dim="n_events", fill_value=np.nan)
        event_loo_results = event_loo_results.assign_coords(
            {"n_events": np.arange(max_events, min_events, -1)}
        )
        event_loo_results = event_loo_results.assign_attrs(method="backward")
        if "sp_parameters" in event_loo_results.attrs:
            del event_loo_results.attrs["sp_parameters"]
            del event_loo_results.attrs["sp_magnitudes"]
            del event_loo_results.attrs["maximization"]
        return event_loo_results
