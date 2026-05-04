"""Models to estimate cumulative event models."""

from warnings import warn

import numpy as np

from hmp.models.base import BaseModel
from hmp.models.event import EventModel
from hmp.trialdata import TrialData

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

default_colors = ["cornflowerblue", "indianred", "orange", "darkblue", "darkgreen", "gold", "brown"]


class CumulativeMethod(BaseModel):
    """Initialize the CumulativeMethod.

    This method initializes the model and sets up parameters for fitting a cumulative event model.
    The fitting process starts with a 1-event model and iteratively adds events based on the
    convergence of the expectation maximization algorithm.

    Parameters
    ----------
    args : tuple
        Extra arguments to be passed to the BaseModel, including at least events and
        distribution objects.
    step : float, optional
        The size of the step from 0 to the mean RT. Defaults to the location defined in the pattern.
        Small values ensure a complete exploration of the parameter space but can be slow.
        Higher values fasten the estimation but risk missing event due to unexplored spaces.
    end : int, optional
        The maximum number of samples to explore within each trial. Defaults to None.
    fastforward : bool, optional
        If True cumulatively estimates HMP models by looking for a new event after the
        last event that improved likelihood. This fastforward version uses shortcuts to speed
        estimation but can miss events in-between two events.
        If False (Default) iteratively test all samples from `start` to `end`, retain times at
        which likelihood increased regardless of whether a subsequent event was found. The N+1
        solution only retains the time parameters found with the N solution.
    tolerance : float, optional
        The tolerance used for convergence in the EM() function for the cumulative step.
        Defaults to 1e-4.
    kwargs : dict
        Additional keyword arguments to be passed to the BaseModel.
    """

    def __init__(
        self,
        *args,
        step: float = None,
        end: int = None,
        fastforward: bool = True,
        tolerance: float = 1e-4,
        **kwargs,
    ):
        self.step = step
        self.end = end
        self.fastforward = fastforward
        self.tolerance = tolerance
        self.submodels = []
        super().__init__(*args, **kwargs)

    def fit(
        self,
        trial_data: TrialData,
        verbose: bool = True,
        cpus: int = 1,
    ) -> None:
        """
        Fit the model starting with a 1-event model and iteratively add events.

        This method fits the cumulative event model to the provided trial data. It begins with a
        single-event model and incrementally adds events based on the convergence of the expectation
        maximization algorithm. The process continues until the maximum number of events (given the
        minimum duration) is reached or the likelihood no longer improves.

        Parameters
        ----------
        trial_data : TrialData
            The trial data to fit the model on.
        verbose : bool, optional
            If True, provides detailed output about the fitting process. Defaults to True.
        cpus : int, optional
            The number of CPU cores to use for computation. Defaults to 1.

        Returns
        -------
        None
        """
        self.trial_data = trial_data
        end = trial_data.durations.mean() if self.end is None else self.end
        self.step = self.location if self.step is None else self.step
        #stop when not possible to insert event
        end = int(np.rint((end - self.location*2)/self.step))

        pbar = tqdm(total=end)  # progress bar
        n_events, j = 1, 1 # j = sample after last placed event

        # final time parameters during estimation, shape x scale
        time_pars = np.zeros((end, 2))
        time_pars[:, 0] = self.distribution.shape

        # Initialize last stage of n=1
        time_pars[0, 1] = self.distribution.mean_to_scale(trial_data.durations.mean())

        # final channel_pars during estimation
        channel_pars = np.zeros((end, trial_data.n_dims))
        lkh_prev = -np.inf

        # Iterative fit
        while j < end:
            prev_j = j
            event_model = EventModel(self.pattern, self.distribution, tolerance=self.tolerance,
                                     n_events=n_events)
            # get new parameters
            channel_pars_props, time_pars_props = self._propose_fit_params(
                n_events, j, channel_pars, time_pars
            )

            # Estimate model based on these propositions
            event_model.fit(
                trial_data,
                np.array([channel_pars_props]),
                np.array([time_pars_props]),
                verbose=False,
                cpus=cpus,
            )

            likelihoods = event_model.lkhs.sum()
            # check solution
            if likelihoods - lkh_prev > 0:  # accept solution if likelihood improved
                lkh_prev = likelihoods
                self.submodels.append(event_model)

                # update channel_pars, params,
                channel_pars[:n_events] = event_model.channel_pars
                time_pars[: n_events + 1] = event_model.time_pars

                if verbose:
                    # Just to track advancement
                    events_so_far = [int(np.round(self.distribution.scale_to_mean(x))
                                         *(1000/self.sfreq))
                                         for x in
                                     np.cumsum(event_model.time_pars[0, :n_events, 1])
                    ]
                    print(
                        f"{n_events} events found around times "
                        f"{events_so_far}"
                    )
                # Search for additional event
                n_events += 1

            if self.fastforward:
                # # If ffwd, the next sample tested follows the last explored time
                max_scale = np.max(
                    [np.sum(x[0, :n_events-1, 1]) for x in event_model.time_pars_dev]
                )
                max_sample = int(np.round(self.distribution.scale_to_mean(max_scale)))
                j = np.max([max_sample, (j + 1) * self.step]) / self.step
            else:
                j += 1
            pbar.update(int(np.rint(j-prev_j)))
        pbar.update(int(np.rint(end -j)+1))

        # done estimating
        n_events = n_events - 1
        if n_events > 0:
            if verbose:
                print(f"Found {n_events} events")
            self._fitted = True
        else:
            warn("Failed to find more than two stages, returning None")
            self._fitted = False

    def transform(self, *args, **kwargs):
        """
        Transform the input data using the last model fitted in the cumulative method.

        This method applies the transformation defined by the final model to the provided data.

        Returns
        -------
        Transformed data as returned by the final model's transform method.

        """
        self._check_fitted("transform data")
        return self.submodels[-1].transform(*args, **kwargs)

    def _propose_fit_params(self, n_events, j, channel_pars, time_pars):

        if (
            not self.fastforward and n_events > 1
        ):  # go through the whole range sample-by-sample, j is sample since start
            scale_j = self.distribution.mean_to_scale(self.step * j)

            # New parameter proposition
            time_pars_props = time_pars[:n_events].copy()  # time_pars so far
            # look between which event the next proposition should go
            n_event_j = np.argwhere(scale_j >= np.cumsum(time_pars_props[:, 1])) + 2
            n_event_j = np.max(n_event_j) if len(n_event_j) > 0 else 1
            n_event_j = np.min([n_event_j, n_events])  # do not insert even after last stage

            # insert j at right spot, subtract prev scales
            time_pars_props = np.insert(
                time_pars_props,
                n_event_j - 1,
                [self.distribution.shape, scale_j - np.sum(time_pars_props[: n_event_j - 1, 1])],
                axis=0,
            )
            # subtract inserted scale from next event
            time_pars_props[n_event_j, 1] = (time_pars_props[n_event_j, 1]
                                             - time_pars_props[n_event_j - 1, 1])
            channel_pars_props = np.zeros((1, n_events, channel_pars.shape[-1]))  # always 0
        else:
            time_pars_props = time_pars[: n_events + 1].copy()
            time_pars_props[n_events,1] = time_pars_props[n_events-1,1]
            # New parameter proposition for the new event based on previous run
            new_event_prop = (np.max([self.distribution.mean_to_scale(j * self.step) -
                np.sum(time_pars_props[:n_events-1, 1]),self.distribution.mean_to_scale(self.step)])
            )
            time_pars_props[n_events-1, 1] = new_event_prop
            # Subtract new proposition from last stage
            time_pars_props[-1, 1] -= new_event_prop

            # Add a neutral event as new proposition
            channel_pars_props = np.zeros((1, n_events, channel_pars.shape[-1]))
            channel_pars_props[:, :n_events-1, :] = channel_pars[:n_events-1]

        return channel_pars_props, np.array([time_pars_props])

    def __getattribute__(self, attr):
        property_list = {
            "xrtraces": "get traces",
            "xrlikelihoods": "get likelihoods",
            "xrtime_pars_dev": "get dev time time_pars",
            "xrchannel_pars": "get xrchannel_pars",
            "xrtime_pars": "get xrtime_pars"
        }
        if attr in property_list:
            self._check_fitted(property_list[attr])
            return getattr(self.submodels[-1], attr)
        return super().__getattribute__(attr)
