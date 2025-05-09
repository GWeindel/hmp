"""Models to estimate event probabilities."""

from warnings import warn

import numpy as np
from scipy.stats import norm as norm_pval

from hmp.models.base import BaseModel
from hmp.models.fixedn import FixedEventModel
from hmp.trialdata import TrialData

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

default_colors = ["cornflowerblue", "indianred", "orange", "darkblue", "darkgreen", "gold", "brown"]


class CumulativeEstimationModel(BaseModel):
    def __init__(self, *args, step=None, end=None, by_sample=False, tolerance=1e-4, fitted_model_tolerance=1e-4,
                 **kwargs):
        """Fit the model starting with 1 event model.

        Instead of fitting an n event model this method starts by fitting a 1 event model
        (two stages) using each sample from the time 0 (stimulus onset) to the mean RT.
        Therefore it tests for the landing point of the expectation maximization algorithm given
        each sample as starting point and the likelihood associated with this landing point.
        As soon as a starting points reaches the convergence criterion, the function fits an n+1
        event model and uses the next samples in the RT as starting point for the following event

        Parameters
        ----------
        args:
            Extra arguments to be passed through to the BaseModel, at least events and distribution
            objects.
        step: float
            The size of the step from 0 to the mean RT, defaults to the widths of
            the expected event.
        end: int
            The maximum number of samples to explore within each trial
        tolerance: float
            The tolerance used for the convergence in the EM() function for the cumulative step
        fitted_model_tolerance: float
            The tolerance used for the final model
        by_sample : bool
            try every sample as the starting point, even if a later event has already
            been identified. This in case the method jumped over a local maximum in an earlier
            estimation.
        kwargs:
            Keyword estimates to be passed on to the BaseModel.
        """
        self.step = step
        self.end = end
        self.by_sample = by_sample
        self.tolerance = tolerance
        self.fitted_model_tolerance = tolerance if fitted_model_tolerance is None else fitted_model_tolerance

        self.fitted_model = None
        super().__init__(*args, **kwargs)

    def fit(
        self,
        trial_data: TrialData,
        verbose: bool = True,
        cpus: int = 1,
    ):
        """Fit the model starting with 1 event model.

        Parameters
        ----------
        trial_data:
            Trial data to fit the data on.
        verbose:
            Set to True for more detail on what is happening.
        cpus:
            Number of cpu cores to be used for the computation.
        """
        self.trial_data = trial_data
        end = trial_data.durations.mean() if self.end is None else self.end
        step = self.event_width_samples if self.step is None else self.step

        max_event_n = self.compute_max_events(trial_data)

        pbar = tqdm(total=int(np.rint(end)))  # progress bar
        n_events, j, time = 1, 1, 0  # j = sample after last placed event
        # Init pars (need this for min_model)
        pars = np.zeros((max_event_n + 1, 2))
        pars[:, 0] = self.shape  # final gamma parameters during estimation, shape x scale
        pars_prop = pars[: n_events + 1].copy()  # gamma params of current estimation
        pars_prop[0, 1] = self.mean_to_scale(
            j * step, self.shape
        )  # initialize gamma_parameters at 1 sample
        last_stage = self.mean_to_scale(end - j * step, self.shape)  # remainder of time
        pars_prop[-1, 1] = last_stage

        # Init mags
        mags = np.zeros((max_event_n, trial_data.n_dims))  # final mags during estimation

        fixed_n_model = FixedEventModel(self.events, self.distribution, tolerance=self.tolerance, n_events=n_events)

        lkh_prev = -np.inf

        # Iterative fit
        while (
            self.scale_to_mean(last_stage, self.shape) >= self.location and n_events <= max_event_n
        ):
            prev_time = time
            fixed_n_model = FixedEventModel(self.events, self.distribution, tolerance=self.tolerance, n_events=n_events)
            # get new parameters
            mags_props, pars_prop = self.propose_fit_params(
                trial_data,
                n_events, self.by_sample, step, j, mags, pars, end
            )
            last_stage = pars_prop[n_events, 1]
            pars_prop = np.array([pars_prop])

            # Estimate model based on these propositions
            fixed_n_model.fit(
                trial_data,
                np.array([mags_props]),
                np.array([pars_prop]),
                verbose=False,
                cpus=cpus,
            )
            sol_sample_new_event = int(
                np.round(
                    self.scale_to_mean(
                        np.sum(fixed_n_model.parameters[0, :n_events, 1]), self.shape
                    )
                )
            )
            likelihoods = fixed_n_model.lkhs.sum()
            # check solution
            if likelihoods - lkh_prev > 0:  # accept solution if likelihood improved
                lkh_prev = likelihoods

                # update mags, params,
                mags[:n_events] = fixed_n_model.magnitudes
                pars[: n_events + 1] = fixed_n_model.parameters

                # search for an additional event, starting again at sample 1 from prev event,
                # or next sample if by_sample
                n_events += 1
                if self.by_sample:
                    j += 1
                    time = j * step
                else:
                    j = 1
                    time = sol_sample_new_event + j * step

                # Diagnostic plot
                if verbose:
                    print(
                        f"Transition event {n_events - 1} found around time "
                        f"{sol_sample_new_event*(1000/self.sfreq)}"
                    )

            else:  # reject solution, search on
                prev_sample = int(
                    np.round(self.scale_to_mean(np.sum(pars[: n_events - 1, 1]), self.shape))
                )
                # find furthest explored param. Note: this also work by_sample
                # just a tiny bit faster this way
                if not self.by_sample:
                    max_scale = np.max(
                        [np.sum(x[:n_events, 1]) for x in fixed_n_model.param_dev]
                    )
                    max_sample = int(np.round(self.scale_to_mean(max_scale, self.shape)))
                    j = (
                        np.max([max_sample - prev_sample + 1, (j + 1) * step]) / step
                    )  # either ffwd to furthest explored sample or add 1 to j
                    time = prev_sample + j * step
                else:
                    j += 1
                    time = j * step

            pbar.update(int(np.rint(time - prev_time)))

        # done estimating

        n_events = n_events - 1
        if verbose:
            print()
            print("All events found, refitting final combination.")

        mags = mags[:n_events, :]
        pars = pars[: n_events + 1, :]

        self.fitted_model = FixedEventModel(
            self.events, self.distribution, tolerance=self.fitted_model_tolerance, n_events=n_events)
        if n_events > 0:
            self.fitted_model.fit(
                trial_data,
                parameters=np.array([[pars]]),
                magnitudes=np.array([[mags]]),
                verbose=verbose,
                cpus=1,
            )
            self._fitted = True

        else:
            warn("Failed to find more than two stages, returning None")
            self._fitted = False
        pbar.update(int(np.rint(end) - int(np.rint(time))))

    def transform(self, *args, **kwargs):
        self._check_fitted("transform data")
        self.fitted_model.transform(*args, **kwargs)

    def propose_fit_params(self, trial_data, n_events, by_sample, step, j, mags, pars, end):
        if (
            by_sample and n_events > 1
        ):  # go through the whole range sample-by-sample, j is sample since start
            scale_j = self.mean_to_scale(step * j, self.shape)

            # New parameter proposition
            pars_prop = pars[:n_events].copy()  # pars so far
            n_event_j = np.argwhere(scale_j > np.cumsum(pars_prop[:, 1])) + 2  # counting from 1
            n_event_j = np.max(n_event_j) if len(n_event_j) > 0 else 1
            n_event_j = np.min([n_event_j, n_events])  # do not insert even after last stage

            # insert j at right spot, subtract prev scales
            pars_prop = np.insert(
                pars_prop,
                n_event_j - 1,
                [self.shape, scale_j - np.sum(pars_prop[: n_event_j - 1, 1])],
                axis=0,
            )
            # subtract inserted scale from next event
            pars_prop[n_event_j, 1] = pars_prop[n_event_j, 1] - pars_prop[n_event_j - 1, 1]
            last_stage = self.mean_to_scale(end, self.shape) - np.sum(pars_prop[:-1, 1])
            pars_prop[n_events, 1] = last_stage
            mags_props = np.zeros((1, n_events, trial_data.n_dims))  # always 0?
            mags_props[:, : n_events - 1, :] = np.tile(
                mags[: n_events - 1, :], (len(mags_props), 1, 1)
            )
            # shift new event to correct position
            mags_props = np.insert(
                mags_props[:, :-1, :], n_event_j - 1, mags_props[:, -1, :], axis=1
            )

        else:
            # New parameter proposition
            pars_prop = pars[: n_events + 1].copy()
            pars_prop[n_events - 1, 1] = self.mean_to_scale(step * j, self.shape)
            last_stage = self.mean_to_scale(end, self.shape) - np.sum(pars_prop[:-1, 1])
            pars_prop[n_events, 1] = last_stage

            mags_props = np.zeros((1, n_events, trial_data.n_dims))  # always 0?
            mags_props[:, : n_events - 1, :] = np.tile(
                mags[: n_events - 1, :], (len(mags_props), 1, 1)
            )

        # in edge cases scale can get negative, make sure that doesn't happen:
        pars_prop[:, 1] = np.maximum(pars_prop[:, 1], self.mean_to_scale(1, self.shape))

        return mags_props, pars_prop

    def __getattribute__(self, attr):
        property_list = {
            "xrtraces": "get traces",
            "xrlikelihoods": "get likelihoods",
            "xrparam_dev": "get dev params",
            "xrmags": "get xrmags",
            "xrparams": "get xrparams"
        }
        if attr in property_list:
            self._check_fitted(property_list[attr])
            return getattr(self.fitted_model, attr)
        return super().__getattribute__(attr)
