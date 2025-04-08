"""Models to estimate event probabilities."""

from itertools import cycle
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm as norm_pval

from hmp.models.base import BaseModel
from hmp.models.fixedn import FixedEventModel

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

default_colors = ["cornflowerblue", "indianred", "orange", "darkblue", "darkgreen", "gold", "brown"]


class CumulativeEstimationModel(BaseModel):
    def __init__(self, *args, step=None, end=None, by_sample=False, pval=None, tolerance=1e-3,
                 **kwargs):
        self.step = step
        self.end = end
        self.by_sample = by_sample
        self.pval = pval
        self.tolerance = tolerance
        self.fitted_model = None
        super().__init__(*args, **kwargs)

    def fit(
        self,
        trial_data,
        verbose=True,
        diagnostic=False,
        cpus=1,
    ):
        """Fit the model starting with 1 event model.

        Instead of fitting an n event model this method starts by fitting a 1 event model
        (two stages) using each sample from the time 0 (stimulus onset) to the mean RT.
        Therefore it tests for the landing point of the expectation maximization algorithm given
        each sample as starting point and the likelihood associated with this landing point.
        As soon as a starting points reaches the convergence criterion, the function fits an n+1
        event model and uses the next samples in the RT as starting point for the following event

        Parameters
        ----------
        step: float
            The size of the step from 0 to the mean RT, defaults to the widths of
            the expected event.
        verbose: bool
            If True print information about the fit
        end: int
            The maximum number of samples to explore within each trial
        trace: bool
            If True keep the scale and magnitudes parameters for each iteration
        tolerance: float
            The tolerance used for the convergence in the EM() function
        diagnostic: bool
            If True print a diagnostic plot of the EM traces for each iteration and several
            statistics at each iteration
        return_estimates : bool
            return all intermediate models
        by_sample : bool
            try every sample as the starting point, even if a later event has already
            been identified. This in case the method jumped over a local maximum in an earlier
            estimation.
        pval: float
            p-value for the detection of the first event, test the first location for significance
            compared to a distribution of noise estimates

        Returns
        -------
                  A the fitted HMP mo
        """
        self.trial_data = trial_data
        end = trial_data.durations.mean() if self.end is None else self.end
        step = self.event_width_samples if self.step is None else self.step
        # if self.end is None:
            # end = self.mean_d
        # if step is None:
            # step = self.event_width_samples
        max_event_n = self.compute_max_events(trial_data) * 10  # not really nedded, if it fits it fits
        if diagnostic:
            cycol = cycle(default_colors)
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

        # The first new detected event should be higher than the bias induced by splitting the RT
        # in two random partition
        if self.pval is not None:
            lkh = fixed_n_model.fit(
                1, maximization=False, starting_points=100, return_max=False, verbose=False
            )
            lkh_prev = lkh.loglikelihood.mean() + lkh.loglikelihood.std() * norm_pval.ppf(1 - self.pval)
        else:
            lkh_prev = -np.inf
        # if return_estimates:
            # estimates = []  # store all n_event solutions

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
            print(mags_props.shape)
            print(pars_prop.shape)

            # Estimate model based on these propositions
            likelihoods, event_probs = fixed_n_model.fit(
                trial_data,
                mags_props,
                pars_prop,
                verbose=False,
                cpus=cpus,
            )
            sol_sample_new_event = int(
                np.round(
                    self.scale_to_mean(
                        np.sum(fixed_n_model.parameters.values[:n_events, 1]), self.shape
                    )
                )
            )

            # Diagnostic plot
            if diagnostic:
                self.plot_diagnosis(fixed_n_model)

            # check solution
            if likelihoods - lkh_prev > 0:  # accept solution if likelihood improved
                lkh_prev = likelihoods

                # update mags, params,
                mags[:n_events] = fixed_n_model.magnitudes.values
                pars[: n_events + 1] = fixed_n_model.parameters.values

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
                # if diagnostic:
                    # color = next(cycol)
                    # plt.plot(solutions.traces.T, c=color, label=f"n-events {n_events - 1}")
                if verbose:
                    print(
                        f"Transition event {n_events - 1} found around sample "
                        f"{sol_sample_new_event}"
                    )

            else:  # reject solution, search on
                prev_sample = int(
                    np.round(self.scale_to_mean(np.sum(pars[: n_events - 1, 1]), self.shape))
                )
                # find furthest explored param. Note: this also work by_sample
                # just a tiny bit faster this way
                if not self.by_sample:
                    max_scale = np.max(
                        [np.sum(x[:n_events, 1]) for x in fixed_n_model.param_dev.values]
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
        # if diagnostic:
        #     plt.ylabel("Log-likelihood")
        #     plt.xlabel("EM iteration")
        #     plt.legend()
        mags = mags[:n_events, :]
        pars = pars[: n_events + 1, :]
        if n_events > 0:
            self.fitted_model = fixed_n_model.fit(
                self.events,
                n_events=n_events,
                parameters=np.array([pars]),
                magnitudes=np.array([mags]),
                verbose=verbose,
                cpus=1,
            )
            self._fitted = True
            # fit = fit.assign_attrs(step=step, by_sample=int(self.by_sample))
            # fit = fit.assign_attrs(method="fit", step=step, by_sample=int(self.by_sample))
        else:
            warn("Failed to find more than two stages, returning None")
            self._fitted = False
        # del fit.attrs["sp_parameters"]
        # del fit.attrs["sp_magnitudes"]
        # del fit.attrs["maximization"]
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


    def plot_diagnosis(self, solutions):
        plt.plot(solutions.traces.T, alpha=0.3, c="k")
        print()
        print("Event found at sample " + str(sol_sample_new_event))
        events_at = np.round(self.scale_to_mean(
                            np.cumsum(solutions.parameters.values[:, 1]),
                            self.shape)).astype(int)
        print(
            f"Events at {events_at}"
        )
        print("lkh change: " + str(solutions.loglikelihood.values - lkh_prev))
