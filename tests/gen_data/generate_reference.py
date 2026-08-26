"""Capture reference values of the numpy likelihood, for checking reimplementations.

Any other implementation of the likelihood, in particular a differentiable one,
has to reproduce these. Values are computed in float64 so that agreement can be
demanded at a tolerance set by precision rather than by what happens to pass.

Run from the repository root:  python tests/gen_data/generate_reference.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_io import init_data

import hmp
from hmp.models import EventModel
from hmp.patterndata import PatternData

OUT = Path(__file__).resolve().parent / "likelihood_reference.npz"


def parameter_grid(model, pdata, n_events, n_dims):
    """Parameters to evaluate at, spanning more than the neighbourhood of the fit."""
    mean_duration = float(pdata.durations.mean())
    shape = model.distribution.shape
    even = model.distribution.mean_to_scale(mean_duration / (n_events + 1))
    rng = np.random.default_rng(0)

    cases = {}

    def add(name, channel_pars, time_pars):
        cases[name] = (np.asarray(channel_pars, dtype=np.float64),
                       np.asarray(time_pars, dtype=np.float64))

    zeros = np.zeros((n_events, n_dims))
    flat = np.column_stack([np.full(n_events + 1, shape), np.full(n_events + 1, even)])

    add("zero_channel_even_time", zeros, flat)
    add("small_channel", rng.normal(0, 0.05, (n_events, n_dims)), flat)
    add("large_channel", rng.normal(0, 2.0, (n_events, n_dims)), flat)

    # scales pushed short and long, the second driving pmf mass past max_duration
    short = flat.copy()
    short[:, 1] = model.distribution.mean_to_scale(mean_duration / (n_events + 1) / 4)
    add("short_scales", rng.normal(0, 0.5, (n_events, n_dims)), short)

    long_ = flat.copy()
    long_[:, 1] = model.distribution.mean_to_scale(mean_duration * 2)
    add("long_scales", rng.normal(0, 0.5, (n_events, n_dims)), long_)

    uneven = flat.copy()
    uneven[:, 1] = model.distribution.mean_to_scale(
        mean_duration / (n_events + 1) * np.linspace(0.5, 1.5, n_events + 1)
    )
    add("uneven_scales", rng.normal(0, 0.5, (n_events, n_dims)), uneven)

    return cases


def main():
    """Evaluate the grid and write the reference file."""
    _, _, epoch_data, _, _, n_events_default = init_data()
    hmp_data = hmp.basedata.default(
        epoch_data, n_comp=3, center=True, duration_id="response_time"
    )
    # all trials, so trial lengths differ, in float64 so precision is not the variable
    pdata = PatternData.from_basedata(hmp_data, dtype=np.float64)
    n_dims = pdata.cross_corr.shape[1]
    durations = pdata.durations.values

    saved = {
        "cross_corr": np.asarray(pdata.cross_corr),
        "starts": np.asarray(pdata.starts),
        "ends": np.asarray(pdata.ends),
        "durations": np.asarray(durations),
        "sfreq": np.asarray(pdata.sfreq),
    }
    manifest = []

    for n_events in (1, 2, 3):
        model = EventModel(n_events=n_events)
        model.n_dims = n_dims
        # must be a boolean mask: estim_probs reinterprets an index array of this
        # length as booleans, which would silently drop trial 0
        subset = np.ones(len(durations), dtype=bool)
        for name, (channel_pars, time_pars) in parameter_grid(
            model, pdata, n_events, n_dims
        ).items():
            key = f"n{n_events}_{name}"
            inter = {}
            lkh, _, trial_lkh = model.estim_probs(
                pdata, channel_pars, time_pars, subset_epochs=subset, intermediates=inter
            )
            saved[f"{key}__channel_pars"] = channel_pars
            saved[f"{key}__time_pars"] = time_pars
            saved[f"{key}__locations_samples"] = np.asarray(inter["locations_samples"])
            for field in ("gains", "pmf", "forward", "backward",
                          "eventprobs_unnormalised", "trial_likelihood"):
                saved[f"{key}__{field}"] = np.asarray(inter[field])
            saved[f"{key}__likelihood"] = np.asarray(lkh)
            manifest.append(key)
            print(f"{key:32s} lkh={lkh: .10f}  finite={np.isfinite(lkh)}  "
                  f"clipped={inter['n_clipped']}")

    saved["cases"] = np.array(manifest)
    np.savez_compressed(OUT, **saved)
    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1024:.1f} KiB, {len(manifest)} cases)")


if __name__ == "__main__":
    main()
