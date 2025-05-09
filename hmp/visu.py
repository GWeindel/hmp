"""Module containing functions to visualize the results of the HMP model."""

from itertools import cycle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ssignal
import xarray as xr
from scipy import stats

from hmp.utils import event_times, event_topo

default_colors = ["cornflowerblue", "indianred", "orange", "darkblue", "darkgreen", "gold"]


def plot_topo_timecourse(
    epoch_data,
    estimates,
    channel_position,
    figsize=None,
    dpi=100,
    magnify=1,
    times_to_display='all',
    cmap="Spectral_r",
    ylabels=[],
    xlabel=None,
    max_time=None,
    vmin=None,
    vmax=None,
    title=False,
    ax=None,
    sensors=False,
    contours=6,
    event_lines="tab:orange",
    colorbar=True,
    topo_size_scaling=False,
    as_time=False,
    linecolors="black",
    estimate_method=None,
    combined=False
):
    """
    Plot the event topographies at the average time of the onset of the next stage.

    Parameters
    ----------
    epoch_data : xr.DataArray
        The original EEG data in HMP format
    estimates : hmp object
        the result from a fitted hmp
    channel_position : ndarray
        Either a 2D array with dimension channel and [x,y] storing channel
        location in meters or an info object from the mne package containing
        digit points for channel location
    figsize : list | tuple | ndarray
        Length and heigth of the matplotlib plot
    dpi : float
        DPI of the  matplotlib plot
    magnify : float
        How much should the events be enlarged, useful to zoom on topographies,
        providing any other value than 1 will however change the displayed size
        of the event
    times_to_display : ndarray
        Times to display (e.g. Reaction time or any other relevant time)
        in the time unit of the fitted data, if 'all' plots the times of all events
    cmap : str
        Colormap of matplotlib
    xlabel : str
        label of x-axis, default = None, which give "Time (samples)"
        or "Time (ms)" in case as_time = True
    ylabels : dict
        dictonary with {label_name : label_values}. E.g. {'Condition': ['Speed','Accuracy']}
    max_time : float
        limit of the x (time) axe
    vmin : float
        Colormap limits to use (see https://mne.tools/dev/generated/mne.viz.plot_topomap.html).
        If not explicitly set, uses min across all topos while keeping colormap symmetric.
    vmax : float
        Colormap limits to use (see https://mne.tools/dev/generated/mne.viz.plot_topomap.html).
        If not explicitly set, uses max across all topos while keeping colormap symmetric.
    title : str
        title of the plot
    ax : matplotlib.pyplot.ax
        Matplotlib object on which to draw the plot, can be useful if you want
        to control specific aspects of the plots outside of this function
    sensors : bool
        Whether to plot the sensors on the topographies
    skip_channel_contribution: bool
        if True assumes that the provided channel argument is already topographies for each channel
    contours : int / array_like
        The number of contour lines to draw (see https://mne.tools/dev/generated/mne.viz.plot_topomap.html)
    event_lines : bool / color
        Whether to plot lines and shading to indicate the moment of the event.
        If True uses tab:orange, if set as color, uses the color
    colorbar : bool
        Whether a colorbar is plotted.
    topo_size_scaling : bool
        Whether to scale the size of the topographies with the event size. If True,
        size of topographies depends on total plotted time interval, if False it is
        only dependent on magnify.
    as_time : bool
        if true, plot time (ms) instead of samples. Ignored if times are provided as array.
    estimate_method : string
        'max' or 'mean', either take the max probability of each event on each trial,
        or the weighted average.
    combined: bool
        Whether to combine levels by averaging across them (True) or plot each level (False, default)

    Returns
    -------
    ax : matplotlib.pyplot.ax
        if ax was specified otherwise returns the plot
    """
    from mne import Info
    from mne.viz import plot_brain_colorbar, plot_topomap
    # if estimates is an fitted HMP instance, calculate topos and times
    assert "event" in estimates.dims
    sfreq = epoch_data.sfreq
    estimates = estimates.copy()
    # Stacking is necessary to retain the common indices, otherwise absent trials are just Nan'd out
    if "trial_x_participant" not in epoch_data.dims:
        epoch_data = epoch_data.rename({"epochs": "trials"}).stack(
            trial_x_participant=["participant", "trials"]
        )
    common_trials = np.intersect1d(
        estimates["trial_x_participant"].values, epoch_data["trial_x_participant"].values
    )
    estimates = estimates.sel(trial_x_participant=common_trials)
    epoch_data = (
        epoch_data.sel(trial_x_participant=common_trials).unstack().rename({"trials": "epochs"})
    )

    if xlabel is None:
        if as_time:
            xlabel = "Time (ms)"
        else:
            xlabel = "Time (in samples)"

    # set color of event_lines
    if event_lines:
        event_color = "tab:orange"
    else:
        event_color = event_lines

    # extract relevant info from estimates to pot 
    channel_data = event_topo(
        epoch_data, estimates,
        estimate_method=estimate_method
    ).data  # compute topographies
    times = event_times(
        estimates,
        mean=True,
        as_time=as_time,
        add_rt=True,
        estimate_method=estimate_method,
    ).data  # compute corresponding times
    if combined:
        times = np.array([np.mean(times, axis=0)])
        levels = [0]
    else:
        levels = np.unique(estimates.levels)
    n_level = len(np.unique(levels))

    # reverse order, to make correspond to level maps
    channel_data = np.flip(channel_data, axis=1)
    times =  np.flip(times, axis=0)
    
    # Time/samples
    if as_time:
        time_step = 1000 / sfreq  # time_step still needed below
    else:
        time_step = 1
    event_size = estimates.event_width_samples * time_step

    # fix vmin/vmax across topos, while keeping symmetric
    if vmax == None:  # vmax = absolute max, unless no positive values
        vmax = np.nanmax(np.abs(channel_data[:]))
        vmin = -vmax if np.nanmin(channel_data[:]) < 0 else 0
        if np.nanmax(channel_data[:]) < 0:
            vmax = 0

    # Time to display
    if times_to_display is None:
        times_to_display = np.array([np.nanmax(times[:, -1])])
    elif isinstance(times_to_display, str) and times_to_display == 'all':
        times_to_display = times
    if len(times_to_display.shape) == 1:
        times_to_display = [times_to_display] * n_level
    
    # based the size of the topographies on event_size and magnify or only on magnify
    if topo_size_scaling:  # does topo width scale with time interval of plot?
        topo_size = event_size * magnify
    else:
        timescale = (
            max_time if max_time else (np.nanmax(times_to_display) * 1.05)
        ) + time_step
        topo_size = 0.08 * timescale * magnify  # 8% of time scale

    # set ylabels to level
    if ylabels == []:
        ylabels = np.arange(n_level)#estimates.clabels
    return_ax = True
    
    # make axis
    if ax is None:
        if figsize is None:
            figsize = (12, n_level * 0.7* np.max([magnify, 1.8]))  # make sure they don't get too flat
        _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        return_ax = False

    axes = []
    # plot row by row
    rowheight = 1 / n_level
    n_event = estimates.event.max()+1
    for i, level in enumerate(levels):
        times_level = times[i]
        missing_evts = np.where(np.isnan(times_level))[0]
        times_level = np.delete(times_level, missing_evts)
        channel_data_ = channel_data[:, i, :]
        channel_data_ = np.delete(channel_data_, missing_evts, axis=1)
        ylow = i * rowheight
        # plot topography per event
        for event in np.arange(n_event):
            if event not in missing_evts:
                # topography
                axes.append(
                    ax.inset_axes(
                        [
                            times_level[event] - topo_size / 2,
                            ylow + 0.1 * rowheight,
                            topo_size,
                            rowheight * 0.8,
                        ],
                        transform=ax.get_xaxis_transform(),
                    )
                )
                plot_topomap(
                    channel_data_[:, event],
                    channel_position,
                    axes=axes[-1],
                    show=False,
                    cmap=cmap,
                    vlim=(vmin, vmax),
                    sensors=sensors,
                    contours=contours,
                )
    
                # lines/fill of detected event
                if event_lines:
                    # bottom of row + 5% if n_level > 1
                    ylow2 = (
                        level * rowheight
                        if n_level == 1
                        else level * rowheight + 0.05 * rowheight
                    )
                    # top of row - 5% if n_level > 1
                    yhigh = (
                        (level + 1) * rowheight
                        if n_level == 1
                        else (level + 1) * rowheight - 0.05 * rowheight
                    )
    
                    ax.vlines(
                        times_level[event] - event_size / 2,
                        ylow2,
                        yhigh,
                        linestyles="dotted",
                        color=event_color,
                        alpha=0.5,
                        transform=ax.get_xaxis_transform(),
                    )
                    ax.vlines(
                        times_level[event] + event_size / 2,
                        ylow2,
                        yhigh,
                        linestyles="dotted",
                        color=event_color,
                        alpha=0.5,
                        transform=ax.get_xaxis_transform(),
                    )
                    ax.fill_between(
                        np.array(
                            [
                                times_level[event] - event_size / 2,
                                times_level[event] + event_size / 2,
                            ]
                        ),
                        ylow2,
                        yhigh,
                        alpha=0.15,
                        color=event_color,
                        transform=ax.get_xaxis_transform(),
                        edgecolor=None,
                    )

        # add lines per level
        # bottom of row + 5% if n_level > 1
        ylow = (
            level * rowheight if n_level == 1 else level * rowheight + 0.05 * rowheight
        )
        # top of row - 5% if n_level > 1
        yhigh = (
            (level + 1) * rowheight
            if n_level == 1
            else (level + 1) * rowheight - 0.05 * rowheight
        )
        ax.vlines(
            np.array(times_to_display[level]),
            ylow,
            yhigh,
            linestyles="--",
            transform=ax.get_xaxis_transform(),
        )

    # legend
    if colorbar:
        cheight = 1 if n_level == 1 else 2 / n_level
        # axins = ax.inset_axes(width="0.5%", height=cheight, loc="lower left", \
        #         bbox_to_anchor=(1.025, 0, 2, 1), bbox_transform=ax.transAxes, borderpad=0)
        axins = ax.inset_axes([1.025, 0, 0.03, cheight])
        if isinstance(channel_position, Info):
            lab = (
                "Voltage (V)"
                if channel_position["chs"][0]["unit"] == 107
                else channel_position["chs"][0]["unit"]._name
            )
        else:
            lab = "Voltage (V)"
        plot_brain_colorbar(
            axins,
            dict(kind="value", lims=[vmin, 0, vmax]),
            colormap=cmap,
            label=lab,
            bgcolor=".5",
            transparent=None,
        )

    # plot ylabels
    if isinstance(ylabels, dict):
        tick_labels = [str(x) for x in list(ylabels.values())[0]]
        if level_plot:
            tick_labels.reverse()
        ax.set_yticks(np.arange(len(list(ylabels.values())[0])) + 0.5, tick_labels)
        ax.set_ylabel(str(list(ylabels.keys())[0]))
    else:
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)

    if not return_ax:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, n_level)  # -1
        ax.set_xlabel(xlabel)
        if title:
            ax.set_title(title)
    if plt.get_backend()[0:2] == "Qt" or plt.get_backend() == "nbAgg":  # fixes issue with yscaling
        plt.gcf().subplots_adjust(top=0.85, bottom=0.2)  # tight layout didn't work anymore
    if return_ax:
        
        ax.set_ylim(0, n_level)  # -1
        return ax


def plot_components_sensor(hmp_data, positions):
    """
    Visualize the topomap of the HMP principal components.

    Parameters
    ----------
    hmp_data : xr.Dataset
        Data returned from the function hmp.utils.transform_data()
    positions : mne.info | ndarray
        List of x and y positions to plot channels on head model OR MNE info object
    """
    from mne.viz import plot_topomap

    fig, ax = plt.subplots(1, len(hmp_data.attrs["pca_weights"].component))
    for comp in hmp_data.attrs["pca_weights"].component:
        plot_topomap(
            hmp_data.attrs["pca_weights"].values[:, comp],
            positions,
            axes=ax[comp],
            show=False,
            cmap="Spectral_r",
        )


def plot_loocv(
    loocv_estimates,
    pvals=True,
    test="t-test",
    figsize=(16, 5),
    indiv=True,
    ax=None,
    mean=False,
    additional_points=None,
):
    """
    Plot the LOOCV results.

    Parameters
    ----------
    loocv_estimates : ndarray or xarra.DataArray
        results from a call to hmp.utils.loocv()
    pvals : bool
        Whether to display the pvalue with the associated test
    test : str
        which statistical test to compute for the difference in LOOCV-likelihood (one sample t-test
        or sign test)
    figsize : list | tuple | ndarray
        Length and heigth of the matplotlib plot
    indiv : bool
        Whether to plot individual lines
    ax : matplotlib.pyplot.ax
        Matplotlib object on which to draw the plot, can be useful if you want to control specific
        aspects of the plots outside of this function
    mean : bool
        Whether to plot the mean
    additional_points :
        Additional likelihood points to be plotted. Should be provided as a list of tuples
        containing the x coordinate and loocv estimates with a single event, e.g. [(5,estimates)].

    Returns
    -------
    ax : matplotlib.pyplot.ax
        if ax was specified otherwise returns the plot
    """
    if pvals:
        if test == "sign":
            from statsmodels.stats.descriptivestats import sign_test
        elif test == "t-test":
            from scipy.stats import ttest_1samp
        else:
            raise ValueError("Expected sign or t-test argument to test parameter")
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        return_ax = False
    else:
        return_ax = True
    loocv_estimates = loocv_estimates.dropna("n_event", how="all")

    # stats
    diffs, diff_bin, labels = [], [], []
    pvalues = []
    for n_event in np.arange(2, loocv_estimates.n_event.max() + 1):
        diffs.append(
            loocv_estimates.sel(n_event=n_event).data
            - loocv_estimates.sel(n_event=n_event - 1).data
        )  # differences
        diff_bin.append([1 for x in diffs[-1] if x > 0])  # nr of positive differences
        labels.append(str(n_event - 1) + "->" + str(n_event))

        if pvals:
            if test == "sign":
                diff_tmp = np.array(diffs)
                diff_tmp[np.isnan(diff_tmp)] = -np.inf
                pvalues.append((sign_test(diff_tmp[-1])))
            elif test == "t-test":
                pvalues.append((ttest_1samp(diffs[-1], 0, alternative="greater")))

    # first plot
    if mean:
        alpha = 0.4  # for the indiv plot
        marker_indiv = "."
        means = np.nanmean(loocv_estimates.data, axis=1)[::-1]
        errs = (
            np.nanstd(loocv_estimates.data, axis=1) / np.sqrt(len(loocv_estimates.participant))
        )[::-1]
        ax[0].errorbar(x=np.arange(len(means)) + 1, y=means, yerr=errs, marker="o", color="k")
    else:
        alpha = 1
        marker_indiv = "o"
    if indiv:
        for loo in loocv_estimates.T:
            ax[0].plot(loocv_estimates.n_event, loo, alpha=alpha, marker=marker_indiv)
    ax[0].set_ylabel("LOOCV Loglikelihood")
    ax[0].set_xlabel("Number of events")
    ax[0].set_xticks(ticks=loocv_estimates.n_event)

    if additional_points:  # only plot average for now
        if not isinstance(additional_points, list):
            additional_points = [additional_points]
        for ap in additional_points:
            xap = ap[0]
            meanap = np.mean(ap[1].values)
            err = np.nanstd(ap[1].values) / np.sqrt(len(ap[1].values))
        ax[0].errorbar(x=xap, y=meanap, yerr=err, marker="o")

    # second plot
    diffs = np.array(diffs)
    diffs[np.isneginf(diffs)] = np.nan
    diffs[np.isinf(diffs)] = np.nan

    ax[1].plot(diffs, ".-", alpha=0.6)
    ax[1].set_xticks(ticks=np.arange(0, loocv_estimates.n_event.max() - 1), labels=labels)
    ax[1].hlines(0, 0, len(np.arange(2, loocv_estimates.n_event.max())), color="lightgrey", ls="--")
    ax[1].set_ylabel("Change in likelihood")
    ax[1].set_xlabel("")

    if pvals:
        ymin = np.nanmin(diffs[:])
        ymintext = ymin - (np.nanmax(diffs[:]) - ymin) * 0.05
        ymin = ymin - (np.nanmax(diffs[:]) - ymin) * 0.1
        ax[1].set_ylim(bottom=ymin)
        for n_event in np.arange(2, loocv_estimates.n_event.max() + 1):
            ax[1].text(
                x=n_event - 2,
                y=ymintext,
                s=str(int(np.nansum(diff_bin[n_event - 2])))
                + "/"
                + str(len(diffs[-1]))
                + ": "
                + str(np.around(pvalues[n_event - 2][-1], 3)),
                ha="center",
            )

    if return_ax:
        if pvals:
            return [ax, pvalues]
        else:
            return ax
    else:
        plt.tight_layout()
        if pvals:
            return pvalues


def plot_latencies(
    estimates,
    init=None,
    labels=[],
    colors=default_colors,
    figsize=False,
    errs=None,
    kind="bar",
    legend=False,
    max_time=None,
    as_time=False,
):
    """
    Plot the average of stage latencies with choosen errors bars.

    Parameters
    ----------
    estimates : hmp results object
        hmp results object
    event_width : float
        Display size of the event in time unit given sampling frequency.

        If drawing a fitted object using hmp you can provide the event_width_sample of
        fitted hmp (e.g. init.event_width_sample)
    labels : tuples | list
        labels to draw on the y axis
    colors : ndarray
        array of colors for the different stages
    figsize : list | tuple | ndarray
        Length and heigth of the matplotlib plot
    errs : str
        Whether to display no error bars (None), standard deviation ('std'),
        or standard error ('se')
    times_to_display : ndarray
        Times to display (e.g. Reaction time or any other relevant time) in the time unit of the
        fitted data
    max_time : float
        limit of the x (time) axe
    kind : str
        bar or point
    as_time : bool
        if true, plot time (ms) instead of samples.
    """
    if as_time and init is not None:
        time_step = 1000 / init.sfreq  # time_step still needed below
    else:
        time_step = 1

    # if hmp estimates are provided, calculate time
    if isinstance(estimates, (xr.DataArray, xr.Dataset)):
        assert init is not None, "If hmp results object provided, init is a required parameter."
        ydim = None
        if (
            "n_events" in estimates.dims and estimates.n_events.count() > 1
        ):  # and there are multiple different fits (eg backward estimation)
            ydim = "n_events"
        elif "levels" in estimates:
            ydim = "levels"
        avg_durations = event_times(
            estimates, mean=True, duration=True, add_rt=True, as_time=as_time
        ).data
        avg_times = event_times(
            estimates, mean=True, duration=False, add_rt=True, as_time=as_time
        ).data
        if errs is not None:
            errorbars = event_times(
                estimates,
                duration=True,
                add_rt=True,
                extra_dim=ydim,
                as_time=as_time,
                errorbars=errs,
            )
        if len(avg_times.shape) == 1:
            avg_durations = np.expand_dims(avg_durations, axis=0)
            avg_times = np.expand_dims(avg_times, axis=0)
            if errs is not None:
                errorbars = np.expand_dims(errorbars, axis=0)
        avg_durations = avg_durations * time_step
        avg_times = avg_times * time_step
        if errs is not None:
            errorbars = errorbars * time_step

        if ydim == "levels":  # reverse order, to make correspond to level maps
            avg_durations = np.flipud(avg_durations)
            avg_times = np.flipud(avg_times)
            if errs is not None:
                errorbars = np.flipud(errorbars)

        n_model = avg_times.shape[0]
    else:
        raise ValueError("Expected an hmp fitted object")

    if labels == []:
        if ydim == "levels":
            labels = [str(x) for x in reversed(list(estimates.clabels.values())[0])]
        else:
            labels = np.arange(n_model)

    if not figsize:
        figsize = (8, 1 * n_model + 2)
    f, axs = plt.subplots(1, 1, figsize=figsize, dpi=100)

    cycol = cycle(colors)
    cur_colors = [next(cycol) for x in np.arange(n_model)]  # color per level/model (line plot)

    for j in range(n_model):  # per level/model
        avg_times_model = np.hstack((0, avg_times[j, :]))
        avg_durations_model = avg_durations[j, :]
        n_stages = len(avg_durations_model)
        errorbars_model = errorbars[j, :, :] if errs is not None else None

        if kind == "bar":
            cycol = cycle(colors)  # get a color per stage
            cur_colors = [next(cycol) for x in np.arange(n_stages)]

            # #remove 0 stages and associated color
            if np.any(np.isnan(avg_durations_model)):
                missing_evts = np.where(np.isnan(avg_durations_model))[0]
                avg_durations_model = np.delete(avg_durations_model, missing_evts)
                avg_times_model = np.delete(avg_times_model, missing_evts + 1)
                if errs is not None:
                    errorbars_model = np.delete(errorbars_model, missing_evts, axis=1)
                cur_colors = np.delete(cur_colors, missing_evts)
                n_stages = n_stages - len(missing_evts)

            for st in reversed(range(n_stages)):  # can't deal with colors in one call
                plt.barh(
                    j,
                    avg_durations_model[st],
                    left=avg_times_model[st],
                    color="w",
                    edgecolor=cur_colors[st],
                )
                if errs is not None:
                    plt.errorbar(
                        np.repeat(avg_times_model[st + 1], 2),
                        np.repeat(j, 2),
                        xerr=errorbars_model[:, st],
                        color=cur_colors[st],
                        fmt="none",
                        capsize=10,
                    )

        elif kind == "point":
            plt.errorbar(
                np.arange(n_stages) + 1,
                avg_durations_model,
                yerr=errorbars_model,
                color=cur_colors[j],
                fmt="o-",
                capsize=10,
                label=labels[j],
            )
        else:
            raise ValueError("Unknown 'kind'")

    # general settings
    if kind == "bar":
        plt.yticks(np.arange(len(labels)), labels)
        plt.ylim(0 - 1, j + 1)

        if not max_time:
            max_time = (
                np.nanmax(avg_times) + np.nanmax(errorbars) if errs else np.nanmax(avg_times)
            ) * 1.05
        axs.set_xlim(0, max_time)
        if as_time:
            plt.xlabel("Cumulative stage durations from stimulus onset (ms)")
        else:
            plt.xlabel("Cumulative stage durations from stimulus onset (samples)")
    elif kind == "point":
        plt.xlim(1 - 0.5, n_stages + 0.5)

        max_y = (
            np.nanmax(avg_durations) + np.nanmax(errorbars) if errs else np.nanmax(avg_durations)
        ) * 1.05
        axs.set_ylim(0, max_y)

        if as_time:
            plt.ylabel("Stage durations (ms)")
        else:
            plt.ylabel("Stage durations (samples)")
        plt.xlabel("Stage")

    plt.tight_layout()
    # Hide the right and top spines
    axs.spines.right.set_visible(False)
    axs.spines.top.set_visible(False)
    # Only show ticks on the left and bottom spines
    axs.yaxis.set_ticks_position("left")
    axs.xaxis.set_ticks_position("bottom")
    if legend:
        axs.legend()
    return axs


def erp_data(epoched_data, times, channel, n_samples=None, pad=1):
    """
    Create a data array compatible with the plot ERP function.

    Optionnally this function can resample the epochs to fit some provided times (e.g. onset of
    the events).

    Parameters
    ----------
        epoched_data: xr.Dataset
            Epoched physiological data with dims 'participant'X 'epochs' X 'channels'X 'samples'
        times: xr.Dataset
            Times between wich to extract or resample the data with dims 'trial_x_participant' X
            'event'
        channel: str
            For which channel to extract the data
        n_samples: int
            How many samples to resample on if any
        pad: int
            padding added to the beginning and the end of the signal

    Returns
    -------
    data : nd.array
        array containing the extracted times for each epoch and stage with format epochs X events X
        samples.
    """
    epoched_data = epoched_data.sel(channels=channel)
    if n_samples is None:
        data = (
            np.zeros((len(times.trial_x_participant), len(times.event), len(epoched_data.samples)))
            * np.nan
        )
    else:
        data = np.zeros((len(times.trial_x_participant), len(times.event), n_samples)) * np.nan

    for i, trial in enumerate(times.trial_x_participant):
        for j, event in enumerate(times.event):
            if event == 0:
                sub_prevtime = 0
            else:
                sub_prevtime = times.sel(trial_x_participant=trial, event=event - 1).data
            sub_times = times.sel(trial_x_participant=trial, event=event).data - 1
            time_diff = sub_times - sub_prevtime
            if time_diff > 1:  # rounds up to 0 otherwise
                subset = epoched_data.sel(
                    trial_x_participant=trial, samples=slice(sub_prevtime, sub_times)
                )
                if n_samples is None:
                    limit = np.sum(~subset.data.isnull()).values
                    data[i, j, :limit] = subset.data
                else:
                    padded_data = np.concatenate(
                        [
                            np.repeat([subset.data[0]], pad),
                            subset.data,
                            np.repeat([subset.data[-1]], pad),
                        ]
                    )
                    # resample_poly seems to have better results than resample
                    data[i, j] = ssignal.resample_poly(
                        padded_data,
                        n_samples * 10,
                        len(padded_data.data) * (n_samples / 10),
                        padtype="median",
                    )
                    # data[i,j] = ssignal.resample(padded_data.data, n_samples)
    return data


def plot_erp(
    times, data, color="k", ax=None, minmax_lines=None, upsample=1, bootstrap=None, label=None
):
    """
    Plot the ERP based on the times extracted by HMP.

    Either around an event or just stimulus and response and the data extracted from ```erp_data```.

    Parameters
    ----------
        times: xr.Dataset
            Times between wich to extract or resample the data with dims
        data: nd.array
            numpy array from the erp_data functino
        color: str
            color for the lines
        ax: matplotlib.pyplot
            ax on which to draw
        minmax_lines: tuple
            Min and max arguments for the vertical lines on the plot
        upsample: float
            Upsampling factor for the times
        bootstrap: int
            how many bootstrap draw to perform
    """
    if ax is None:
        ax = plt
    for event in times.event[1:]:
        time = times.sel(event=event - 1).mean()
        time_current = int(times.sel(event=event).mean())
        if len(times.event) > 2:
            x = np.linspace(time, time_current, num=np.shape(data)[-1]) * upsample
            mean_signal = np.nanmean(data[:, event, :], axis=0)
        else:
            x = np.arange(time, time_current) * upsample
            mean_signal = np.nanmean(data[:, event, :], axis=0)[:time_current]
        ax.plot(x, mean_signal, color=color, label=label)
        if minmax_lines is not None:
            ax.vlines(
                times.mean("trial_x_participant") * upsample,
                minmax_lines[0],
                minmax_lines[1],
                color=color,
                ls=":",
                alpha=0.25,
            )
        if bootstrap is not None:
            test_boot = stats.bootstrap(
                (data[:, event, :],), statistic=np.nanmean, n_resamples=bootstrap, axis=0, batch=10
            )
            if len(times.event) > 2:
                ax.fill_between(
                    x,
                    test_boot.confidence_interval[0],
                    test_boot.confidence_interval[1],
                    alpha=0.5,
                    color=color,
                )
            else:
                ax.fill_between(
                    x,
                    test_boot.confidence_interval[0][:time_current],
                    test_boot.confidence_interval[1][:time_current],
                    alpha=0.5,
                    color=color,
                )
