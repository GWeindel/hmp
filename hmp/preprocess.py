def transform_data(
    epoch_data,
    participants_variable="participant",
    apply_standard=False,
    averaged=False,
    apply_zscore="trial",
    zscore_across_pcs=False,
    method="pca",
    cov=True,
    centering=True,
    n_comp=None,
    n_ppcas=None,
    pca_weights=None,
    bandfilter=None,
    mcca_reg=0,
    copy=False,
):
    """Adapt EEG epoched data (in xarray format) to the expected data format for hmp.

    First this code can apply standardization of individual variances (if apply_standard=True).
    Second, a spatial PCA on the average variance-covariance matrix is performed (if method='pca',
    more methods in development).
    Third,stacks the data going from format [participant * epochs * samples * channels] to
    [samples * channels].
    Last, performs z-scoring on each epoch and for each principal component (PC), or for each
    participant and PC, or across all data for each PC.

    Parameters
    ----------
    data : xarray
        unstacked xarray data from transform_data() or anyother source yielding an xarray with
        dimensions [participant * epochs * samples * channels]
    participants_variable : str
        name of the dimension for participants ID
    apply_standard : bool
        Whether to apply standardization of variance between participants, recommended when they
        are few of them (e.g. < 10)
    averaged : bool
        Applying the pca on the averaged ERP (True) or single trial ERP (False, default).
        No effect if cov = True
    apply_zscore : str
        Whether to apply z-scoring and on what data, either None, 'all', 'participant', 'trial',
        for zscoring across all data, by participant, or by trial, respectively. If set to true,
        evaluates to 'trial' for backward compatibility.
    method : str
        Method to apply, 'pca' or 'mcca'
    cov : bool
        Wether to apply the pca/mcca to the variance covariance (True, default) or the epoched data
    n_comp : int
        How many components to select from the PC space, if None plots the scree plot and a prompt
        requires user to specify how many PCs should be retained
    n_ppcas : int
        If method = 'mcca', controls the number of components retained for the by-participant PCAs
    pca_weigths : xarray
        Weights of a PCA to apply to the data (e.g. in the resample function)
    bandfilter: None | (lfreq, hfreq)
        If none, no filtering is appliedn. If tuple, data is filtered between lfreq-hfreq.
        NOTE: filtering at this step is suboptimal, filter before epoching if at all possible, see
            also https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html
    mcca_reg: float
        regularization used for the mcca computation (see mcca.py)
    copy: bool
        Whether to copy the data before transforming
    Returns
    -------
    data : xarray.Dataset
        xarray dataset [n_samples * n_comp] data expressed in the PC space, ready for HMP fit
    """
    if copy is True:
        data = epoch_data.copy(deep=True)
    else:
        data = epoch_data
        warn(
                "Data will be modified inplace, re-read the data or use copy=True if multiple"
                "calls to this function"
            )
    if isinstance(data, xr.DataArray):
        raise ValueError(
            "Expected a xarray Dataset with data and event as DataArrays, check the data format"
        )
    if apply_zscore not in ["all", "participant", "trial"] and not isinstance(apply_zscore, bool):
        raise ValueError(
            "apply_zscore should be either a boolean or one of ['all', 'participant', 'trial']"
        )
    assert (
        np.sum(
            np.isnan(
                data.groupby("participant", squeeze=False).mean(["epochs", "samples"]).data.values
            )
        )
        == 0
    ), "at least one participant has an empty channel"
    if method == "mcca" and data.sizes["participant"] == 1:
        raise ValueError("MCCA cannot be applied to only one participant")
    sfreq = data.sfreq
    if bandfilter:
        data = _filtering(data, bandfilter, sfreq)
    if apply_standard:
        if "participant" not in data.dims or len(data.participant) == 1:
            warn(
                "Requested standardization of between participant variance yet no participant "
                "dimension is found in the data or only one participant is present. "
                "No standardization is done, set apply_standard to False to avoid this warning."
            )
        else:
            mean_std = data.groupby(participants_variable, squeeze=False).std(dim=...).data.mean()
            data = data.assign(mean_std=mean_std.data)
            data = data.groupby(participants_variable, squeeze=False).map(_standardize)
    if isinstance(data, xr.Dataset):  # needs to be a dataset if apply_standard is used
        data = data.data
    if centering or method == "mcca":
        data = _center(data)
    if apply_zscore is True:
        apply_zscore = "trial"  # defaults to trial
    data = data.transpose("participant", "epochs", "channels", "samples")
    if method == "pca":
        if pca_weights is None:
            if cov:
                indiv_data = np.zeros(
                    (data.sizes["participant"], data.sizes["channels"], data.sizes["channels"])
                )
                for i in range(data.sizes["participant"]):
                    x_i = np.squeeze(data.data[i])
                    indiv_data[i] = np.mean(
                        [
                            np.cov(x_i[trial, :, ~np.isnan(x_i[trial, 0, :])].T)
                            for trial in range(x_i.shape[0])
                            if ~np.isnan(x_i[trial, 0, :]).all()
                        ],
                        axis=0,
                    )
                pca_ready_data = np.mean(np.array(indiv_data), axis=0)
            elif averaged:
                erps = []
                for part in data.participant:
                    erps.append(data.sel(participant=part).groupby("samples").mean("epochs").T)
                pca_ready_data = np.nanmean(erps, axis=0)
            else:
                pca_ready_data = data.stack(
                    {"all": ["participant", "epochs", "samples"]}
                ).dropna("all")
                pca_ready_data = pca_ready_data.transpose("all", "channels")
            # Performing spatial PCA on the average var-cov matrix
            pca_weights = _pca(pca_ready_data, n_comp, data.coords["channels"].values)
            data = data @ pca_weights
            data.attrs["pca_weights"] = pca_weights
    elif method == "mcca":
        ori_coords = data.drop_vars("channels").coords
        if n_ppcas is None:
            n_ppcas = n_comp * 3
        mcca_m = mcca.MCCA(n_components_pca=n_ppcas, n_components_mcca=n_comp, r=mcca_reg)
        if cov:
            fitted_data = data.transpose("participant", "epochs", "samples", "channels").data
            ccs = mcca_m.obtain_mcca_cov(fitted_data)
        else:
            if averaged:
                fitted_data = (
                    data.mean("epochs").transpose("participant", "samples", "channels").data
                )
            else:
                fitted_data = (
                    data.stack({"all": ["epochs", "samples"]})
                    .transpose("participant", "all", "channels")
                    .data
                )
            ccs = mcca_m.obtain_mcca(fitted_data)
        trans_ccs = np.tile(
            np.nan,
            (data.sizes["participant"], data.sizes["epochs"], data.sizes["samples"], ccs.shape[-1]),
        )
        for i, part in enumerate(data.participant):
            trans_ccs[i] = mcca_m.transform_trials(
                data.sel(participant=part).transpose("epochs", "samples", "channels").data.copy()
            )
        data = xr.DataArray(
            trans_ccs,
            dims=["participant", "epochs", "samples", "component"],
            coords=dict(
                participant=data.participant,
                epochs=data.epochs,
                samples=data.samples,
                component=np.arange(n_comp),
            ),  # n_comp
        )
        data = data.assign_coords(ori_coords)
        data.attrs["mcca_weights"] = mcca_m.mcca_weights
        data.attrs["pca_weights"] = mcca_m.pca_weights
    elif method is None:
        data = data.rename({"channels": "component"})
        data["component"] = np.arange(len(data.component))
        data.attrs["pca_weights"] = np.identity(len(data.component))
    else:
        raise ValueError(f"method {method} is unknown, choose either 'pca', 'mcca' or None")

    if apply_zscore:
        ori_coords = data.coords
        match apply_zscore:
            case "all":
                if zscore_across_pcs:
                    data = zscore_xarray(data)
                else:
                    data = (
                        data.stack(comp=["component"])
                        .groupby("comp", squeeze=False)
                        .map(zscore_xarray)
                        .unstack()
                    )
            case "participant":
                if zscore_across_pcs:
                    data = data.groupby("participant").map(zscore_xarray)
                else:
                    data = (
                        data.stack(participant_comp=[participants_variable, "component"])
                        .groupby("participant_comp", squeeze=False)
                        .map(zscore_xarray)
                        .unstack()
                    )
            case "trial":
                if zscore_across_pcs:
                    data = (
                        data.stack(trial=[participants_variable, "epochs"])
                        .groupby("trial")
                        .map(zscore_xarray)
                        .unstack()
                    )
                else:
                    data = (
                        data.stack(trial=[participants_variable, "epochs", "component"])
                        .groupby("trial", squeeze=False)
                        .map(zscore_xarray)
                        .unstack()
                    )
        data = data.transpose("participant", "epochs", "samples", "component")
        data = data.assign_coords(ori_coords)

    data.attrs["pca_weights"] = pca_weights
    data.attrs["sfreq"] = sfreq
    data = stack_data(data)
    return data
