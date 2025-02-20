def save(data, filename):
    """Save fit."""
    data = data.copy()
    attributes = data.attrs.copy()
    for attr in attributes:
        if isinstance(data.attrs[attr], np.ndarray):
            del data.attrs[attr]
    data.unstack().to_netcdf(filename)

    print(f"{filename} saved")


def load(filename):
    """Load fit or data."""
    with xr.open_dataset(filename) as data:
        data.load()
    if "trials" in data:
        data = data.stack(trial_x_participant=["participant", "trials"]).dropna(
            dim="trial_x_participant", how="all"
        )
    if "eventprobs" in data and all(
        key in data for key in ["trial_x_participant", "samples", "event"]
    ):
        # Ensures correct order of dimensions for later index use
        if "iteration" in data:
            data["eventprobs"] = data.eventprobs.transpose(
                "iteration", "trial_x_participant", "samples", "event"
            )
        else:
            data["eventprobs"] = data.eventprobs.transpose(
                "trial_x_participant", "samples", "event"
            )
    return data


def save_eventprobs(eventprobs, filename):
    """Save eventprobs to filename csv file."""
    eventprobs = eventprobs.unstack()
    eventprobs.to_dataframe().to_csv(filename)
    print(f"Saved at {filename}")
