"""Different methods for transforming E/MEG data to HMP ready data."""

from hmp.io.bids_raw import read_bids_raw
from hmp.io.mne_raw import read_mne_raw
from hmp.io.mne_epochs import read_mne_epochs

__all__ = ["read_bids_raw", "read_mne_raw", "read_mne_epochs"]
