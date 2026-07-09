"""Different methods for transforming E/MEG data to HMP ready data."""

from hmp.io.bids_raw import read_bids_raw
from hmp.io.metadata import add_metadata
from hmp.io.mne_epochs import read_mne_epochs
from hmp.io.mne_raw import read_mne_raw

__all__ = ["read_bids_raw", "read_mne_raw", "read_mne_epochs", "add_metadata"]
