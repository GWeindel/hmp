"""Different methods for projecting data into subspace."""

from .base import Projector
from .custom import Custom
from .identity import Identity
from .pca import PCA

__all__ = ["Projector", "Custom", "Identity", "PCA"]
