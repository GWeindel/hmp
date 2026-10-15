"""Different methods for cross-validation."""

from hmp.crossvalidation.loo import LOOCV
from hmp.crossvalidation import kfold

__all__ = ["LOOCV", "kfold"]
