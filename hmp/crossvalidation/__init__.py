"""Different methods for cross-validation."""

from hmp.crossvalidation.loo import LOOCV
from hmp.crossvalidation.pseudo_kfold import KFoldCV

__all__ = ["LOOCV", "KFoldCV"]
