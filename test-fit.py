import numpy as np
import scipy.io
import math
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import pyhsmm_mvpa as hsmm

import os

import importlib
importlib.reload(hsmm)


mat = scipy.io.loadmat('matlab/newbumps/starter.mat')

data = np.array(mat['normedscore10'])
starts = np.array(mat['x'][:,0]) -1#correcting to be 0 indexed
ends = np.array(mat['y'][:,0])-1#correcting to be 0 indexed
subjects = np.array(mat['subjects'])-1 #correcting to be 0 indexed

init = hsmm.hsmm(data, starts, ends)

init.fit_single(1,True,threshold=1)

init.fit_iterative(2,True,threshold=1)