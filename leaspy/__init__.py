__version__ = '1.0.3-dev'

dtype = 'float32'

# Plotter
from leaspy.io.logs.visualization.plotter import Plotter
from leaspy.io.logs.visualization.plotting import Plotting
# API
from .api import Leaspy
# Inputs
from .io.data.data import Data
from .io.data.dataset import Dataset
# Outputs
from .io.outputs.individual_parameters import IndividualParameters
from .io.outputs.result import Result
# Algorithm Settings
from .io.settings.algorithm_settings import AlgorithmSettings

# add a watermark with all pkg versions (for trace)
from torch import __version__ as v_torch
from numpy import __version__ as v_numpy
from pandas import __version__ as v_pandas
from scipy import __version__ as v_scipy
from sklearn import __version__ as v_sklearn
from joblib import __version__ as v_joblib
from statsmodels import __version__ as v_statsmodels # for LME only
from matplotlib import __version__ as v_matplotlib

__watermark__ = {
    'leaspy': __version__,
    'torch': v_torch,
    'numpy': v_numpy,
    'pandas': v_pandas,
    'scipy': v_scipy,
    'sklearn': v_sklearn,
    'joblib': v_joblib,
    'statsmodels': v_statsmodels,
    'matplotlib': v_matplotlib,
}
