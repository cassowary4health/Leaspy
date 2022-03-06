__version__ = '1.1.2'

dtype = 'float32'

# Plotter
from leaspy.leaspy_io.logs.visualization.plotter import Plotter
from leaspy.leaspy_io.logs.visualization.plotting import Plotting
# API
from .api import Leaspy
# Inputs
from .leaspy_io.data.data import Data
from .leaspy_io.data.dataset import Dataset
# Outputs
from .leaspy_io.outputs.individual_parameters import IndividualParameters
from .leaspy_io.outputs.result import Result
# Algorithm Settings
from .leaspy_io.settings.algorithm_settings import AlgorithmSettings

# add a watermark with all pkg versions (for trace)
from importlib import import_module

pkg_deps = ['torch', 'numpy', 'pandas', 'scipy', # core
            'sklearn', 'joblib', # parallelization / ML utils
            'statsmodels', # LME benchmark only
            'matplotlib' # plots
            ]

__watermark__ = {
    'leaspy': __version__,
    **{pkg_name: import_module(pkg_name).__version__ for pkg_name in pkg_deps}
}

del pkg_deps
