__version__ = '0.1.0rc1'

dtype = 'float32'

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

# Plotter
from leaspy.io.logs.visualization.plotter import Plotter
