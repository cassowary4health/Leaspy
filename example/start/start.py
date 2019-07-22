import os
from src.main import Leaspy
from src.inputs.data.data import Data
from src.inputs.algorithm_settings import AlgorithmSettings

# Inputs
data = Data(os.path.join(os.path.dirname(__file__), 'data_tiny.csv'))
algo_settings = AlgorithmSettings(os.path.join(os.path.dirname(__file__), 'algorithm_settings.json'))

# Launch
leaspy = Leaspy("multivariate_parallel")
#leaspy = Leaspy("multivariate")
#leaspy = Leaspy.from_model_settings(path_to_parameters)
leaspy.fit(data, algorithm_settings=algo_settings)
