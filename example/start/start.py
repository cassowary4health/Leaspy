import os
from src.main import Leaspy
from src.inputs.data.data import Data
from src.inputs.algorithm_settings import AlgorithmSettings

# Inputs
data = Data(os.path.join(os.path.dirname(__file__), '_inputs', 'data_tiny.csv'))
algo_settings = AlgorithmSettings(os.path.join(os.path.dirname(__file__), '_inputs', 'algorithm_settings.json'))

# Initialize
leaspy = Leaspy("multivariate_parallel")
#leaspy = Leaspy("multivariate")

# Fit the model on the data
leaspy.fit(data, algorithm_settings=algo_settings)

# Save the model
path_to_saved_model = os.path.join(os.path.dirname(__file__), '_outputs', 'fitted_multivariate_model.json')
leaspy.save(path_to_saved_model)

# Load the model as if it is another day of your life
leaspy2 = Leaspy.load(path_to_saved_model)

# Fit a second time
leaspy2.fit(data, algorithm_settings=algo_settings)
