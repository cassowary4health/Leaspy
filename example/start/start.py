import os
from src.main import Leaspy
from src.inputs.data.data import Data
from src.inputs.settings.algorithm_settings import AlgorithmSettings
import json

# Inputs
data = Data.from_csv_file(os.path.join(os.path.dirname(__file__), '_inputs', 'data.csv'))
algo_settings = AlgorithmSettings(os.path.join(os.path.dirname(__file__), '_inputs', 'algorithm_settings.json'))
#algo_personalize_settings = AlgorithmSettings(os.path.join(os.path.dirname(__file__), '_inputs', 'algorithm_personalize_settings.json'))

# Initialize
#leaspy = Leaspy("multivariate_parallel")
leaspy = Leaspy("multivariate_parallel")
leaspy.model.load_hyperparameters({'source_dimension': 0})

# Fit the model on the data
leaspy.fit(data, algorithm_settings=algo_settings)


# Save the model
path_to_saved_model = os.path.join(os.path.dirname(__file__), '_outputs', 'fitted_multivariate_model.json')
leaspy.save(path_to_saved_model)

# Compute individual parameters
#individual_parameters = leaspy.personalize(data, settings=algo_personalize_settings)

# Save the individual parameters
#path_to_individual_parameters = os.path.join(os.path.dirname(__file__), '_outputs', 'individual_parameters.json')

#leaspy.save_individual_parameters(path_to_individual_parameters,individual_parameters)


# Load the model as if it is another day of your life
#leaspy2 = Leaspy.load(path_to_saved_model)

# Fit a second time
#leaspy2.fit(data, algorithm_settings=algo_settings)