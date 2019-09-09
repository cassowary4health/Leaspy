import os
from leaspy.main import Leaspy
from leaspy.inputs.data.data import Data
from leaspy.inputs.settings.algorithm_settings import AlgorithmSettings

# Create the output dir
if not os.path.isdir('_outputs/logs/fit'):
    os.makedirs('_outputs/logs/fit')

# Inputs
data = Data.from_csv_file(os.path.join(os.path.dirname(__file__), '_inputs', 'data.csv'))
algo_settings = AlgorithmSettings('mcmc_saem', n_iter=200)
algo_settings.set_logs('_outputs/logs/fit')

# Initialize
leaspy = Leaspy("logistic_parallel")
leaspy.model.load_hyperparameters({'source_dimension': 2})

# Fit the model on the data
leaspy.fit(data, algorithm_settings=algo_settings)

# Save the model
path_to_saved_model = os.path.join(os.path.dirname(__file__), '_outputs', 'fitted_multivariate_model.json')
leaspy.save(path_to_saved_model)

# Compute individual parameters
algo_personalize_settings = AlgorithmSettings('scipy_minimize')
result = leaspy.personalize(data, settings=algo_personalize_settings)

# Save the individual parameters
path_to_individual_parameters = os.path.join(os.path.dirname(__file__), '_outputs', 'individual_parameters.json')
#leaspy.save_individual_parameters(path_to_individual_parameters, result.individual_parameters)



