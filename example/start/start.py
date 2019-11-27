import os
from leaspy import Leaspy, Data, AlgorithmSettings, Plotter

# Create the output dir
if not os.path.isdir('_outputs/logs/fit'):
    os.makedirs('_outputs/logs/fit')

# Inputs
data = Data.from_csv_file(os.path.join(os.path.dirname(__file__), '_inputs', 'data.csv'))
algo_settings = AlgorithmSettings('mcmc_saem', n_iter=20, n_burn_in_iter=10)
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
algo_personalize_settings = AlgorithmSettings('mode_real', n_iter = 5, n_burn_in_iter=2)
result = leaspy.personalize(data, settings=algo_personalize_settings)

#%% Add cofactors
import pandas as pd
df = pd.DataFrame({'ID' : list(data.individuals.keys()),
                   'cof1': [1]*data.n_individuals,
                   'cof2': [2] * data.n_individuals}).set_index('ID')
data.load_cofactors(df, ['cof1','cof2'])
res2 = result.get_dataframe_individual_parameters(cofactors=['cof1','cof2'])

# Save the individual parameters
path_to_individual_parameters = os.path.join(os.path.dirname(__file__), '_outputs', 'individual_parameters.json')
#leaspy.save_individual_parameters(path_to_individual_parameters, result.individual_parameters)



