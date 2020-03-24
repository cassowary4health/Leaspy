import os
from leaspy import Leaspy, Data, AlgorithmSettings, Plotter

# Create the logs dir
if not os.path.isdir('_outputs/logs/fit'):
    os.makedirs('_outputs/logs/fit')

# Inputs
data = Data.from_csv_file(os.path.join(os.path.dirname(__file__), '_inputs', 'data.csv'))
algo_settings = AlgorithmSettings('mcmc_saem', n_iter=200, n_burn_in_iter=40)
algo_settings.set_logs('_outputs/logs/fit')

# Initialize
leaspy = Leaspy("logistic_parallel")
leaspy.model.load_hyperparameters({'source_dimension': 2})

#path_to_saved_model = os.path.join(os.path.dirname(__file__), '_outputs', 'fitted_multivariate_model.json')
#leaspy = Leaspy.load(path_to_saved_model)


#%%

# Fit the model on the data
leaspy.fit(data, algorithm_settings=algo_settings)




# Save the model
#path_to_saved_model = os.path.join(os.path.dirname(__file__), '_outputs', 'fitted_multivariate_model.json')
#leaspy.save(path_to_saved_model)

# Compute individual parameters
algo_personalize_settings = AlgorithmSettings('mode_real', n_iter = 5, n_burn_in_iter=2)
result = leaspy.personalize(data, settings=algo_personalize_settings)


# Save the individual parameters
#path_to_individual_parameters = os.path.join(os.path.dirname(__file__), '_outputs', 'individual_parameters.json')
#leaspy.save_individual_parameters(path_to_individual_parameters, result.individual_parameters)

#%%

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1, sharex=True)

leaspy.plotting.average_trajectory(ax=ax)
leaspy.plotting.patient_trajectories(result, ['116'], ax=ax)
leaspy.plotting.patient_observations(result, ['116'], ax=ax)

plt.show()