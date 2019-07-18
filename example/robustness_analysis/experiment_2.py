
#%%
import os
from src.main import Leaspy
from src.legacy import generate_data_from_model
from src.inputs.algorithm_settings import AlgorithmSettings
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import torch


"""
Experiment 1 : Convergence robustness analysis

From different initial parameters, with the same synthetic data, do we get the same final average trajectories ???
"""

# Parameters
n_patients = 100
n_iter = 500
n_jobs = 3

file_path = os.path.dirname(__file__)

path_output_file = os.path.join(file_path, "../../../output_leaspy/synthetic_data_validation")
path_output = os.path.join(file_path, "../../../output_leaspy/")
path_to_algo_parameters = os.path.join(file_path,"../data", '_generate_data', "algorithm_settings.json")
path_to_true_model_parameters = os.path.join(file_path,"../data", '_generate_data', 'true_model_parameters.json')

if not os.path.exists(path_output):
    #if not os.path.exists(path_output):
    #    os.mkdir(path_output)
    os.mkdir(path_output)

# Algorithm settings

algo_settings = AlgorithmSettings(path_to_algo_parameters)
algo_settings.parameters['n_iter'] = n_iter

# Create the data
leaspy_dummy = Leaspy.from_model_settings(path_to_true_model_parameters)
data = generate_data_from_model(leaspy_dummy.model, n_patients=n_patients)





#leaspy = Leaspy.from_model_settings(path_to_initial_model_parameters)
#leaspy.fit(data, algo_settings, seed=seed)

def run_experiment_2(data, model_settings_path, algo_settings, seed):
    # output path
    run_path = os.path.join(path_output, "seed_{0}".format(seed))
    if not os.path.exists(run_path):
        os.mkdir(run_path)
    algo_settings.output_path = run_path

    # Load a leaspy
    leaspy = Leaspy.from_model_settings(model_settings_path)

    # Modify initialization
    leaspy.model.model_parameters['p0'] += np.random.normal(loc=0, scale=0.2)
    leaspy.model.model_parameters['tau_mean'] += np.random.normal(loc=0, scale=0.6)
    leaspy.model.model_parameters['xi_mean'] += np.random.normal(loc=0, scale=1.5)

    # Fit a Leaspy
    leaspy.fit(data, algo_settings, seed=seed)
    
    return leaspy

#res = run_experiment_1(data,path_to_initial_model_parameters,algo_settings, 0)


from joblib import Parallel, delayed
res = Parallel(n_jobs=n_jobs)(delayed(lambda seed:
                           run_experiment_2(data,
                                            path_to_true_model_parameters,
                                            algo_settings,
                                            seed))(seed) for seed in range(n_jobs))


#%%

# Plot the estimated average trajectory vs true average trajectory

numpy_timepoints = np.linspace(-2,2,50)
tensor_timepoints = torch.Tensor(numpy_timepoints)
true_average_trajectory = leaspy_dummy.model.compute_average(tensor_timepoints).detach().numpy()
estimated_average_trajectories = [leaspy.model.compute_average(tensor_timepoints).detach().numpy() for leaspy in res]
estimated_model_parameters_array = [leaspy.model.model_parameters for leaspy in res]

fig, ax = plt.subplots(1,1, figsize=(20,10))

 # Plot true parameters
true_model_parameters = leaspy_dummy.model.model_parameters
ax.plot(numpy_timepoints, true_average_trajectory, linewidth=5, c='black', alpha=0.4)
ax.plot([true_model_parameters['tau_mean'], true_model_parameters['tau_mean']],
        [0, true_model_parameters['p0'][0]], c='black')
ax.plot([-2, true_model_parameters['tau_mean']],
        [true_model_parameters['p0'][0], true_model_parameters['p0'][0]], c='black')


colors = cm.rainbow(np.linspace(0,1,n_jobs))

i=0
for estimated_model_parameters, estimated_average_trajectory in zip(estimated_model_parameters_array,
                                                                    estimated_average_trajectories):

    ax.plot(numpy_timepoints, estimated_average_trajectory, linewidth=3, alpha= 0.7, c=colors[i])

    ax.plot([estimated_model_parameters['tau_mean'], estimated_model_parameters['tau_mean']],
            [0, estimated_model_parameters['p0'][0]], c=colors[i])

    ax.plot([-2, estimated_model_parameters['tau_mean']],
            [estimated_model_parameters['p0'][0], estimated_model_parameters['p0'][0]], c=colors[i])

    i+=1

plt.savefig(os.path.join(path_output, "average_trajectories_2.pdf"))
plt.show()



