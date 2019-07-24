import os
from src.main import Leaspy
from src.inputs.data.data import Data
from src.inputs.algorithm_settings import AlgorithmSettings
from src.algo.algo_factory import AlgoFactory
from src.inputs.data.dataset import Dataset
import matplotlib.cm as cm
import numpy as np

def plot_patient_reconstructions_predict(maximum_patient_number, data, model, realizations, ax):
    colors = cm.rainbow(np.linspace(0, 1, maximum_patient_number + 2))

    patient_values = model.compute_individual_tensorized(data, realizations)

    for i in range(10):
        model_value = patient_values[i, 0:data.nb_observations_per_individuals[i], :]
        score = data.values[i, 0:data.nb_observations_per_individuals[i], :]
        ax.plot(data.timepoints[i, 0:data.nb_observations_per_individuals[i]].detach().numpy(),
                model_value.detach().numpy(), c=colors[i])
        ax.plot(data.timepoints[i, 0:data.nb_observations_per_individuals[i]].detach().numpy(),
                score.detach().numpy(), c=colors[i], linestyle='--',
                marker='o')

        if i > maximum_patient_number:
            break



# Inputs
data = Data.from_csv_file(os.path.join(os.path.dirname(__file__), '_inputs', 'data_tiny.csv'))
algo_settings = AlgorithmSettings(os.path.join(os.path.dirname(__file__), '_inputs', 'algorithm_settings.json'))

# Initialize
leaspy = Leaspy("multivariate_parallel")
#leaspy = Leaspy("multivariate")

# Fit the model on the data
leaspy.fit(data, algorithm_settings=algo_settings)

# Save the model
path_to_saved_model = os.path.join(os.path.dirname(__file__), '_outputs', 'fitted_multivariate_model.json')
leaspy.save(path_to_saved_model)

# Personalize

"""
prediction_settings = AlgorithmSettings(os.path.join(os.path.dirname(__file__), '_inputs', 'prediction_settings_mcmc.json'))
data_res = leaspy.personalize(data, prediction_settings=prediction_settings)

algorithm = AlgoFactory.algo(prediction_settings)
dataset = Dataset(data, algo=algorithm, model=leaspy.model)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1,figsize=(20,10))
for i in range(50):
    plot_patient_reconstructions_predict(5 ,dataset, leaspy.model, data_res.personalize_output["distribution"][i], ax=ax)
plt.savefig("predict.pdf")
plt.show()"""


# Simulate
data_synthetic = leaspy.simulate(data, n_individuals=10)
