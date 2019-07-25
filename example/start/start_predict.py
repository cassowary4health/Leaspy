import os
from src.main import Leaspy
from src.inputs.data.data import Data
from src.inputs.algorithm_settings import AlgorithmSettings



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

# Plot
# plot
from src.visualization.visualization_toolbox import VisualizationToolbox
import matplotlib.pyplot as plt
import matplotlib.cm as cm
visualization_toolbox = VisualizationToolbox()

# Personalize
prediction_settings = AlgorithmSettings(os.path.join(os.path.dirname(__file__), '_inputs', 'prediction_settings_mcmc.json'))
data_predict = leaspy.personalize(data, prediction_settings=prediction_settings)

fig, ax = plt.subplots(1,1,figsize=(20,10))
visualization_toolbox.plot_patients(leaspy.model, data_predict, indices=list(data_predict.individuals.keys())[:5], ax=ax)
visualization_toolbox.plot_mean(leaspy.model, ax=ax)
plt.show()


# Simulate
data_synthetic = leaspy.simulate(data, n_individuals=10)

fig, ax = plt.subplots(1,1,figsize=(20,10))
visualization_toolbox.plot_patients(leaspy.model, data_synthetic, indices=[5,6,7], ax=ax)
visualization_toolbox.plot_mean(leaspy.model, ax=ax)
plt.show()
