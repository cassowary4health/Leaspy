import os
from src.main import Leaspy
from src.inputs.data.data_reader import DataReader
from src.inputs.algorithm_settings import AlgorithmSettings
import matplotlib.pyplot as plt
from src.utils.output_manager import OutputManager
import matplotlib.cm as cm
import numpy as np

#%%

### FIT a longitudinal cohort

## 1. Initialize Data / Paramaters

leaspy_path = os.getcwd()
univariate_example_path = os.path.join(leaspy_path, "example", "data", "univariate")
output_path = os.path.join(leaspy_path, "..", "output_leaspy", "fit")

# 1.1 Instanciate Data object from csv file
data_path = os.path.join(univariate_example_path, "data.csv")
reader = DataReader()
data = reader.read(data_path)

# 1.2 Instanciate algosettings object from json file
algosettings_path = os.path.join(univariate_example_path, "algorithm_settings.json")
algosettings = AlgorithmSettings(algosettings_path)
algosettings.output_path = output_path

## 2. Launch leaspy

# 2.1 Instanciate leaspy object
leaspy = Leaspy("univariate")

# 2.2 Fit
leaspy.fit(data, algosettings)

# 2.3 See and save results
print(leaspy.model)
leaspy.save(os.path.join(output_path, "model.json"))


 #%%

### Predict a new patient

## 1. Load a model
leaspy = Leaspy.load(os.path.join(output_path, "model.json"))

## 2. Predict a new patient

# 2.1 Instanciate prediction settings object
prediction_algosettings_path = os.path.join(univariate_example_path, "predict_algorithm_settings.json")
prediction_settings = AlgorithmSettings(prediction_algosettings_path)

# 2.2 Predict
individual = data.subset([116])
individual_parameters = leaspy.personalize(individual, prediction_settings, seed=3)

# 2.3 Plot the Prediction
fig, ax = plt.subplots(1,1)
output_manager = OutputManager(path_output=None)
output_manager.plot_model_patient_reconstruction(individual[116], leaspy.model, individual_parameters, ax=ax)
plt.show()

#%%

# 2.4 Repeat for multiple patients
fig, ax = plt.subplots(1,1)
output_manager = OutputManager(path_output=None)
n_patients_to_plot = 5

indices_to_plot = data.indices[:n_patients_to_plot]

colors = cm.rainbow(np.linspace(0, 1, n_patients_to_plot+2))

for i, idx in enumerate(indices_to_plot):
    individual = data.subset([idx])
    individual_parameters = leaspy.personalize(individual, prediction_settings, seed=0)
    output_manager.plot_model_patient_reconstruction(individual[idx], leaspy.model, individual_parameters,
                                                     color=colors[i], ax=ax)

plt.show()



#%%



