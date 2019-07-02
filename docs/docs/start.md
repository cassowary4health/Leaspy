---
title: Documentation
summary: User documentation
---



# FIT a longitudinal cohort
```python
import os
from src.inputs.data_reader import DataReader
from src.inputs.algo_settings import AlgoSettings
from src.main import Leaspy

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
algosettings = AlgoSettings(algosettings_path)
algosettings.output_path = output_path

## 2. Launch leaspy

# 2.1 Instanciate leaspy object
leaspy = Leaspy("univariate")

# 2.2 Fit
leaspy.fit(data, algosettings)

# 2.3 See and save results
print(leaspy.model)
leaspy.save(os.path.join(output_path, "model.json"))
```




# Predict a new patient
```python
import os
from src.main import Leaspy
from src.inputs.algo_settings import AlgoSettings
from src.utils.output_manager import OutputManager
from src.inputs.data_reader import DataReader
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

## 1. Initialize Data / Paramaters

leaspy_path = os.getcwd()
univariate_example_path = os.path.join(leaspy_path, "example", "data", "univariate")
output_path = os.path.join(leaspy_path, "..", "output_leaspy", "fit")

# 1.1 Instanciate Data object from csv file

data_path = os.path.join(univariate_example_path, "data.csv")
reader = DataReader()
data = reader.read(data_path)

## 2. Load leaspy from parameters
leaspy = Leaspy.from_model_settings(os.path.join(output_path, "model.json"))



## 3. PREDICT a new patient

# 2.1 Instanciate prediction settings object
prediction_algosettings_path = os.path.join(univariate_example_path, "predict_algorithm_settings.json")
prediction_settings = AlgoSettings(prediction_algosettings_path)

# 2.2 Predict
individual = data.subset([116])
individual_parameters = leaspy.predict(individual, prediction_settings, seed=4)

# 2.3 Plot the Prediction
fig, ax = plt.subplots(1,1)
output_manager = OutputManager(path_output=None)
output_manager.plot_model_patient_reconstruction(individual[116], leaspy.model, individual_parameters, ax=ax)
plt.show()

# 2.4 Repeat for multiple patients

fig, ax = plt.subplots(1,1)
output_manager = OutputManager(path_output=None)
n_patients_to_plot = 5

indices_to_plot = data.indices[:n_patients_to_plot]

colors = cm.rainbow(np.linspace(0, 1, n_patients_to_plot+2))

for i, idx in enumerate(indices_to_plot):
    individual = data.subset([idx])
    individual_parameters = leaspy.predict(individual, prediction_settings, seed=0)
    output_manager.plot_model_patient_reconstruction(individual[idx], leaspy.model, individual_parameters,
                                                     color = colors[i] ,ax=ax)

plt.show()
```




## Simulate new parameters

```python
import leaspy as lp

simulation_settings = read_prediction_settings(simulation_settings_path)

model = lp.from_model_settings(path_to_parameters)
new_individuals = model.simulate(simulation_settings)
```
