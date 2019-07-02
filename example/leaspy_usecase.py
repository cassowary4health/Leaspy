import os
from src.main import Leaspy
from src.inputs.data_reader import DataReader
from src.inputs.algo_settings import AlgoSettings
import torch
import matplotlib.pyplot as plt
import numpy as np

### 1. Fit a model

## Create leaspy object
leaspy = Leaspy("univariate")

## Data / Algo
test_data_dir = os.path.join("../tests", "_data")

# Load Data
data_path = os.path.join(test_data_dir, 'univariate_data.csv')
reader = DataReader()
data = reader.read(data_path)

# Path algo
path_to_algorithm_settings = os.path.join(test_data_dir,
                                          '_fit_univariatesigmoid_gradientdescent',
                                          "algorithm_settings.json")
algosettings = AlgoSettings(path_to_algorithm_settings)
algosettings.output_path = "../../output_leaspy/usecase/"

## FIT
leaspy.fit(data, algosettings)

## Play
# Print
print(leaspy.model)

# Plot
tensor_timepoints = torch.Tensor(np.linspace(-1, 1, 10)).reshape(-1, 1)
average = leaspy.model.compute_average(tensor_timepoints)
plt.plot(tensor_timepoints.detach().numpy(), average.detach().numpy())
plt.show()

## Save
leaspy.save("../../output_leaspy/usecase/model_usecase_1.json")



### 2. Load a model from parameters

fig, ax = plt.subplots(1,1)
tensor_timepoints = torch.Tensor(np.linspace(-1, 1, 10)).reshape(-1, 1)

## Load leaspy object
leaspy2 = Leaspy.from_model_settings("../../output_leaspy/usecase/model_usecase_1.json")

## Play
print(leaspy2.model)
average_before = leaspy2.model.compute_average(tensor_timepoints)

## Fit leaspy object
path_to_algorithm_settings = os.path.join(test_data_dir,
                                          '_fit_univariatesigmoid_mcmcsaem',
                                          "algorithm_settings.json")
algosettings = AlgoSettings(path_to_algorithm_settings)
algosettings.output_path = "../../output_leaspy/usecase/"
leaspy2.fit(data, algosettings)

## Play
print(leaspy2.model)
average_after = leaspy2.model.compute_average(tensor_timepoints)

## Save
leaspy2.save("../../output_leaspy/usecase/model_usecase_2.json")

## Plot
fig, ax = plt.subplots(1,1)
ax.plot(tensor_timepoints.detach().numpy(), average_before.detach().numpy(), c='blue')
ax.plot(tensor_timepoints.detach().numpy(), average_after.detach().numpy(), c ='red')
plt.show()


### 3. Predict a new patient / iterable of patients




