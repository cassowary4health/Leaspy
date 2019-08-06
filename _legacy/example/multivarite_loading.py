

import os
from src.main import Leaspy
from src.inputs.data.data_reader import DataReader
from src.inputs.algorithm_settings import AlgorithmSettings
import torch
import numpy as np

# Settings

project_dir = os.path.join(os.path.dirname(__file__), '..')



#%%
data_dir = os.path.join(project_dir, "example/data/multivariate/")
path_to_model_parameters = os.path.join(data_dir, 'model_settings_univariate.json')
path_to_algo_parameters = os.path.join(data_dir, "algorithm_settings.json")
algo_settings = AlgorithmSettings(path_to_algo_parameters)

# Output folder
path_output = os.path.join(project_dir, '../output_leaspy/quickstart/')
algo_settings.output_path = path_output

# Data
data_path = os.path.join(data_dir, 'data2_tiny.csv')
reader = DataReader()
data = reader.read(data_path)


# Leaspy instanciate
leaspy = Leaspy.load(path_to_model_parameters)

#%% Perform operations

# Compute average
tensor_timepoints = torch.Tensor(np.linspace(-1, 1, 10)).reshape(-1, 1)


#%%

average = leaspy.model.compute_average(tensor_timepoints)

import matplotlib.pyplot as plt
plt.plot(average.detach().numpy())
plt.show()
