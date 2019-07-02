import os
from src.main import Leaspy
from src.inputs.data_reader import DataReader
from src.inputs.algo_settings import AlgoSettings

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


#%%

### Predict a new patient

## 1. Load a model
leaspy = Leaspy.from_model_settings(os.path.join(output_path, "model.json"))

## 2. Predict a new patient

# 2.1 Instanciate prediction settings object
prediction_algosettings_path = os.path.join(univariate_example_path, "predict_algorithm_settings.json")
prediction_settings = AlgoSettings(prediction_algosettings_path)

# 2.2 Predict
individual = data.subset([116])
individual_parameters = leaspy.predict(individual, prediction_settings, seed=0)