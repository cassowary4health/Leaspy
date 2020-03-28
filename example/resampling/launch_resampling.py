
import os
import pandas as pd
from leaspy import Leaspy, Data, AlgorithmSettings
from leaspy.utils.parallel.leaspy_parallel import leaspy_parallel_calibrate, leaspy_parallel_personalize
import json
import numpy as np

#%% Parameters

# N iter
n_iter = 500

# Model parameters
features = ["Y_{}".format(i) for i in range(4)]
source_dimension = 4
seed = 0
leaspy_model = "logistic"
n_jobs = 3
resampling_method = "RepCV"
n_rep = 2
n_folds = 3
n_resampling_iter = n_rep*n_folds

# Paths
output_directory = 'example/resampling/_outputs/'
experiment_folder = "Experiment_0"
path_output = os.path.join(output_directory, experiment_folder)
path_output_calibrate = os.path.join(output_directory, experiment_folder, "calibrate")

# Create if needed
if not os.path.exists(path_output_calibrate):
    os.makedirs(path_output_calibrate)

#%% Load Data
input_directory = 'example/resampling/_inputs/'
df = pd.read_csv(os.path.join(input_directory, 'data.csv')).set_index(['ID','TIME'])

#%% Create the Resempling procedure and get indices in train/test for each run

from sklearn.model_selection import RepeatedKFold
X = df.index.unique('ID')

if resampling_method=="RepCV":
    skf = RepeatedKFold(n_splits=n_folds, n_repeats=n_rep, random_state=seed)
    resampling_indices = {}
    for j, (train_index, test_index) in enumerate(skf.split(X)):
       patients_train, patients_test = np.array(X)[train_index], np.array(X)[test_index]
       resampling_indices[j] = (patients_train.tolist(), patients_test.tolist())
else:
    raise NotImplementedError("Other resampling methods than RepCV not yet implemented")

#%% Run Leaspy

# Data as list
data_iter = []
for cv_iter in range(n_folds*n_rep):
    df_split = df.loc[resampling_indices[cv_iter][0]]
    data = Data.from_dataframe(df_split.reset_index())
    data_iter.append(data)

# Also settings as list
algo_settings_iter = []
algo_settings = AlgorithmSettings('mcmc_saem', n_iter=n_iter, initialization_method="random", seed=seed)
for i in range(n_rep*n_folds):
    algo_settings_iter.append(algo_settings)

# Save experiment infos
df.to_csv(os.path.join(output_directory, "df.csv"))
with open(os.path.join(path_output, "resampling_indices.json"), "w") as json_file:
    json.dump(resampling_indices, json_file)
algo_settings.save(os.path.join(path_output, "calibrate", "algo_settings.json"))

def leaspy_factory(i):
    ll = Leaspy(leaspy_model)
    ll.model.load_hyperparameters({'source_dimension': source_dimension})
    return ll

def leaspy_callback(leaspy_model, i):
    if not os.path.isdir(os.path.join(path_output, 'calibrate','fold_'+str(i), 'logs')):
        os.makedirs(os.path.join(path_output, 'calibrate','fold_'+str(i), 'logs'))
    leaspy_model.save(os.path.join(path_output, 'calibrate','fold_'+str(i), 'model_parameters.json'))
    return leaspy_model

# Calibrate the model x Resampling
leaspy_parallel_calibrate(data_iter, algo_settings_iter, leaspy_factory, leaspy_callback, n_jobs=n_jobs)


#%% Personalize

# paths and parameters
personalize_algorithm = "scipy_minimize"
n_iter_personalize = 100
name = "personalize_0"
path_output_personalize = os.path.join(output_directory, experiment_folder, "personalize", name)
if not os.path.exists(path_output_personalize):
    os.makedirs(path_output_personalize)

# Get calibrated models paths
model_paths = [os.path.join(path_output_calibrate, "fold_{}".format(i), "model_parameters.json") for i in range(n_resampling_iter)]

# Load estimated leaspy models
leaspy_iter = []
for i in range(n_resampling_iter):
    leaspy = Leaspy.load(model_paths[i])
    leaspy_iter.append(leaspy)

# Algo settings iter
algo_settings_personalize_iter = []
algo_settings_personalize = AlgorithmSettings(personalize_algorithm, seed=seed, n_iter=n_iter_personalize)
for i in range(n_resampling_iter):
    algo_settings_personalize_iter.append(algo_settings_personalize)

# Save algo settings
algo_settings_personalize.save(os.path.join(path_output_personalize, "algo_settings_personalize.json"))

def leaspy_res_cb(ind_params, i):
    # Save result somewhere ????
    path_res = os.path.join(path_output_personalize, "individual_parameters_{}.json".format(i))
    ind_params.save(path_res)
    print(ind_params._indices)
    print(ind_params._individual_parameters)

res_parallel = leaspy_parallel_personalize(leaspy_iter, data_iter, algo_settings_personalize_iter, leaspy_res_cb, n_jobs)
