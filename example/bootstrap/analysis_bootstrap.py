
#%%
## Load experiment (calibrate + 1 personalize)

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# Leaspy
run_aramis_machine = 0
if run_aramis_machine:
    sys.path.append("/network/lustre/dtlake01/aramis/users/raphael.couronne/projects/leaspy")
else:
    sys.path.append('/Users/raphael.couronne/Programming/ARAMIS/Projects/LEASPY_project/leaspy')

from leaspy import Leaspy, Data, AlgorithmSettings, Plotter, Dataset
from leaspy.utils.parallel.leaspy_parallel import leaspy_parallel_calibrate, leaspy_parallel_personalize


###################################
## Loading Experiments
###################################

#%% Parameters

# Path output
output_directory = 'example/bootstrap/_outputs/'
experiment_folder = "Experiment_0"
path_output = os.path.join(output_directory, experiment_folder)

# Path calibrate
path_output_calibrate = os.path.join(output_directory, experiment_folder, "calibrate")

# Path personalize
name = "personalize_0"
path_output_personalize = os.path.join(output_directory, experiment_folder, "personalize", name)

# Path analysis
path_analysis = os.path.join(path_output, "analysis")
if not os.path.exists(path_analysis):
    os.makedirs(path_analysis)


n_folds = 3
n_rep = 2
n_resampling_iter = n_folds*n_rep

#%% Load data from folds

# Load Calibration
#from analysis.utils import load_experiment_data
#experiment, df_all, indices_train_test, model_parameters, model_paths, df = load_experiment_data(path_output)
#n_folds = experiment['parameters']['n_folds']
#n_rep = experiment['parameters']['n_rep']
#features = experiment['parameters']['features']


# Load individual_parameters objects
from leaspy.io.outputs.individual_parameters import IndividualParameters
individual_parameters = []
for cv_iter in range(n_resampling_iter):
    individual_parameters.append(IndividualParameters.load(os.path.join(path_output_personalize, "individual_parameters_{}.json".format(cv_iter))))

# Get calibrated models paths
model_paths = [os.path.join(path_output_calibrate, "fold_{}".format(i), "model_parameters.json") for i in range(n_resampling_iter)]

# Load leaspy objects
leaspy_iter = []
for i in range(n_folds*n_rep):
    leaspy = Leaspy.load(model_paths[i])
    leaspy_iter.append(leaspy)



#%%
## 1 run

resampling_iter = 0

# Get average trajectory
import torch
#from leaspy.utils.posterior_analysis.general import *
from leaspy.utils.posterior_analysis.general import compute_trajectory_of_population
timepoints = torch.tensor(np.linspace(30,100))
trajectory = compute_trajectory_of_population(leaspy_iter[resampling_iter],
                                              individual_parameters[resampling_iter],
                                              timepoints)

# Get Time of conversion

def get_age_at_abnormality_conversion(abnormality_thresholds, individual_parameter, leaspy):
    #TODO : Raphael
    return 0