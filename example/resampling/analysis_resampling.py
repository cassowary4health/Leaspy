
#%%
## Load experiment (calibrate + 1 personalize)
import torch
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from leaspy.io.outputs.individual_parameters import IndividualParameters
import pandas as pd
0
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

# Path input
input_directory = 'example/resampling/_inputs/'

# Path output
output_directory = 'example/resampling/_outputs/'
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


# TODO save and load that info somewhere
n_folds = 3
n_rep = 2
n_resampling_iter = n_folds*n_rep

#%% Load results from folds

# Load individual_parameters objects
individual_parameters_iter = []
for cv_iter in range(n_resampling_iter):
    individual_parameters_iter.append(IndividualParameters.load(os.path.join(path_output_personalize, "individual_parameters_{}.json".format(cv_iter))))

# Get calibrated models paths
model_paths = [os.path.join(path_output_calibrate, "fold_{}".format(i), "model_parameters.json") for i in range(n_resampling_iter)]

# Load leaspy objects
leaspy_iter = []
for i in range(n_folds*n_rep):
    leaspy = Leaspy.load(model_paths[i])
    leaspy_iter.append(leaspy)

#%% 1. Compute Trajectories and times of abnormalities

from leaspy.utils.posterior_analysis.general import compute_trajectory_of_population
from leaspy.utils.posterior_analysis.abnormality import get_age_at_abnormality_conversion

# Compute cutoffs (left to user)
dummy_cutoffs = {"Y{}".format(i):0.5 for i in range(4)}

#### No resampling
resampling_iter = 0
timepoints = np.linspace(30,100, 200).tolist()

# Get average trajectory
trajectory = compute_trajectory_of_population(timepoints,
                                              individual_parameters_iter[resampling_iter],
                                              leaspy_iter[resampling_iter])

# Get Time of abnormality
abnormality_ages = get_age_at_abnormality_conversion(dummy_cutoffs,
                                     individual_parameters_iter[resampling_iter],
                                      timepoints,
                                     leaspy_iter[resampling_iter])


#### With resampling
from leaspy.utils.resampling.general import compute_trajectory_of_population_resampling, get_age_at_abnormality_conversion_resampling

trajectory_resampling = compute_trajectory_of_population_resampling(timepoints,
                                                individual_parameters_iter,
                                                leaspy_iter)


times_resampling = get_age_at_abnormality_conversion_resampling(leaspy_iter,
                                                individual_parameters_iter,
                                                timepoints,
                                                dummy_cutoffs)


#%% 2. Compare 2 subgroups

## Get dummy groups
df = pd.read_csv(os.path.join(output_directory, "df.csv")).set_index(["ID","TIME"])
idx_group1 = np.array(df.index.unique("ID")[:100]).astype(str)
idx_group2 = np.array(df.index.unique("ID")[100:]).astype(str)

# TODO : type problem of json loaded idx of ind param
for i in range(n_resampling_iter):
    individual_parameters_iter[i]._indices = [str(idx) for idx in individual_parameters_iter[i]._indices]

# Get dummy cofactors
df_cofactors_dummy = pd.read_csv(os.path.join(input_directory, "df_cofactor.csv")).set_index('ID')
df_cofactors_dummy.index = df_cofactors_dummy.index.astype(str)

## With resampling
from leaspy.utils.resampling.general import compute_subgroup_statistics_resampling
stats_group1 = compute_subgroup_statistics_resampling(leaspy_iter,
                                 individual_parameters_iter,
                                 df_cofactors_dummy,
                                 idx_group1)

stats_group2 = compute_subgroup_statistics_resampling(leaspy_iter,
                                 individual_parameters_iter,
                                 df_cofactors_dummy,
                                 idx_group2)


#%% 3. Compute Correlations

## Without resampling
from leaspy.utils.posterior_analysis.statistical_analysis import compute_correlation
corr_value, corr_log10pvalue = compute_correlation(leaspy_iter[0], individual_parameters_iter[0], df_cofactors_dummy)

## With resampling

