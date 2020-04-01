
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

# Load indices
import json
with open(os.path.join(path_output, "resampling_indices.json"), "r") as json_file:
    indices_resampling = json.load(json_file)


###################################
## Computations
###################################


#%% 1. Compute Trajectories and times of abnormalities

from leaspy.utils.posterior_analysis.general import compute_trajectory_of_population
from leaspy.utils.posterior_analysis.abnormality import get_age_at_abnormality_conversion

# Compute cutoffs (left to user)
dummy_cutoffs = {"Y{}".format(i):0.1*(1+i) for i in range(4)}

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
from leaspy.utils.resampling.general import compute_correlation_resampling
correlation_resampling = compute_correlation_resampling(leaspy_iter, individual_parameters_iter, df_cofactors_dummy)
corr_value_mean, corr_log10pvalue_mean, corr_value_std, corr_log10pvalue_std, corr_log10pvalue_95percent = correlation_resampling



###################################
## Plots
###################################
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
alpha_lines = 0.5
width_lines = 2
fontsize = 20
colors = [
    'grey',
    'goldenrod',
    'darkviolet',
    'green',
    'firebrick',
    'royalblue',
    'darksalmon',
    'lightskyblue',
]


#%% 0. Spaghetti plots of raw data

# Get args
df = pd.read_csv(os.path.join(output_directory, "df.csv"))
data = Data.from_dataframe(df)

# Args : data
fig, ax = plt.subplots(data.dimension, 1, figsize=(16,16))

for patient in data:
    for dim in range(data.dimension):
        ax[dim].plot(patient.timepoints,
                     [patient.observations[i][dim] for i,_ in enumerate(patient.timepoints)],
                     alpha=0.6,
                     c=colors[dim])
        ax[dim].set_title(data.headers[dim])
plt.show()

#%% 1. Compute Trajectories and times of abnormalities

## Get args
cutoffs = dummy_cutoffs
# Diagnostic ages # TODO real ones
diagnostic_ages = {str(idx) : data.individuals[idx].timepoints[0] for idx in data.individuals.keys()}
# Average on resampling subgroup
diagnostic_ages_mean_resampling = [np.mean([diagnostic_ages[str(idx)] for idx in indices_resampling[str(resampling_iter)][0]]) for resampling_iter in range(n_resampling_iter)]
# Average of reparametrized age on resampling subgroup
from leaspy.utils.posterior_analysis.general import get_reparametrized_ages
diagnostic_ages_reparametrized_mean_resampling = {resampling_iter:
                                                 np.mean(list(get_reparametrized_ages(
                                                     {idx: [diagnostic_ages[idx]] for idx in
                                                      individual_parameters_iter[resampling_iter]._indices},
                                                     individual_parameters_iter[resampling_iter],
                                                     leaspy_iter[resampling_iter]).values())) for resampling_iter in range(n_resampling_iter)}


# Args : cutoffs, trajectory_resampling, diagnostic_ages_resampling
features = data.headers

fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 8))

# Average trajectory values
for resampling_iter in range(n_resampling_iter):
    idx_features_randomized = np.random.choice(list(range(len(features))), len(features), replace=False)
    features_randomized = np.array(features)[idx_features_randomized]
    for j, feature in zip(idx_features_randomized, features_randomized):
        ax.plot(timepoints,
                trajectory_resampling[resampling_iter][:,j],
                   linewidth=width_lines, alpha=alpha_lines, c=colors[j])


# Plot cutoffs and mean age of when it is reached
for j, feature in enumerate(cutoffs):
    for resampling_iter in range(n_resampling_iter):
        t_abnormal = times_resampling[resampling_iter,0,j]
        ax.hlines(cutoffs[feature], min(timepoints), t_abnormal, colors=colors[j])
        ax.vlines(t_abnormal, 0,  cutoffs[feature], colors=colors[j])

        height_text = 1.0

        ax.vlines(t_abnormal, cutoffs[feature], height_text,  colors=colors[j], linestyles='--')

# Diagnostic Age
for resampling_iter in range(n_resampling_iter):
    ax.vlines(diagnostic_ages_mean_resampling[resampling_iter], 0 ,0.82,
              color="black", linewidth = 6, alpha=0.3)
    ax.vlines(diagnostic_ages_reparametrized_mean_resampling[resampling_iter], 0, 0.82,
              color="red", linewidth=6, alpha=0.3)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(np.mean(diagnostic_ages_mean_resampling), 0.85, "PD Age",
        rotation=0, fontsize=fontsize+5, bbox=props,
        horizontalalignment="center")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(np.mean(list(diagnostic_ages_reparametrized_mean_resampling.values())), 0.95, "PD Age Reparam",
        rotation=0, fontsize=fontsize+5, bbox=props, c="red",
        horizontalalignment="center")

# Legend
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=colors[i], lw=8, alpha=0.9) for i in range(len(features))]
print(custom_lines)
ax.legend(custom_lines, features, loc='upper left', fontsize=fontsize)
plt.tight_layout()
plt.ylim(0,1.05)
plt.show()


#%% 2. Compare 2 subgroups




#%% 3. Compute Correlations

# Compute args
correlation_resampling = compute_correlation_resampling(leaspy_iter, individual_parameters_iter, df_cofactors_dummy)
corr_value_mean, corr_log10pvalue_mean, corr_value_std, corr_log10pvalue_std, corr_log10pvalue_95percent = correlation_resampling

# Args : cutoffs, trajectory_resampling, diagnostic_ages_resampling

# TODO : do a pls for the sources --> change over resampling iterations

import seaborn as sns
fig, ax = plt.subplots(2,2,figsize=(14,14))


sns.heatmap(corr_log10pvalue_mean,annot=True, ax=ax[0,0])
ax[0,0].set_title("Mean of log10 pvalue")

sns.heatmap(corr_log10pvalue_std,annot=True, ax=ax[1,0])
ax[1,0].set_title("Std of log10 pvalue")

corr_value_mean[corr_log10pvalue_mean>np.log10(0.05)]=np.nan
sns.heatmap(corr_value_mean,annot=True, ax=ax[0,1])
ax[0,1].set_title("Mean of value (resampling mean pvalue under 0.05)")

corr_value_mean[corr_log10pvalue_95percent>np.log10(0.05)]=np.nan
sns.heatmap(corr_value_mean,annot=True, ax=ax[1,1])
ax[1,1].set_title("Mean of value (resampling quantile 95% pvalue under 0.05)")

for axi in ax:
    for axij in axi:
        b, t = axij.get_ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        axij.set_ylim(b, t) # update the ylim(bottom, top) values

plt.show()
