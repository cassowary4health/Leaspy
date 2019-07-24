
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pandas as pd
import numpy as np
import torch
from src.inputs.data.dataset import Dataset

class VisualizationToolbox():

    def __init__(self):
        pass

    # Plot model directly
    def plot_mean(self, model):
        NotImplementedError

    def plot_patients(self, model, data, indices, ax=None):

        # Get dataset from data
        dataset = Dataset(data=data, model=model, algo=None)

        # Instanciate realizations
        realizations = data.realizations

        colors = cm.rainbow(np.linspace(0, 1, len(indices)+2))
        fig, ax = plt.subplots(1, 1)


        patient_values = model.compute_individual_tensorized(dataset, realizations)

        # TODO only the 10 first, change that to specified indices

        dict_correspondence = {}
        for i, idx in enumerate(data.individuals.keys()):
            dict_correspondence[idx] = i

        for p,idx in enumerate(indices):
            i = dict_correspondence[idx]
            model_value = patient_values[i,0:dataset.nb_observations_per_individuals[i],:]
            score = dataset.values[i,0:dataset.nb_observations_per_individuals[i],:]
            ax.plot(dataset.timepoints[i,0:dataset.nb_observations_per_individuals[i]].detach().numpy(),
                    model_value.detach().numpy(), c=colors[p])
            ax.plot(dataset.timepoints[i,0:dataset.nb_observations_per_individuals[i]].detach().numpy(),
                    score.detach().numpy(), c=colors[p], linestyle='--',
                    marker='o')

        # Plot average model
        #tensor_timepoints = torch.Tensor(np.linspace(data.time_min, data.time_max, 40).reshape(-1,1))
        #model_average = model.compute_average(tensor_timepoints)
        #ax.plot(tensor_timepoints.detach().numpy(), model_average.detach().numpy(), c='black', linewidth=4, alpha=0.3)

        return 0


    # Plot distributions
    def plot_distributions_individual_parameter(self, data, real_ind_name, covariable=None):
        raise NotImplementedError
