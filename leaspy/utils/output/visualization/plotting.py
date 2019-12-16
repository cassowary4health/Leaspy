import os
import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.cm as cm

#import matplotlib.backends.backend_pdf
from matplotlib.lines import Line2D



class Plotting():

    def __init__(self, model):
        self.update_model(model)

        # Default plot parameters
        self.max_color = 8
        self.color_palette = cm.get_cmap('Set2', self.max_color)
        self.colors = self.color_palette(range(self.max_color))
        self.standard_size = (11, 6)

        self.linestyle = {'average_model' : '-',
                          'individual_model' : '-', 'individual_data': '--'}

        self.linewidth = {'average_model' : 5,
                          'individual_model' : 1, 'individual_data': 1}

        self.alpha = {'average_model' : 0.5,
                          'individual_model' : 1, 'individual_data': 1}

    def update_model(self, model):
        self.model = model


    def handle_kwargs_begin(self, kwargs):

        # Check if model is intialized
        # Break if model is not initialized
        if not self.model.is_initialized:
            raise ValueError("Please initialize the model before plotting")

        # Colors
        colors = kwargs['color'] if 'color' in kwargs.keys() else self.colors

        # linestyle / linewidth / alpha
        linestyle = kwargs['linestyle'] if 'linestyle' in kwargs.keys() else self.linestyle
        linewidth = kwargs['linewidth'] if 'linewidth' in kwargs.keys() else self.linewidth
        alpha = kwargs['alpha'] if 'alpha' in kwargs.keys() else self.alpha

        # Ax
        ax = kwargs['ax'] if 'ax' in kwargs.keys() else None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.standard_size)

        # Handle ylim
        if self.model.name in ['logistic', 'logistic_parallel']:
            ax.set_ylim(0, 1)

        return colors, ax, linestyle, linewidth, alpha

    def handle_kwargs_end(self, ax, kwargs):

        # Labels
        labels = kwargs['labels'] if 'labels' in kwargs.keys() else self.model.features
        # Legend
        custom_lines = [Line2D([0], [0], color=self.colors[i%self.max_color], lw=4) for i in range((self.model.dimension))]
        print(custom_lines)
        ax.legend(custom_lines, labels, loc='upper right')

        # Title
        title = kwargs['title'] if 'title' in kwargs.keys() else None
        if title is not None:
            ax.set_title(title)

        # Save
        if 'save_as' in kwargs.keys():
            plt.savefig(os.path.join(self.output_path, kwargs['save_as']))

        return ax


    def average_trajectory(self, **kwargs):

        colors, ax, linestyle, linewidth, alpha = self.handle_kwargs_begin(kwargs)

        # Get timepoints
        mean_time = self.model.parameters['tau_mean']
        std_time = max(self.model.parameters['tau_std'], 4)
        timepoints = np.linspace(mean_time - 3 * std_time, mean_time + 6 * std_time, 100)
        timepoints = torch.Tensor([timepoints])

        # Compute average trajectory
        mean_trajectory = self.model.compute_mean_traj(timepoints).detach().numpy()

        # plot it for each dimension
        for i in range(mean_trajectory.shape[-1]):
            ax.plot(timepoints[0, :].detach().numpy(), mean_trajectory[0, :, i],
                    linewidth=linewidth['average_model'],
                    linestyle=linestyle['average_model'],
                    alpha=alpha['average_model'],
                    c=colors[i])  # , c=colors[i])

        ax = self.handle_kwargs_end(ax, kwargs)


    def patient_observations(self, result, patient_IDs, **kwargs):

        colors, ax, linestyle, linewidth, alpha = self.handle_kwargs_begin(kwargs)

        if type(patient_IDs) is not list:
            patient_IDs = [patient_IDs]

        for idx in patient_IDs:
            indiv = result.data.get_by_idx(idx)
            timepoints = indiv.timepoints
            observations = np.array(indiv.observations)

            for dim in range(self.model.dimension):
                not_nans_idx = np.array(1-np.isnan(observations[:, dim]),dtype=bool)

                ax.plot(np.array(timepoints)[not_nans_idx], observations[:, dim][not_nans_idx], c=colors[dim],
                        linewidth=linewidth['individual_data'],
                        linestyle=linestyle['individual_data'],
                        alpha=alpha['individual_data'],)

        ax = self.handle_kwargs_end(ax, kwargs)


    def patient_trajectories(self, result, patient_IDs, **kwargs):

        colors, ax, linestyle, linewidth, alpha = self.handle_kwargs_begin(kwargs)

        if type(patient_IDs) is not list:
            patient_IDs = [patient_IDs]

        for idx in patient_IDs:
            indiv = result.data.get_by_idx(idx)
            timepoints = indiv.timepoints
            t = torch.Tensor(timepoints).unsqueeze(0)
            indiv_parameters = result.get_torch_individual_parameters(idx)
            trajectory = self.model.compute_individual_tensorized(t, indiv_parameters).squeeze(0)
            for dim in range(self.model.dimension):
                ax.plot(np.array(timepoints), trajectory.detach().numpy()[:, dim], c=colors[dim],
                        linewidth=linewidth['individual_model'],
                        linestyle=linestyle['individual_model'],
                        alpha=alpha['individual_model'],
                        )

        ax = self.handle_kwargs_end(ax, kwargs)










