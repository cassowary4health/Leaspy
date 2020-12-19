import os
import warnings

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch


# import matplotlib.backends.backend_pdf


# TODO: outdated -
class Plotting:

    def __init__(self, model, output_path='.', palette='tab10', max_colors=10):
        self.model = model

        # ---- Graphical options
        self.color_palette = None
        self.standard_size = (8, 4)
        self.linestyle = {'average_model': '-', 'individual_model': '-', 'individual_data': '-'}
        self.linewidth = {'average_model': 5, 'individual_model': 2, 'individual_data': 2}
        self.alpha = {'average_model': 0.5, 'individual_model': 1, 'individual_data': 1}
        self.output_path = output_path

        self.set_palette(palette, max_colors)

    def update_model(self, model):
        self.model = model

    def set_palette(self, palette, max_colors=None):
        """
        Set palette of plots

        Parameters
        ----------
        palette : string (palette name) or matplotlib.colors.Colormap (ListedColormap or LinearSegmentedColormap)

        max_colors : positive int or None (default, corresponding to model nb of features)
            Only used if palette is a string
        """

        if isinstance(palette, mpl.colors.Colormap):
            self.color_palette = palette
        else:
            if max_colors is None:
                max_colors = self.model.dimension
            self.color_palette = cm.get_cmap(palette, max_colors)

    def colors(self, at=None):
        """
        Wrapper over color_palette iterator to get colors

        Parameters
        ----------
        at : any legit color_palette arg (int, float or iterable of any of these) or None (default)
            if None returns all colors of palette upto model dimension

        Returns
        -------
        colors : single color tuple (RGBA) or np.array of RGBA colors (number of colors x 4)
        """
        if at is None:
            at = [i % self.color_palette.N for i in range(self.model.dimension)]

        return self.color_palette(at)

    def handle_kwargs_begin(self, kwargs):
        # /!\ Break if model is not initialized
        if not self.model.is_initialized:
            raise ValueError("Please initialize the model before plotting")

        # ---- Colors
        colors = kwargs.get('color', self.colors())

        # ---- Labels
        labels = kwargs.get('labels', self.model.features)

        # ---- Ax
        ax = kwargs.get('ax', None)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize', self.standard_size))

        # ---- Handle ylim
        if self.model.name in ['logistic', 'logistic_parallel']:
            ax.set_ylim(0, 1)

        return colors, ax, labels

    def handle_kwargs_end(self, ax, kwargs, colors, labels, dimension=None):
        # ---- Legend
        if dimension is None:
            dimension = self.model.dimension
        custom_lines = [mpl.lines.Line2D([0], [0], color=colors[i], lw=4) for i in range(dimension)]
        ax.legend(custom_lines, labels, title='Features')
        # ax.legend(title='Features')
        ax.set_ylabel('Normalized score')

        # ---- Save
        if 'save_as' in kwargs.keys():
            plt.savefig(os.path.join(self.output_path, kwargs['save_as']))

    def average_trajectory(self, **kwargs):
        """
        Plot the population average trajectories. They are parametrized by the population parameters derivated
        during the calibration.

        Parameters
        ----------
        kwargs
            * alpha: float, default 0.6
                Matplotlib's transparency option. Must be in [0, 1].
            * linestyle: {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
                Matplotlib's linestyle option.
            * linewidth: float
                Matplotlib's linewidth option.
            * colors: list of str
                Contains matplotlib compatible colors.
            * labels: list of str
                Used to rename features in the plot.
            * ax: matplotlib.axes.Axes
                Axes object to modify, instead of creating a new one.
            * figsize: tuple of int
                The figure's size.
            * save_as: str, default None
                Path to save the figure.
            * title: str

        Returns
        -------
        ax: matplotlib.axes.Axes
        """
        # ---- Input manager
        alpha = kwargs.get('alpha', self.alpha['average_model'])
        linestyle = kwargs.get('linestyle', self.linestyle['average_model'])
        linewidth = kwargs.get('linewidth', self.linewidth['average_model'])

        colors, ax, labels = self.handle_kwargs_begin(kwargs)

        # ---- Get timepoints
        mean_time = self.model.parameters['tau_mean']
        std_time = max(self.model.parameters['tau_std'], 4)
        timepoints = np.linspace(mean_time - 3 * std_time, mean_time + 6 * std_time, 100)
        timepoints = torch.tensor([timepoints])

        # ---- Compute average trajectory
        mean_trajectory = self.model.compute_mean_traj(timepoints).detach().numpy()

        # ---- plot it for each dimension
        for i in range(mean_trajectory.shape[-1]):
            ax.plot(timepoints[0, :].detach().numpy(),
                    mean_trajectory[0, :, i],
                    linewidth=linewidth,
                    linestyle=linestyle,
                    alpha=alpha,
                    c=colors[i],
                    label=labels[i])

        self.handle_kwargs_end(ax, kwargs, colors, labels)

        # ---- Title & labels
        ax.set_title('Average trajectories')
        ax.set_xlabel('Age')

        return ax

    def patient_observations(self, data, patients_idx='all', **kwargs):
        # ---- Input manager
        alpha = kwargs.get('alpha', self.alpha['individual_data'])
        linestyle = kwargs.get('linestyle', self.linestyle['individual_data'])
        linewidth = kwargs.get('linewidth', self.linewidth['individual_data'])
        marker = kwargs.get('marker', 'o')
        markersize = kwargs.get('markersize', '3')
        if 'patient_IDs' in kwargs.keys():
            warnings.warn("Keyword argument <patient_IDs> is deprecated! Use <patients_idx> instead.",
                          warnings.DeprecationWarning)
            patients_idx = kwargs.get('patient_IDs')

        if patients_idx == 'all':
            patients_idx = list(data.iter_to_idx.values())

        if self.model.is_initialized:
            assert data.headers == self.model.features
            colors, ax, labels = self.handle_kwargs_begin(kwargs)
        else:
            colors = kwargs.get('labels', self.color_palette([i % self.color_palette.N for i in range(data.dimension)]))
            labels = kwargs.get('labels', data.headers)
            ax = kwargs.get('ax', None)
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize', self.standard_size))

        dimension = data.dimension

        if type(patients_idx) is str:
            patients_idx = [patients_idx]

        # ---- Plot
        for idx in patients_idx:
            indiv = data.get_by_idx(idx)
            timepoints = indiv.timepoints
            observations = np.array(indiv.observations)

            for dim in range(data.dimension):
                not_nans_idx = np.array(1-np.isnan(observations[:, dim]),dtype=bool)

                ax.plot(np.array(timepoints)[not_nans_idx],
                        observations[:, dim][not_nans_idx],
                        marker=marker,
                        markersize=markersize,
                        c=colors[dim],
                        linewidth=linewidth,
                        linestyle=linestyle,
                        alpha=alpha,
                        label=labels[dim])

        self.handle_kwargs_end(ax, kwargs, colors, labels, dimension=dimension)

        # ---- Title & labels
        ax.set_title('Observations')
        ax.set_xlabel('Age')

        return ax

    def patient_observations_reparametrized(self, data, individual_parameters, patients_idx='all', **kwargs):
        # ---- Input manager
        alpha = kwargs.get('alpha', self.alpha['individual_data'])
        linestyle = kwargs.get('linestyle', self.linestyle['individual_data'])
        linewidth = kwargs.get('linewidth', self.linewidth['individual_data'])
        marker = kwargs.get('marker', 'o')
        markersize = kwargs.get('markersize', '3')
        if 'patient_IDs' in kwargs.keys():
            warnings.warn("Keyword argument <patient_IDs> is deprecated! Use <patients_idx> instead.",
                          warnings.DeprecationWarning)
            patients_idx = kwargs.get('patient_IDs')

        if patients_idx == 'all':
            patients_idx = list(data.iter_to_idx.values())

        if self.model.is_initialized:
            assert data.headers == self.model.features
            colors, ax, labels = self.handle_kwargs_begin(kwargs)
        else:
            colors = kwargs.get('labels', self.color_palette([i % self.color_palette.N for i in range(data.dimension)]))
            labels = kwargs.get('labels', data.headers)
            ax = kwargs.get('ax', None)
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize', self.standard_size))

        dimension = data.dimension

        if type(patients_idx) is str:
            patients_idx = [patients_idx]

        # ---- Plot
        t0 = self.model.parameters['tau_mean'].item()
        ip_df = individual_parameters.to_dataframe()
        ip_df = ip_df.join(data.to_dataframe().reset_index()[['TIME']])
        ip_df['TIME_rep'] = np.exp(ip_df['xi'].values) * (ip_df['TIME'].values - ip_df['tau'].values) + t0

        for idx in patients_idx:
            indiv = data.get_by_idx(idx)
            timepoints = ip_df.loc[idx, 'TIME_rep'].values
            observations = np.array(indiv.observations)

            for dim in range(dimension):
                not_nans_idx = np.array(1-np.isnan(observations[:, dim]), dtype=bool)

                ax.plot(np.array(timepoints)[not_nans_idx],
                        observations[:, dim][not_nans_idx],
                        marker=marker,
                        markersize=markersize,
                        c=colors[dim],
                        linewidth=linewidth,
                        linestyle=linestyle,
                        alpha=alpha,
                        label=labels[dim])

        self.handle_kwargs_end(ax, kwargs, colors, labels, dimension=dimension)

        # ---- Title & labels
        ax.set_title('Observations')
        ax.set_xlabel('Reparametrized age')

        return ax

    def patient_trajectories(self, data, individual_parameters, patients_idx, **kwargs):
        # ---- Input manager
        alpha = kwargs.get('alpha', self.alpha['individual_model'])
        linestyle = kwargs.get('linestyle', self.linestyle['individual_model'])
        linewidth = kwargs.get('linewidth', self.linewidth['individual_model'])

        if 'patient_IDs' in kwargs.keys():
            warnings.warn("Keyword argument <patient_IDs> is deprecated! Use <patients_idx> instead.",
                          warnings.DeprecationWarning)
            patients_idx = kwargs.get('patient_IDs')

        if patients_idx == 'all':
            patients_idx = list(data.iter_to_idx.values())

        if self.model.is_initialized:
            assert data.headers == self.model.features
            colors, ax, labels = self.handle_kwargs_begin(kwargs)
        else:
            colors = kwargs.get('labels', self.color_palette([i % self.color_palette.N for i in range(data.dimension)]))
            labels = kwargs.get('labels', data.headers)
            ax = kwargs.get('ax', None)
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize', self.standard_size))

        dimension = data.dimension

        if type(patients_idx) is str:
            patients_idx = [patients_idx]

        for idx in patients_idx:
            indiv = data.get_by_idx(idx)
            timepoints = indiv.timepoints
            min_t, max_t = min(timepoints), max(timepoints)
            timepoints = np.linspace(min_t - (max_t - min_t) / 2, max_t + (max_t - min_t) / 2, 100)
            t = torch.tensor(timepoints, dtype=torch.float32).unsqueeze(0)
            ip = {key: torch.tensor(val).unsqueeze(0) for key, val in
                  individual_parameters._individual_parameters[idx].items()}
            trajectory = self.model.compute_individual_tensorized(t, ip).squeeze(0)
            for dim in range(dimension):
                ax.plot(timepoints,
                        trajectory.detach().numpy()[:, dim],
                        c=colors[dim],
                        linewidth=linewidth,
                        linestyle=linestyle,
                        alpha=alpha,
                        )

        self.handle_kwargs_end(ax, kwargs, colors, labels, dimension=dimension)

        # ---- Title & labels
        ax.set_title('Individual trajectories')
        ax.set_xlabel('Age')

        return ax

