import os
import math
from typing import Optional

import pandas as pd
import numpy as np
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import matplotlib.backends.backend_pdf

from leaspy.io.data.dataset import Dataset
from leaspy.exceptions import LeaspyInputError


class Plotter:
    """
    Class defining some plotting tools.

    Parameters
    ----------
    output_path : str (optional)
        Folder where plots will be saved.
        If None, default to current working directory.
    """

    def __init__(self, output_path=None):
        # TODO : Do all the check up if the path exists, and if yes, if removing or not
        if output_path is None:
            output_path = os.getcwd()
        self.output_path = output_path

        # https://stackoverflow.com/questions/40395659/view-and-then-close-the-figure-automatically-in-matplotlib/40395799
        self._block = False
        self._show = True

    def plt_show(self):
        if self._show:
            plt.show(block=self._block)

    def plot_mean_trajectory(self, model, *, attribute_type: Optional[str] = None, **kwargs):
        # colors = kwargs['color'] if 'color' in kwargs.keys() else cm.gist_rainbow(np.linspace(0, 1, model.dimension))

        labels = model.features
        fig, ax = plt.subplots(1, 1, figsize=(11, 6))

        #colors = color_palette(range(8))

        colors = cm.get_cmap("tab20").colors

        try:
            iter(model)
        except Exception:

            # Break if model is not initialized
            if not model.is_initialized:
                raise LeaspyInputError("Please initialize the model before plotting")

            # not iterable
            if 'logistic' in model.name:
                plt.ylim(0, 1)

            mean_time = model.parameters['tau_mean']
            std_time = max(model.parameters['tau_std'], 4)
            timepoints = np.linspace(mean_time - 3 * std_time, mean_time + 6 * std_time, 100)
            timepoints = torch.tensor(timepoints).unsqueeze(0)

            mean_trajectory = model.compute_mean_traj(timepoints, attribute_type=attribute_type).cpu().detach().numpy()

            for i in range(mean_trajectory.shape[-1]):
                ax.plot(timepoints[0, :].cpu().detach().numpy(), mean_trajectory[0, :, i], label=labels[i],
                        linewidth=4, alpha=0.9, c=colors[i])  # , c=colors[i])
            plt.legend()

        else:

            # Break if model is not initialized
            if not model[0].is_initialized:
                raise LeaspyInputError("Please initialize the model before plotting")

            # iterable
            if 'logistic' in model[0].name:
                plt.ylim(0, 1)

            timepoints = np.linspace(model[0].parameters['tau_mean'] - 3 * np.sqrt(model[0].parameters['tau_std']),
                                     model[0].parameters['tau_mean'] + 6 * np.sqrt(model[0].parameters['tau_std']),
                                     100)
            timepoints = torch.tensor(timepoints).unsqueeze(0)

            for j, el in enumerate(model):
                mean_trajectory = el.compute_mean_traj(timepoints, attribute_type=attribute_type).cpu().detach().numpy()

                for i in range(mean_trajectory.shape[-1]):
                    ax.plot(timepoints[0, :].cpu().detach().numpy(), mean_trajectory[0, :, i], label=labels[i],
                            linewidth=4, alpha=0.5, c=colors[i])  # , )

                if j == 0:
                    plt.legend()

        title = kwargs['title'] if 'title' in kwargs.keys() else None
        if title is not None:
            ax.set_title(title)

        if 'save_as' in kwargs.keys():
            plt.savefig(os.path.join(self.output_path, kwargs['save_as']))

        self.plt_show()
        plt.close()


    def plot_mean_validity(self, model, results, **kwargs):
        t0 = model.parameters['tau_mean'].numpy()
        hist = []

        for i, individual in enumerate(results.data):
            ages = individual.timepoints
            xi = results.individual_parameters['xi'][i].numpy()
            tau = results.individual_parameters['tau'][i].numpy()
            reparametrized = np.exp(xi) * (ages - tau) + t0
            hist.append(reparametrized)

        hist = [_ for l in hist for _ in l]
        plt.hist(hist)

        if 'save_as' in kwargs.keys():
            plt.savefig(os.path.join(self.output_path, kwargs['save_as']))

        self.plt_show()
        plt.close()

    def plot_patient_trajectory(self, model, results, indices, **kwargs):

        colors = kwargs['color'] if 'color' in kwargs.keys() else cm.Dark2(np.linspace(0, 1, model.dimension))
        labels = model.features
        if 'ax' in kwargs.keys():
            ax = kwargs['ax']
        else:
            (fig, ax) = plt.subplots(1, 1, figsize=(8, 8))

        if 'logistic' in model.name:
            plt.ylim(0, 1)

        if not isinstance(indices, list):
            indices = [indices]

        for idx in indices:
            indiv = results.data.get_by_idx(idx)
            timepoints = indiv.timepoints
            observations = np.array(indiv.observations)
            t = torch.tensor(timepoints).unsqueeze(0)
            indiv_parameters = results.get_patient_individual_parameters(idx)

            trajectory = model.compute_individual_tensorized(t, indiv_parameters).squeeze(0)
            for dim in range(model.dimension):
                not_nans_idx = np.array(~np.isnan(observations[:, dim]), dtype=bool)
                ax.plot(np.array(timepoints), trajectory.cpu().detach().numpy()[:, dim], c=colors[dim])
                ax.plot(np.array(timepoints)[not_nans_idx], observations[:, dim][not_nans_idx], c=colors[dim], linestyle='--')

        if 'save_as' in kwargs.keys():
            plt.savefig(os.path.join(self.output_path, kwargs['save_as']))

        if 'title' in kwargs.keys():
            plt.title(kwargs['title'])

        custom_lines = [Line2D([0], [0], color=colors[i%8], lw=4) for i in range((model.dimension))]

        ax.legend(custom_lines, labels, loc='upper right')

        if 'ax' not in kwargs.keys():
            self.plt_show()
            plt.close()

    def plot_from_individual_parameters(self, model, indiv_parameters, timepoints, **kwargs):
        colors = kwargs['color'] if 'color' in kwargs.keys() else cm.Dark2(np.linspace(0, 1, model.dimension))
        labels = model.features
        fig, ax = plt.subplots(1, 1, figsize=(11, 6))

        trajectory = model.compute_individual_trajectory(timepoints, indiv_parameters).squeeze()
        for dim in range(model.dimension):
            ax.plot(timepoints, trajectory[:, dim], c=colors[dim], label=labels[dim])

        if 'save_as' in kwargs.keys():
            plt.savefig(os.path.join(self.output_path, kwargs['save_as']))

        plt.legend()
        self.plt_show()
        plt.close()


    def plot_distribution(self, results, parameter: str, cofactor=None, **kwargs):
        fig, ax = plt.subplots(1, 1, figsize=(11, 6))

        distribution = results.get_parameter_distribution(parameter, cofactor)

        if cofactor is None:
            ax.hist(distribution)
        else:
            for k, v in distribution.items():
                ax.hist(v, label=k, alpha=0.7)
            plt.legend()

        if 'save_as' in kwargs.keys():
            plt.savefig(os.path.join(self.output_path, kwargs['save_as']))

        self.plt_show()
        plt.close()

    def plot_correlation(self, results, parameter_1, parameter_2, cofactor=None, **kwargs):
        fig, ax = plt.subplots(1, 1, figsize=(11, 6))

        d1 = results.get_parameter_distribution(parameter_1, cofactor)
        d2 = results.get_parameter_distribution(parameter_2, cofactor)

        if cofactor is None:
            ax.scatter(d1, d2)

        else:
            for possibility in d1.keys():
                ax.scatter(d1[possibility], d2[possibility], label=possibility)

        plt.legend()
        if 'save_as' in kwargs.keys():
            plt.savefig(os.path.join(self.output_path, kwargs['save_as']))

        self.plt_show()
        plt.close()

    def plot_patients_mapped_on_mean_trajectory(self, model, results):
        dataset = Dataset(results.data, model)

        patient_values = model.compute_individual_tensorized(dataset.timepoints, results.individual_parameters)
        timepoints = np.linspace(model.parameters['tau_mean'] - 2 * np.sqrt(model.parameters['tau_std']),
                                 model.parameters['tau_mean'] + 4 * np.sqrt(model.parameters['tau_std']),
                                 100)
        timepoints = torch.tensor(timepoints).unsqueeze(0)

        xi = results.individual_parameters['xi']
        tau = results.individual_parameters['tau']

        reparametrized_time = model.time_reparametrization(dataset.timepoints, xi, tau) / torch.exp(
            model.parameters['xi_mean']) + model.parameters['tau_mean']

        for i in range(dataset.values.shape[-1]):
            fig, ax = plt.subplots(1, 1)
            # ax.plot(timepoints[0,:].detach().numpy(), mean_values[0,:,i].detach().numpy(), c=colors[i])
            for idx in range(min(50, len(tau))):
                ax.plot(reparametrized_time[idx, 0:dataset.n_visits_per_individual[idx]].cpu().detach().numpy(),
                        dataset.values[idx, 0:dataset.n_visits_per_individual[idx], i].cpu().detach().numpy(), 'x', )
                ax.plot(reparametrized_time[idx, 0:dataset.n_visits_per_individual[idx]].cpu().detach().numpy(),
                        patient_values[idx, 0:dataset.n_visits_per_individual[idx], i].cpu().detach().numpy(),
                        alpha=0.8)
            if 'logistic' in model.name:
                plt.ylim(0, 1)

        self.plt_show()
        plt.close()

    ############## TODO : The next functions are related to the plots during the fit. Disentangle them properly

    @staticmethod
    def plot_error(path, dataset, model, param_ind, colors=None, labels=None):
        patient_values = model.compute_individual_tensorized(dataset.timepoints, param_ind)

        if colors is None:
            colors = cm.rainbow(np.linspace(0, 1, patient_values.shape[-1]))
        if labels is None:
            labels = np.arange(patient_values.shape[-1])
            labels = [str(k) for k in labels]

        err = {}
        err['all'] = []
        for i in range(dataset.values.shape[-1]):
            err[i] = []
            for idx in range(patient_values.shape[0]):
                err[i].extend(dataset.values[idx, 0:dataset.n_visits_per_individual[idx], i].cpu().detach().numpy() -
                              patient_values[idx, 0:dataset.n_visits_per_individual[idx], i].cpu().detach().numpy())

            err['all'].extend(err[i])
            err[i] = np.array(err[i])
        err['all'] = np.array(err['all'])
        pdf = matplotlib.backends.backend_pdf.PdfPages(path)
        for i in range(dataset.values.shape[-1]):
            fig, ax = plt.subplots(1, 1)
            # sns.distplot(err[i], color='blue')
            plt.title(labels[i] + ' sqrt mean square error: ' + str(np.sqrt(np.mean(err[i] ** 2))))
            pdf.savefig(fig)
            plt.close()
        fig, ax = plt.subplots(1, 1)
        # sns.distplot(err['all'], color='blue')
        plt.title('global sqrt mean square error: ' + str(np.sqrt(np.mean(err['all'] ** 2))))
        pdf.savefig(fig)
        plt.close()
        pdf.close()

    @staticmethod
    def plot_patient_reconstructions(path, dataset, model, param_ind, *, max_patient_number=5, attribute_type=None):

        if isinstance(max_patient_number, int):
            max_patient_number = min(max_patient_number, dataset.n_individuals)
            patients_list = range(max_patient_number)
            n_pats = max_patient_number
        else:
            # list of ints (not the ID but the indices of wanted patients [0, 1, 2, 3...])
            patients_list = max_patient_number
            n_pats = len(patients_list)

        colors = cm.Dark2(np.linspace(0, 1, n_pats + 2))

        fig, ax = plt.subplots(1, 1)

        patient_values = model.compute_individual_tensorized(dataset.timepoints, param_ind, attribute_type=attribute_type)


        for i in patients_list:
            model_value = patient_values[i, 0:dataset.n_visits_per_individual[i], :]
            score = dataset.values[i, 0:dataset.n_visits_per_individual[i], :]
            ax.plot(dataset.timepoints[i, 0:dataset.n_visits_per_individual[i]].cpu().detach().numpy(),
                    model_value.cpu().detach().numpy(), c=colors[i])
            ax.plot(dataset.timepoints[i, 0:dataset.n_visits_per_individual[i]].cpu().detach().numpy(),
                    score.cpu().detach().numpy(), c=colors[i], linestyle='--',
                    marker='o')

        # Plot the mean also
        # min_time, max_time = torch.min(dataset.timepoints[dataset.timepoints>0.0]), torch.max(dataset.timepoints)

        min_time, max_time = np.percentile(dataset.timepoints[dataset.timepoints > 0.0].cpu().detach().numpy(), [10, 90])

        timepoints = np.linspace(min_time,
                                 max_time,
                                 100)
        timepoints = torch.tensor(timepoints).unsqueeze(0)

        patient_values = model.compute_mean_traj(timepoints, attribute_type=attribute_type)
        for i in range(patient_values.shape[-1]):
            ax.plot(timepoints[0, :].cpu().detach().numpy(), patient_values[0, :, i].cpu().detach().numpy(),
                    c="black", linewidth=3, alpha=0.3)

        plt.savefig(path)
        plt.close()

        return ax

    @staticmethod
    def plot_param_ind(path, param_ind):
        # <!> param_ind is expected to be iterable of values not the usual dictionary

        pdf = matplotlib.backends.backend_pdf.PdfPages(path)
        fig, ax = plt.subplots(1, 1)
        if len(param_ind) == 2:
            # no sources
            xi, tau = param_ind
            sources = torch.zeros((0,0))
        else:
            # with sources
            xi, tau, sources = param_ind
        ax.plot(xi.squeeze(1).cpu().detach().numpy(), tau.squeeze(1).cpu().detach().numpy(), 'x')
        plt.xlabel('xi')
        plt.ylabel('tau')
        pdf.savefig(fig)
        plt.close()

        nb_sources = sources.shape[1]

        for i in range(nb_sources):
            fig, ax = plt.subplots(1, 1)
            ax.plot(sources[:, i].cpu().detach().numpy(), 'x')
            plt.title("sources " + str(i))
            pdf.savefig(fig)
            plt.close()
        pdf.close()

    ## TODO : Refaire avec le path qui est fourni en haut!
    @staticmethod
    def plot_convergence_model_parameters(path, path_saveplot_1, path_saveplot_2, model):
        # TODO? add legends (color <-> feature, esp. for g/v0/noise_std/deltas)
        # TODO? add loss (log-likelihood) or some information criteria AIC/BIC

        # figure dimensions
        width = 10
        height_per_row = 3.5

        # don't keep the sources parameters (fixed mean = 0 & std = 1 by design)
        skip_sources = True

        # Make the plot 1

        to_skip_1 = ['betas'] + ['sources_mean', 'sources_std']*int(skip_sources)
        params_to_plot_1 = [p for p in model.parameters.keys() if p not in to_skip_1]

        n_plots_1 = len(params_to_plot_1)
        n_rows_1 = math.ceil(n_plots_1 / 2)
        _, ax = plt.subplots(n_rows_1, 2, figsize=(width, n_rows_1 * height_per_row))

        for i, key in enumerate(params_to_plot_1):

            import_path = os.path.join(path, key + ".csv")
            df_convergence = pd.read_csv(import_path, index_col=0, header=None)
            df_convergence.index.rename("iter", inplace=True)

            x_position = i // 2
            y_position = i % 2
            # ax[x_position][y_position].plot(df_convergence.index.values, df_convergence.values)
            df_convergence.plot(ax=ax[x_position][y_position], legend=False)
            ax[x_position][y_position].set_title(key)

        plt.tight_layout()
        plt.savefig(path_saveplot_1)
        plt.close()

        # Make the plot 2

        reals_pop_name = model.get_population_realization_names()
        reals_ind_name = model.get_individual_realization_names()

        additional_plots = 1 # for noise_std
        if model.noise_model == 'bernoulli':
            additional_plots += 1 # for crossentropy

        if skip_sources and 'sources' in reals_ind_name:
            additional_plots -= 1

        n_plots_2 = len(reals_pop_name) + len(reals_ind_name) + additional_plots
        _, ax = plt.subplots(n_plots_2, 1, figsize=(width, n_plots_2*height_per_row))

        # nonposy is deprecated since Matplotlib 3.3
        mpl_version = mpl.__version__.split('.')
        if int(mpl_version[0]) < 3 or ((int(mpl_version[0]) == 3) and (int(mpl_version[1]) < 3)):
            yscale_kw = dict(nonposy='clip')
        else: # >= 3.3
            yscale_kw = dict(nonpositive='clip')

        # Noise std-dev
        import_path = os.path.join(path, 'noise_std.csv')
        df_convergence = pd.read_csv(import_path, index_col=0, header=None)
        df_convergence.index.rename("iter", inplace=True)
        y_position = 0
        df_convergence.plot(ax=ax[y_position], legend=False)
        ax[y_position].set_title('noise_std')
        ax[y_position].set_yscale("log", **yscale_kw)
        plt.grid(True)

        if model.noise_model == 'bernoulli':
            import_path = os.path.join(path, 'crossentropy.csv')
            df_convergence = pd.read_csv(import_path, index_col=0, header=None)
            df_convergence.index.rename("iter", inplace=True)
            y_position = 1
            df_convergence.plot(ax=ax[y_position], legend=False)
            ax[y_position].set_title('crossentropy')
            ax[y_position].set_yscale("log", **yscale_kw)
            plt.grid(True)

        for i, key in enumerate(reals_pop_name):
            y_position += 1
            ax[y_position].set_title(key)
            if key not in ['betas']:
                import_path = os.path.join(path, key + ".csv")
                df_convergence = pd.read_csv(import_path, index_col=0, header=None)
                df_convergence.index.rename("iter", inplace=True)
                df_convergence.plot(ax=ax[y_position], legend=False)
            else:
                for source_dim in range(model.source_dimension):
                    # TODO: better legend?
                    import_path = os.path.join(path, key + "_" + str(source_dim) + ".csv")
                    df_convergence = pd.read_csv(import_path, index_col=0, header=None)
                    df_convergence.index.rename("iter", inplace=True)
                    df_convergence.plot(ax=ax[y_position], legend=False)

        quartiles_factor = 0.6745 # = scipy.stats.norm.ppf(.75)

        for i, key in enumerate(reals_ind_name):

            if skip_sources and key in ['sources']:
                continue

            import_path_mean = os.path.join(path, f"{key}_mean.csv")
            df_convergence_mean = pd.read_csv(import_path_mean, index_col=0, header=None)
            df_convergence_mean.index.rename("iter", inplace=True)

            import_path_std = os.path.join(path, f"{key}_std.csv")
            df_convergence_std = pd.read_csv(import_path_std, index_col=0, header=None)
            df_convergence_std.index.rename("iter", inplace=True)

            df_convergence_mean.columns = [f"{key}_mean"]
            df_convergence_std.columns = [f"{key}_std"] # is it variance or std-dev??

            df_convergence = pd.concat([df_convergence_mean, df_convergence_std], axis=1)

            y_position += 1
            df_convergence.plot(use_index=True, y=f"{key}_mean", ax=ax[y_position], legend=False)

            mu, sd = df_convergence[f"{key}_mean"], df_convergence[f"{key}_std"]
            ax[y_position].fill_between(df_convergence.index,
                                        mu - quartiles_factor*sd, mu + quartiles_factor*sd,
                                        color='b', alpha=0.2)
            ax[y_position].set_title(key)

        plt.tight_layout()
        plt.savefig(path_saveplot_2)
        plt.close()
