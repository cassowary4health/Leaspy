
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.backends.backend_pdf
import seaborn as sns

from leaspy.inputs.data.dataset import Dataset


class Plotter:

    def __init__(self, output_path=None):
        # TODO : Do all the check up if the path exists, and if yes, if removing or not
        if output_path is None:
            output_path = os.getcwd()
        self.output_path = output_path

    def plot_mean_trajectory(self, model, **kwargs):
        #colors = kwargs['color'] if 'color' in kwargs.keys() else cm.gist_rainbow(np.linspace(0, 1, model.dimension))
        labels = kwargs['labels'] if 'labels' in kwargs.keys() else ['label_'+str(i) for i in range(model.dimension)]
        fig, ax = plt.subplots(1, 1, figsize=(11, 6))
        plt.ylim(0, 1)

        timepoints = np.linspace(model.parameters['tau_mean'] - 3 * np.sqrt(model.parameters['tau_std']),
                                 model.parameters['tau_mean'] + 6 * np.sqrt(model.parameters['tau_std']),
                                 100)
        timepoints = torch.Tensor([timepoints])
        mean_trajectory = model.compute_mean_traj(timepoints).detach().numpy()


        for i in range(mean_trajectory.shape[-1]):
            ax.plot(timepoints[0, :].detach().numpy(), mean_trajectory[0, :, i], label=labels[i], linewidth=4)#, c=colors[i])
        plt.legend()


        if 'save_as' in kwargs.keys():
            plt.savefig(os.path.join(self.output_path, kwargs['save_as']))

        plt.show()

    def plot_patient_trajectory(self, model, results, indices, **kwargs):

        colors = kwargs['color'] if 'color' in kwargs.keys() else cm.gist_rainbow(np.linspace(0, 1, model.dimension))
        labels = kwargs['labels'] if 'labels' in kwargs.keys() else ['label_'+str(i) for i in range(model.dimension)]
        fig, ax = plt.subplots(1, 1, figsize=(11, 6))
        plt.ylim(0, 1)

        if type(indices) is not list:
            indices = [indices]

        for idx in indices:
            indiv = results.data.get_by_idx(idx)
            timepoints = indiv.timepoints
            observations = np.array(indiv.observations)
            t = torch.Tensor(timepoints).unsqueeze(0)
            p = results.individual_parameters[idx]
            p = (torch.Tensor([p['xi']]).unsqueeze(0),
                 torch.Tensor([p['tau']]).unsqueeze(0),
                 torch.Tensor(p['sources']).unsqueeze(0))

            trajectory = model.compute_individual_tensorized(t, p).squeeze(0)
            for dim in range(model.dimension):
                ax.plot(timepoints, trajectory.detach().numpy()[:, dim], c=colors[dim])
                ax.plot(timepoints, observations[:, dim], c=colors[dim])
        plt.show()

        if 'save_as' in kwargs.keys():
            plt.savefig(os.path.join(self.output_path, kwargs['save_as']))
            plt.close()

    def plot_distribution(self, results, parameter, cofactor=None, **kwargs):
        fig, ax = plt.subplots(1, 1, figsize=(11, 6))
        distribution = results.get_parameter_distribution(parameter, cofactor)

        if cofactor is None:
            ax.hist(distribution)
        else:

            for k, v in distribution.items():
                ax.hist(v, label=k, alpha=0.7)
            plt.legend()
        plt.show()

        if 'save_as' in kwargs.keys():
            plt.savefig(os.path.join(self.output_path, kwargs['save_as']))
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
        plt.show()

        if 'save_as' in kwargs.keys():
            plt.savefig(os.path.join(self.output_path, kwargs['save_as']))
            plt.close()

    def plot_patients_mapped_on_mean_trajectory(self, model, results):
        dataset = Dataset(results.data, model)
        individual_parameters = []
        xi = torch.Tensor([results.individual_parameters[_.idx]['xi'] for _ in results.data]).unsqueeze(1)
        tau = torch.Tensor([results.individual_parameters[_.idx]['tau'] for _ in results.data]).unsqueeze(1)
        sources = torch.Tensor([results.individual_parameters[_.idx]['sources'] for _ in results.data])

        individual_parameters = (xi, tau, sources)
        patient_values = model.compute_individual_tensorized(dataset.timepoints, individual_parameters)
        timepoints = np.linspace(model.parameters['tau_mean'] - 2 * np.sqrt(model.parameters['tau_std']),
                                 model.parameters['tau_mean'] + 4 * np.sqrt(model.parameters['tau_std']),
                                 100)
        timepoints = torch.Tensor([timepoints])

        reparametrized_time = model.time_reparametrization(dataset.timepoints, xi, tau) / torch.exp(
            model.parameters['xi_mean']) + model.parameters['tau_mean']

        for i in range(dataset.values.shape[-1]):
            fig, ax = plt.subplots(1, 1)
            #ax.plot(timepoints[0,:].detach().numpy(), mean_values[0,:,i].detach().numpy(), c=colors[i])
            for idx in range(50):
                ax.plot(reparametrized_time[idx, 0:dataset.nb_observations_per_individuals[idx]].detach().numpy(),
                        dataset.values[idx,0:dataset.nb_observations_per_individuals[idx],i].detach().numpy(),'x', )
                ax.plot(reparametrized_time[idx, 0:dataset.nb_observations_per_individuals[idx]].detach().numpy(),
                        patient_values[idx,0:dataset.nb_observations_per_individuals[idx],i].detach().numpy(), alpha=0.8)
            plt.ylim(0, 1)

    ### TODO TODO : Check what the following function is

    @staticmethod
    def plot_within_mean_traj(path, dataset, model, param_ind, max_patient_number=25,colors =None,labels=None):
        xi,tau = model.get_xi_tau(param_ind)

        patient_values = model.compute_individual_tensorized(dataset.timepoints, param_ind, MCMC=False)
        timepoints = np.linspace(model.parameters['tau_mean'] - 2 * np.sqrt(model.parameters['tau_std']),
                                 model.parameters['tau_mean'] + 4 * np.sqrt(model.parameters['tau_std']),
                                 100)
        timepoints = torch.Tensor([timepoints])
        mean_values = model.compute_mean_traj(timepoints)
        if colors is None:
            colors = cm.rainbow(np.linspace(0, 1, patient_values.shape[-1]))
        if labels is None:
            labels = np.arange(patient_values.shape[-1])
            labels = [str(k) for k in labels]

        reparametrized_time = model.time_reparametrization(dataset.timepoints,xi,tau)/torch.exp(model.parameters['xi_mean'])+model.parameters['tau_mean']

        pdf = matplotlib.backends.backend_pdf.PdfPages(path)
        for i in range(dataset.values.shape[-1]):
            fig, ax = plt.subplots(1, 1)
            ax.plot(timepoints[0,:].detach().numpy(), mean_values[0,:,i].detach().numpy(), c=colors[i])
            for idx in range(max_patient_number):
                ax.plot(reparametrized_time[idx, 0:dataset.nb_observations_per_individuals[idx]].detach().numpy(),
                        dataset.values[idx,0:dataset.nb_observations_per_individuals[idx],i].detach().numpy(),'x', c=colors[i])
                ax.plot(reparametrized_time[idx, 0:dataset.nb_observations_per_individuals[idx]].detach().numpy(),
                        patient_values[idx,0:dataset.nb_observations_per_individuals[idx],i].detach().numpy(), c=colors[i],alpha=0.3)
            plt.title(labels[i])
            pdf.savefig(fig)
        pdf.close()
        return 0


    ############## TODO : The next functions are related to the plots during the fit. Disentangle them properly

    @staticmethod
    def plot_error(path, dataset, model, param_ind,colors =None,labels=None):
        patient_values = model.compute_individual_tensorized(dataset.timepoints, param_ind, MCMC=False)

        if colors is None:
            colors = cm.rainbow(np.linspace(0, 1, patient_values.shape[-1]))
        if labels is None:
            labels = np.arange(patient_values.shape[-1])
            labels = [str(k) for k in labels]

        err = {}
        err['all'] = []
        for i in range(dataset.values.shape[-1]):
            err[i]=[]
            for idx in range(patient_values.shape[0]):
                err[i].extend(dataset.values[idx,0:dataset.nb_observations_per_individuals[idx],i].detach().numpy()-
                              patient_values[idx,0:dataset.nb_observations_per_individuals[idx],i].detach().numpy())
            err['all'].extend(err[i])
            err[i] = np.array(err[i])
        err['all'] = np.array(err['all'])
        pdf = matplotlib.backends.backend_pdf.PdfPages(path)
        for i in range(dataset.values.shape[-1]):
            fig, ax = plt.subplots(1, 1)
            sns.distplot(err[i], color='blue')
            plt.title(labels[i]+' sqrt mean square error: '+str(np.sqrt(np.mean(err[i] ** 2))))
            pdf.savefig(fig)
            plt.close()
        fig, ax = plt.subplots(1, 1)
        sns.distplot(err['all'], color='blue')
        plt.title('global sqrt mean square error: ' + str(np.sqrt(np.mean(err['all'] ** 2))))
        pdf.savefig(fig)
        plt.close()
        pdf.close()
        return 0

    @staticmethod
    def plot_patient_reconstructions(path, data, model, param_ind, max_patient_number=10, MCMC=False):

        colors = cm.rainbow(np.linspace(0, 1, max_patient_number + 2))

        fig, ax = plt.subplots(1, 1)

        patient_values = model.compute_individual_tensorized(data.timepoints, param_ind, MCMC)

        if type(max_patient_number) == int:
            patients_list = range(max_patient_number)
        else:
            patients_list = max_patient_number

        for i in patients_list:
            model_value = patient_values[i, 0:data.nb_observations_per_individuals[i], :]
            score = data.values[i, 0:data.nb_observations_per_individuals[i], :]
            ax.plot(data.timepoints[i, 0:data.nb_observations_per_individuals[i]].detach().numpy(),
                    model_value.detach().numpy(), c=colors[i])
            ax.plot(data.timepoints[i, 0:data.nb_observations_per_individuals[i]].detach().numpy(),
                    score.detach().numpy(), c=colors[i], linestyle='--',
                    marker='o')

            if i > max_patient_number:
                break

        plt.savefig(path)
        plt.close()

        return ax

    @staticmethod
    def plot_param_ind(path, param_ind):

        pdf = matplotlib.backends.backend_pdf.PdfPages(path)
        fig, ax = plt.subplots(1, 1)
        xi, tau, sources = param_ind
        ax.plot(xi.squeeze(1).detach().numpy(), tau.squeeze(1).detach().numpy(), 'x')
        plt.xlabel('xi')
        plt.ylabel('tau')
        pdf.savefig(fig)
        plt.close()

        nb_sources = sources.shape[1]

        for i in range(nb_sources):
            fig, ax = plt.subplots(1, 1)
            ax.plot(sources[:, i].detach().numpy(), 'x')
            plt.title("sources " + str(i))
            pdf.savefig(fig)
            plt.close()
        pdf.close()


    ## TODO : Refaire avec le path qui est fourni en haut!
    @staticmethod
    def plot_convergence_model_parameters(path, path_saveplot_1, path_saveplot_2, model):

        # Make the plot 1

        fig, ax = plt.subplots(int(len(model.parameters.keys()) / 2) + 1, 2, figsize=(10, 20))

        for i, key in enumerate(model.parameters.keys()):

            if key not in ['betas']:
                import_path = os.path.join(path, key + ".csv")
                df_convergence = pd.read_csv(import_path, index_col=0, header=None)
                df_convergence.index.rename("iter", inplace=True)

                x_position = int(i / 2)
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

        fig, ax = plt.subplots(len(reals_pop_name + reals_ind_name) + 1, 1, figsize=(10, 20))

        # Noise var
        import_path = os.path.join(path, 'noise_std' + ".csv")
        df_convergence = pd.read_csv(import_path, index_col=0, header=None)
        df_convergence.index.rename("iter", inplace=True)
        y_position = 0
        df_convergence.plot(ax=ax[y_position], legend=False)
        ax[y_position].set_title('noise_std')
        ax[y_position].set_yscale("log", nonposy='clip')
        plt.grid(True)

        for i, key in enumerate(reals_pop_name):
            y_position += 1
            if key not in ['betas']:
                import_path = os.path.join(path, key + ".csv")
                df_convergence = pd.read_csv(import_path, index_col=0, header=None)
                df_convergence.index.rename("iter", inplace=True)
                df_convergence.plot(ax=ax[y_position], legend=False)
                ax[y_position].set_title(key)
            if key in ['betas']:
                for source_dim in range(model.source_dimension):
                    import_path = os.path.join(path, key + "_" + str(source_dim) + ".csv")
                    df_convergence = pd.read_csv(import_path, index_col=0, header=None)
                    df_convergence.index.rename("iter", inplace=True)
                    df_convergence.plot(ax=ax[y_position], legend=False)
                    ax[y_position].set_title(key)

        for i, key in enumerate(reals_ind_name):
            import_path_mean = os.path.join(path, "{}_mean.csv".format(key))
            df_convergence_mean = pd.read_csv(import_path_mean, index_col=0, header=None)
            df_convergence_mean.index.rename("iter", inplace=True)

            import_path_var = os.path.join(path, "{}_std.csv".format(key))
            df_convergence_var = pd.read_csv(import_path_var, index_col=0, header=None)
            df_convergence_var.index.rename("iter", inplace=True)

            df_convergence_mean.columns = [key + "_mean"]
            df_convergence_var.columns = [key + "_sigma"]

            df_convergence = pd.concat([df_convergence_mean, df_convergence_var], axis=1)

            y_position += 1
            df_convergence.plot(use_index=True, y="{0}_mean".format(key), ax=ax[y_position], legend=False)
            ax[y_position].fill_between(df_convergence.index,
                                        df_convergence["{0}_mean".format(key)] - np.sqrt(
                                            df_convergence["{0}_sigma".format(key)]),
                                        df_convergence["{0}_mean".format(key)] + np.sqrt(
                                            df_convergence["{0}_sigma".format(key)]),
                                        color='b', alpha=0.2)
            ax[y_position].set_title(key)

        plt.tight_layout()
        plt.savefig(path_saveplot_2)
        plt.close()




