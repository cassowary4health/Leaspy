from ....inputs.data.dataset import Dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.backends.backend_pdf
import seaborn as sns


class Plotter():

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
                #ax[x_position][y_position].plot(df_convergence.index.values, df_convergence.values)
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
                    import_path = os.path.join(path, key + "_" + str(source_dim) +".csv")
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

            df_convergence_mean.columns = [key+"_mean"]
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


    @staticmethod
    def plot_param_ind(path,model,param_ind):
        model.plot_param_ind(path,param_ind)


    @staticmethod
    def plot_patient_reconstructions(path, data, model, param_ind, max_patient_number=10, MCMC=False, ax=None):

        colors = cm.rainbow(np.linspace(0, 1, max_patient_number + 2))

        ax_provided = False
        if ax is None:
            ax_provided = True
            fig, ax = plt.subplots(1, 1)

        patient_values = model.compute_individual_tensorized(data.timepoints,param_ind,MCMC)

        if type(max_patient_number) == int:
            patients_list = range(max_patient_number)
        else:
            patients_list = max_patient_number

        for i in patients_list:
            model_value = patient_values[i,0:data.nb_observations_per_individuals[i],:]
            score = data.values[i,0:data.nb_observations_per_individuals[i],:]
            ax.plot(data.timepoints[i,0:data.nb_observations_per_individuals[i]].detach().numpy(),
                    model_value.detach().numpy(), c=colors[i])
            ax.plot(data.timepoints[i,0:data.nb_observations_per_individuals[i]].detach().numpy(),
                    score.detach().numpy(), c=colors[i], linestyle='--',
                    marker='o')

            if i > max_patient_number:
                break

        plt.savefig(path)
        plt.close()

        return ax


    @staticmethod
    def plot_mean(path,model, ax=None,colors=None,labels=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        timepoints = np.linspace(model.parameters['tau_mean'] - 2 * np.sqrt(model.parameters['tau_std']),
                                 model.parameters['tau_mean'] + 4 * np.sqrt(model.parameters['tau_std']),
                                 100)
        timepoints = torch.Tensor([timepoints])

        patient_values = model.compute_mean_traj(timepoints)
        if colors is None:
            colors = cm.rainbow(np.linspace(0, 1, patient_values.shape[-1]))
        for i in range(patient_values.shape[-1]):
            ax.plot(timepoints[0,:].detach().numpy(), patient_values[0,:,i].detach().numpy(), c=colors[i])
        plt.savefig(path)
        plt.close()
        return 0

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


