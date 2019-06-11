
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pandas as pd
import numpy as np
import torch

class Plotter():

    @staticmethod
    def plot_convergence_model_parameters(path, path_saveplot_1, path_saveplot_2, model):

        # Make the plot 1

        fig, ax = plt.subplots(int(len(model.model_parameters.keys()) / 2) + 1, 2, figsize=(10, 20))

        for i, key in enumerate(model.model_parameters.keys()):
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

        reals_pop_name = model.reals_pop_name
        reals_ind_name = model.reals_ind_name

        fig, ax = plt.subplots(len(reals_pop_name + reals_ind_name) + 1, 1, figsize=(10, 20))

        # Noise var
        import_path = os.path.join(path, 'noise_var' + ".csv")
        df_convergence = pd.read_csv(import_path, index_col=0, header=None)
        df_convergence.index.rename("iter", inplace=True)
        y_position = 0
        df_convergence.plot(ax=ax[y_position], legend=False)
        ax[y_position].set_title('noise_var')
        ax[y_position].set_yscale("log", nonposy='clip')
        plt.grid(True)

        for i, key in enumerate(reals_pop_name):
            y_position+=1
            import_path = os.path.join(path, key + ".csv")
            df_convergence = pd.read_csv(import_path, index_col=0, header=None)
            df_convergence.index.rename("iter", inplace=True)
            df_convergence.plot(ax=ax[y_position], legend=False)
            ax[y_position].set_title(key)

        for i, key in enumerate(reals_ind_name):
            import_path_mean = os.path.join(path, key + "_mean.csv")
            df_convergence_mean = pd.read_csv(import_path_mean, index_col=0, header=None)
            df_convergence_mean.index.rename("iter", inplace=True)

            import_path_var = os.path.join(path, key + "_var.csv")
            df_convergence_var = pd.read_csv(import_path_var, index_col=0, header=None)
            df_convergence_var.index.rename("iter", inplace=True)

            df_convergence_mean.columns = [key+"_mean"]
            df_convergence_var.columns = [key + "_var"]

            df_convergence = pd.concat([df_convergence_mean, df_convergence_var], axis=1)

            y_position += 1
            df_convergence.plot(use_index=True, y="{0}_mean".format(key), ax=ax[y_position], legend=False)
            ax[y_position].fill_between(df_convergence.index,
                                        df_convergence["{0}_mean".format(key)] - np.sqrt(
                                            df_convergence["{0}_var".format(key)]),
                                        df_convergence["{0}_mean".format(key)] + np.sqrt(
                                            df_convergence["{0}_var".format(key)]),
                                        color='b', alpha=0.2)
            ax[y_position].set_title(key)

        plt.tight_layout()
        plt.savefig(path_saveplot_2)
        plt.close()





    @staticmethod
    def plot_patient_reconstructions(path, maximum_patient_number, data, model, realizations):

        colors = cm.rainbow(np.linspace(0, 1, maximum_patient_number+2))
        reals_pop, reals_ind = realizations

        fig, ax = plt.subplots(1, 1)

        for i, idx in enumerate(data.indices):
            model_value = model.compute_individual(data[idx], reals_pop, reals_ind[idx])
            score = data[idx].tensor_observations
            ax.plot(data[idx].tensor_timepoints.detach().numpy(), model_value.detach().numpy(), c=colors[i])
            ax.plot(data[idx].tensor_timepoints.detach().numpy(), score.detach().numpy(), c=colors[i], linestyle='--',
                    marker='o')

            if i > maximum_patient_number:
                break

        # Plot average model
        tensor_timepoints = torch.Tensor(np.linspace(data.time_min, data.time_max, 40).reshape(-1,1))
        model_average = model.compute_average(tensor_timepoints)
        ax.plot(tensor_timepoints.detach().numpy(), model_average.detach().numpy(), c='black', linewidth=4, alpha=0.3)

        plt.savefig(path)
        plt.close()
