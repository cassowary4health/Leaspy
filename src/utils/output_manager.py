
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import os
import csv
import pandas as pd
import shutil
import time

class OutputManager():

    # TODO: add a loading bar for a run

    def __init__(self, path_output):
        # Paths
        self.path_output = path_output
        self.path_save_model = os.path.join(path_output, "model", "model.json")
        self.path_save_model_parameters_convergence = os.path.join(path_output, "model_parameters_convergence/")
        self.path_plot = os.path.join(path_output, "plot/")
        self.path_plot_patients = os.path.join(self.path_plot, 'patients')
        self.path_plot_convergence_model_parameters_1 = os.path.join(path_output, "plot_model_parameters_convergence_1.pdf")
        self.path_plot_convergence_model_parameters_2 = os.path.join(path_output, "plot_model_parameters_convergence_2.pdf")


        # print every
        # plot every
        # save every
        self.print_periodicity = 50
        self.plot_periodicity = 50
        self.save_periodicity = 50

        # Options
        self.plot_options = {}
        self.plot_options['maximum_patient_number'] = 10

        self.initialize()

    def initialize(self):
        self.clean_output_folder()
        self.time = time.time()

    def iter(self, algo, data, model, realizations):

        iteration = algo.iteration

        if self.print_periodicity is not None:
            if iteration % self.print_periodicity == 0:
                self.print_algo_statistics(algo)
                self.print_model_statistics(model)
                self.print_time()

        if self.save_periodicity is not None:
            if iteration % self.save_periodicity == 0:
                self.save_model_parameters_convergence(iteration, model)
                self.save_model(model)

        if self.plot_periodicity is not None:
            if iteration % self.plot_periodicity == 0:
                self.plot_patient_reconstructions(iteration, data, model, realizations)
                self.plot_convergence_model_parameters(model)




    ########
    ## Printing methods
    ########

    def print_time(self):
        current_time = time.time()
        print("Duration since last print : {0}s".format(np.round(current_time-self.time), decimals=4))
        self.time = current_time



    def print_model_statistics(self, model):
        print(model)

    def print_algo_statistics(self, algo):
        print(algo)

    ########
    ## Saving methods
    ########


    def clean_output_folder(self):
        # Remove what exists
        if os.path.exists(self.path_save_model_parameters_convergence):
            shutil.rmtree(self.path_save_model_parameters_convergence)

        if os.path.exists(self.path_plot):
            shutil.rmtree(self.path_plot)

        if os.path.exists(self.path_plot_convergence_model_parameters_1):
            os.remove(self.path_plot_convergence_model_parameters_1)

        if os.path.exists(self.path_plot_convergence_model_parameters_2):
            os.remove(self.path_plot_convergence_model_parameters_2)

        # Create if not exists
        if not os.path.exists(self.path_save_model_parameters_convergence):
            os.mkdir(self.path_save_model_parameters_convergence)

        if not os.path.exists(self.path_plot):
            os.mkdir(self.path_plot)

        if not os.path.exists(self.path_plot_patients):
            os.mkdir(self.path_plot_patients)



    def save_model_parameters_convergence(self, iteration, model):
        model_parameters = model.get_parameters()

        for key, value in model_parameters.items():
            if type(value) in [float]:
                value = [value]
            elif value.shape==():
                value = [float(value)]
            path = os.path.join(self.path_save_model_parameters_convergence, key+".csv")
            with open(path, 'a', newline='') as filename:
                writer = csv.writer(filename)
                #writer.writerow([iteration]+list(model_parameters.values()))
                writer.writerow([iteration]+list(value))

    def save_model(self, model):
        if not os.path.exists(os.path.join(self.path_output, "model")):
            os.mkdir(os.path.join(self.path_output, "model"))

        model.save_parameters(self.path_save_model)



    def save_realizations(self, realizations):
        raise NotImplementedError

    ########
    ## Plotting methods
    ########

    def plot_convergence_model_parameters(self, model):

        # Make the plot 1

        fig, ax = plt.subplots(int(len(model.model_parameters.keys()) / 2) + 1, 2, figsize=(10, 20))

        for i, key in enumerate(model.model_parameters.keys()):
            path = os.path.join(self.path_save_model_parameters_convergence, key + ".csv")
            df_convergence = pd.read_csv(path, index_col=0, header=None)
            df_convergence.index.rename("iter", inplace=True)

            x_position = int(i / 2)
            y_position = i % 2
            #ax[x_position][y_position].plot(df_convergence.index.values, df_convergence.values)
            df_convergence.plot(ax=ax[x_position][y_position], legend=False)
            ax[x_position][y_position].set_title(key)
        plt.tight_layout()
        plt.savefig(self.path_plot_convergence_model_parameters_1)
        plt.close()

        # Make the plot 2

        reals_pop_name = model.reals_pop_name
        reals_ind_name = model.reals_ind_name

        fig, ax = plt.subplots(len(reals_pop_name + reals_ind_name) + 1, 1, figsize=(10, 20))

        # Noise var
        path = os.path.join(self.path_save_model_parameters_convergence, 'noise_var' + ".csv")
        df_convergence = pd.read_csv(path, index_col=0, header=None)
        df_convergence.index.rename("iter", inplace=True)
        y_position = 0
        df_convergence.plot(ax=ax[y_position], legend=False)
        ax[y_position].set_title('noise_var')

        for i, key in enumerate(reals_pop_name):
            y_position+=1
            path = os.path.join(self.path_save_model_parameters_convergence, key + ".csv")
            df_convergence = pd.read_csv(path, index_col=0, header=None)
            df_convergence.index.rename("iter", inplace=True)
            df_convergence.plot(ax=ax[y_position], legend=False)
            ax[y_position].set_title(key)

        for i, key in enumerate(reals_ind_name):
            path_mean = os.path.join(self.path_save_model_parameters_convergence, key + "_mean.csv")
            df_convergence_mean = pd.read_csv(path_mean, index_col=0, header=None)
            df_convergence_mean.index.rename("iter", inplace=True)

            path_var = os.path.join(self.path_save_model_parameters_convergence, key + "_var.csv")
            df_convergence_var = pd.read_csv(path_var, index_col=0, header=None)
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
        plt.savefig(self.path_plot_convergence_model_parameters_2)
        plt.close()




    def plot_model_average_trajectory(self, model):
        raise NotImplementedError

    def plot_patient_reconstructions(self, iteration, data, model, realizations):

        colors = cm.rainbow(np.linspace(0, 1, self.plot_options['maximum_patient_number']+2))
        reals_pop, reals_ind = realizations

        fig, ax = plt.subplots(1, 1)

        for i, idx in enumerate(data.indices):
            model_value = model.compute_individual(data[idx], reals_pop, reals_ind[idx])
            score = data[idx].tensor_observations
            ax.plot(data[idx].tensor_timepoints.detach().numpy(), model_value.detach().numpy(), c=colors[i])
            ax.plot(data[idx].tensor_timepoints.detach().numpy(), score.detach().numpy(), c=colors[i], linestyle='--',
                    marker='o')

            if i > self.plot_options['maximum_patient_number']:
                break

        # Plot average model
        tensor_timepoints = torch.Tensor(np.linspace(data.time_min, data.time_max, 40).reshape(-1,1))
        model_average = model.compute_average(tensor_timepoints)
        ax.plot(tensor_timepoints.detach().numpy(), model_average.detach().numpy(), c='black', linewidth=4, alpha=0.3)

        plt.savefig(os.path.join(self.path_plot_patients,'plot_patients_{0}.pdf'.format(iteration)))
        plt.close()

    # with open(self.path_save_model_parameters_convergence, 'r') as filename:
    #       df_convergence = pd.read_csv(filename)

    """
    df_convergence.set_index('Iteration', inplace=True)
    cols = df_convergence.columns.tolist()
    cols.remove('noise_var')
    df_convergence = df_convergence[["noise_var"]+cols]

    # Plot 1
    fig, ax = plt.subplots(int(len(cols)/2)+2, 2, figsize=(10,20))

    df_convergence.plot(use_index=True, y='noise_var', ax=ax[0][0], legend=False)
    ax[0][0].set_title('noise_var')

    for i, key in enumerate(cols):
        x_position = int(i/2)+1
        y_position = i % 2
        ax[x_position][y_position].plot(x=df_convergence.index.values, y=df_convergence[key].values, legend=False)
        ax[x_position][y_position].set_title(key)

    plt.tight_layout()
    plt.show()
    plt.savefig(self.path_plot_convergence_model_parameters)
    plt.close()


    fig, ax = plt.subplots(1,1)
    ax.plot(x=df_convergence.index.values, y=df_convergence['p0'].values, legend=False)
    plt.show()


    # Plot 2
    reals_pop_name = model.reals_pop_name
    reals_ind_name = model.reals_ind_name


    fig, ax = plt.subplots(len(reals_pop_name+reals_ind_name)+1, 1, figsize = (10,20))

    df_convergence.plot(use_index=True, y='noise_var', ax=ax[0], legend=False)
    ax[0].set_title('noise_var')
    y_position = 0
    for i, key in enumerate(reals_pop_name):
        y_position += 1
        ax[y_position].plot(x=df_convergence.index.values, y=df_convergence[key].values, legend=False)
        ax[y_position].set_title(key)

    for i, key in enumerate(reals_ind_name):
        y_position += 1
        df_convergence.plot(use_index=True, y="{0}_mean".format(key), ax=ax[y_position], legend=False)
        ax[y_position].fill_between(df_convergence.index,
                         df_convergence["{0}_mean".format(key)]-np.sqrt(df_convergence["{0}_var".format(key)]),
                         df_convergence["{0}_mean".format(key)]+np.sqrt(df_convergence["{0}_var".format(key)]),
                         color='b', alpha=0.2)
        ax[y_position].set_title(key)

    plt.tight_layout()
    plt.savefig(self.path_plot_convergence_model_parametersv2)
    plt.close()
    """
