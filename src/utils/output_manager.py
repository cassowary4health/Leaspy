
import numpy as np
import torch
import os
import csv
import pandas as pd
import shutil
import time
from src.utils.plotter import Plotter

import matplotlib.pyplot as plt
import matplotlib.cm as cm

class OutputManager():

    # TODO: add a loading bar for a run

    def __init__(self, path_output):



        # print every
        # plot every
        # save every
        self.print_periodicity = 50
        self.plot_periodicity = None
        self.save_periodicity = None

        # Paths
        self.path_output = path_output

        #TODO clean
        if path_output is not None:
            self.path_save_model = os.path.join(path_output, "model", "model.json")
            self.path_save_model_parameters_convergence = os.path.join(path_output, "model_parameters_convergence/")
            self.path_plot = os.path.join(path_output, "plot/")
            self.path_plot_patients = os.path.join(self.path_plot, 'patients')
            self.path_plot_convergence_model_parameters_1 = os.path.join(path_output, "plot_model_parameters_convergence_1.pdf")
            self.path_plot_convergence_model_parameters_2 = os.path.join(path_output, "plot_model_parameters_convergence_2.pdf")

            self.plot_periodicity = 50
            self.save_periodicity = 50


        # Options
        self.plot_options = {}
        self.plot_options['maximum_patient_number'] = 5
        self.plotter = Plotter()

        self.initialize()

    def initialize(self):
        self.time = time.time()

        # #TODO clean
        if self.path_output is not None:
            self.clean_output_folder()


    def iteration(self, algo, data, model, realizations):

        #print("Iteration : {0}, path_output: {1}".format(algo.iteration, self.path_output))
        iteration = algo.current_iteration

        if self.print_periodicity is not None:
            if iteration % self.print_periodicity == 0:
                self.print_algo_statistics(algo)
                self.print_model_statistics(model)
                self.print_time()

        if self.path_output is None:
            return

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

        # TODO maybe better way ???
        model_parameters_save = model_parameters.copy()

        #TODO I Stopped here, 2d array saves should be fixed.

        # Transform the types
        for key, value in model_parameters.items():
            if type(value) in [float]:
                model_parameters_save[key] = [value]
            elif type(value) in [list]:
                model_parameters_save[key] = np.array(value)
            elif value.shape == ():
                model_parameters_save[key] = [float(value)]
            # TODO, apriori only for beta
            elif type(value) in [np.ndarray]:
                # Beta
                if value.shape[0] > 1:
                    model_parameters_save.pop(key)
                    for column in range(value.shape[1]):
                        model_parameters_save["{0}_{1}".format(key, column)] = value[:, column]
                # P0, V0
                elif value.shape[0] == 1 and len(value.shape) > 1:
                    model_parameters_save[key] = value[0]

        # Save the dictionnary
        for key, value in model_parameters_save.items():
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
        self.plotter.plot_convergence_model_parameters(self.path_save_model_parameters_convergence,
                                                       self.path_plot_convergence_model_parameters_1,
                                                       self.path_plot_convergence_model_parameters_2,
                                                       model)


    def plot_model_average_trajectory(self, model):
        raise NotImplementedError

    def plot_patient_reconstructions(self, iteration, data, model, realizations):
        path_iteration = os.path.join(self.path_plot_patients,'plot_patients_{0}.pdf'.format(iteration))
        self.plotter.plot_patient_reconstructions(path_iteration, self.plot_options['maximum_patient_number'],
                                                  data, model, realizations)


    # TODO better this, difference between from model and realspop ???
    def plot_model_patient_reconstruction(self, data, model, real_ind, color="blue", ax=None):
        reals_pop = model.initialize_population_realizations()
        realizations = (reals_pop, real_ind)
        self.plotter.plot_patient_reconstruction(data, model, realizations, color=color, ax=ax)


        """
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
        plt.close()"""

