
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import os
import csv
import pandas as pd

class OutputManager():

    # TODO: add a loading bar for a run

    def __init__(self, path_output):
        # Paths
        self.path_output = path_output
        self.path_save_model = os.path.join(path_output, "model", "model.json")
        self.path_save_model_parameters_convergence = os.path.join(path_output, "model_parameters_convergence.csv")
        self.path_plot_convergence_model_parameters = os.path.join(path_output, "plot_model_parameters_convergence.pdf")
        self.path_plot_convergence_model_parametersv2 = os.path.join(path_output, "plot_model_parameters_convergencev2.pdf")

        # print every
        # plot every
        # save every
        self.print_periodicity = 100
        self.plot_periodicity = 100
        self.save_periodicity = 100

        # Options
        self.plot_options = {}
        self.plot_options['maximum_patient_number'] = 10

    def initialize(self, model):
        self.initialize_model_statistics(model)

    def iter(self, algo, data, model, realizations):

        iteration = algo.iteration

        if self.print_periodicity is not None:
            if iteration % self.print_periodicity == 0:
                self.print_algo_statistics(algo)
                self.print_model_statistics(model)

        if self.save_periodicity is not None:
            if iteration % self.save_periodicity == 0:
                self.save_model_parameters_convergence(iteration, model)
                self.save_model(model)
                #self.save_alo_statistics()

        if self.plot_periodicity is not None:
            if iteration % self.plot_periodicity == 0:
                self.plot_patient_reconstructions(iteration, data, model, realizations)
                self.plot_convergence_model_parameters(model)




    ########
    ## Printing methods
    ########

    def print_model_statistics(self, model):
        print(model)

    def print_algo_statistics(self, algo):
        print(algo)

    ########
    ## Saving methods
    ########

    def initialize_model_statistics(self, model):
        model_parameters = model.get_parameters()
        with open(self.path_save_model_parameters_convergence, 'w', newline='') as filename:
            writer = csv.writer(filename)
            writer.writerow(["Iteration"] + list(model_parameters.keys()))

    def save_model_parameters_convergence(self, iteration, model):
        model_parameters = model.get_parameters()

        with open(self.path_save_model_parameters_convergence, 'a', newline='') as filename:
            writer = csv.writer(filename)
            writer.writerow([iteration]+list(model_parameters.values()))

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

        with open(self.path_save_model_parameters_convergence, 'r') as filename:
            df_convergence = pd.read_csv(filename)

        df_convergence.set_index('Iteration', inplace=True)
        cols = df_convergence.columns.tolist()
        cols.remove('noise_var')
        df_convergence = df_convergence[["noise_var"]+cols]


        # Plot 1
        fig, ax = plt.subplots(int(len(cols)/2)+2, 2, figsize = (10,20))

        df_convergence.plot(use_index=True, y='noise_var', ax=ax[0][0], legend=False)
        ax[0][0].set_title('noise_var')

        for i, key in enumerate(cols):
            x_position = int(i/2)+1
            y_position = i % 2
            df_convergence.plot(use_index=True, y=key, ax=ax[x_position][y_position], legend=False)
            ax[x_position][y_position].set_title(key)

        plt.tight_layout()
        plt.savefig(self.path_plot_convergence_model_parameters)
        plt.close()

        # Plot 2
        reals_pop_name = model.reals_pop_name
        reals_ind_name = model.reals_ind_name


        fig, ax = plt.subplots(len(reals_pop_name+reals_ind_name)+1, 1, figsize = (10,20))

        df_convergence.plot(use_index=True, y='noise_var', ax=ax[0], legend=False)
        ax[0].set_title('noise_var')
        y_position = 0
        for i, key in enumerate(reals_pop_name):
            y_position += 1
            df_convergence.plot(use_index=True, y=key, ax=ax[y_position], legend=False)
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
        tensor_timepoints = torch.Tensor(np.linspace(data.time_min, data.time_max, 40).reshape(-1))
        model_average = model.compute_average(tensor_timepoints)
        ax.plot(tensor_timepoints.detach().numpy(), model_average.detach().numpy(), c='black', linewidth=4, alpha=0.3)

        if not os.path.exists(os.path.join(self.path_output, 'plots/')):
            os.mkdir(os.path.join(self.path_output, 'plots/'))

        plt.savefig(os.path.join(self.path_output, 'plots', 'plot_patients_{0}.pdf'.format(iteration)))
        plt.close()


"""

fig, ax = plt.subplots(6, 1, figsize=(6,12))

iters.append(iteration)
noise_var_list.append(model.model_parameters['noise_var'])

xi_mean_list.append(np.mean([x.detach().numpy() for _,x in reals_ind['xi'].items()]))
xi_std_list.append(np.std([x.detach().numpy() for _,x in reals_ind['xi'].items()]))
tau_mean_list.append(np.mean([x.detach().numpy() for _,x in reals_ind['tau'].items()]))
tau_std_list.append(np.std([x.detach().numpy() for _,x in reals_ind['tau'].items()]))
p0_list.append((reals_pop['p0'].detach().numpy()))

ax[0].plot(iters, noise_var_list)
ax[0].set_title('Noise variance')
ax[1].plot(iters, xi_mean_list)
ax[1].set_title('Xi mean, rate {0}'.format(np.mean(self.samplers_ind['xi'].acceptation_temp)))
ax[2].plot(iters, xi_std_list)
ax[2].set_title('Xi std')
ax[3].plot(iters, tau_mean_list)
ax[3].set_title('tau mean, rate {0}'.format(np.mean(self.samplers_ind['tau'].acceptation_temp)))
ax[4].plot(iters, tau_std_list)
ax[4].set_title('Tau std')
ax[5].plot(iters, p0_list)
ax[5].set_title('p0, rate {0}'.format(np.mean(self.samplers_pop['p0'].acceptation_temp)))



intercept_var_list.append(np.var([x.detach().numpy() for _,x in reals_ind['intercept'].items()]))
mu_list.append(np.mean([x.detach().numpy() for _,x in reals_ind['intercept'].items()]))

ax[0].plot(iters, noise_var_list)
ax[0].set_title('Noise')
ax[1].plot(iters, intercept_var_list)
ax[1].set_title('Intercept var list')
ax[2].plot(iters, mu_list)
ax[2].set_title("Mu list")


plt.tight_layout()
plt.savefig(os.path.join(output_path,'Convergence_parameters.pdf'))
"""

