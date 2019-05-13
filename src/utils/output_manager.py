
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import os


class OutputManager():

    # TODO: add a loading bar for a run

    def __init__(self, path_output):
        self.path_output = path_output
        # print every
        # plot every
        # save every

        self.print_periodicity = 100
        self.plot_periodicity = 100
        self.save_periodicity = None

        # Options
        self.plot_options = {}
        self.plot_options['maximum_patient_number'] = 10

    def iter(self, algo, data, model, realizations):

        iteration = algo.iteration

        if self.print_periodicity is not None:
            if iteration % self.print_periodicity == 0:
                self.print_algo_statistics(algo)
                self.print_model_statistics(model)

        if self.save_periodicity is not None:
            if iteration % self.save_periodicity == 0:
                self.save_model_statistics()
                self.save_alo_statistics()

        if self.plot_periodicity is not None:
            if iteration % self.plot_periodicity == 0:
                self.plot_patient_reconstructions(iteration, data, model, realizations)




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

    def save_model_statistics(self, model):
        raise NotImplementedError

    def save_realizations(self, reals_pop, reals_ind):
        raise NotImplementedError

    ########
    ## Plotting methods
    ########

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

