import json
import torch
import numpy as np
import os

class AbstractModel():
    def __init__(self):
        self.model_parameters = {}

    def load_parameters(self, model_parameters):
        for k, v in model_parameters.items():
            if k in self.model_parameters.keys():
                previous_v = self.model_parameters[k]
                print("Replacing {} parameter from value {} to value {}".format(k, previous_v, v))
            self.model_parameters[k] = v

    def save_parameters(self, path):
        with open(path, 'w') as outfile:
            json.dump(self.model_parameters, outfile)

    def initialize_realizations(self):
        raise NotImplementedError

    def simulate_individual_parameters(self):
        raise NotImplementedError

    def __str__(self):
        output = "         Model Summary         \n"

        for key in self.model_parameters.keys():
            output += "{0} : {1}\n".format(key, self.model_parameters[key])

        return output

    def plot(self, data, iter, realizations, path_output):

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        colors = cm.rainbow(np.linspace(0, 1, 12))

        reals_pop, reals_ind = realizations

        fig, ax = plt.subplots(1,1)


        for i, idx in enumerate(data.indices):
            model_value = self.compute_individual(data[idx], reals_pop, reals_ind[idx])
            score = data[idx].tensor_observations

            ax.plot(data[idx].tensor_timepoints.detach().numpy(), model_value.detach().numpy(), c=colors[i])
            ax.plot(data[idx].tensor_timepoints.detach().numpy(), score.detach().numpy(), c=colors[i], linestyle='--', marker='o')

            if i>10:
                break
        # Plot average model
        tensor_timepoints = torch.Tensor(np.linspace(data.time_min,data.time_max,40).reshape(-1))
        model_average = self.compute_average(tensor_timepoints)
        ax.plot(tensor_timepoints.detach().numpy(), model_average.detach().numpy(), c='black', linewidth = 4, alpha = 0.3)


        if not os.path.exists(os.path.join(path_output, 'plots/')):
            os.mkdir(os.path.join(path_output, 'plots/'))

        plt.savefig(os.path.join(path_output, 'plots', 'plot_patients_{0}.pdf'.format(iter)))
        plt.close()

    def compute_attachment(self, data, reals_pop, reals_ind):
        return np.sum(
            [self.compute_individual_attachment(data[idx], reals_pop, reals_ind[idx]) for idx in data.indices])

    def compute_regularity(self, data, reals_pop, reals_ind):
        return np.sum([self.compute_individual_regularity(reals_ind[idx]) for idx in data.indices])



