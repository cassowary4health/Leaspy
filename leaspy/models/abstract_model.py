import torch
from leaspy.utils.realizations.collection_realization import CollectionRealization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

class AbstractModel():
    def __init__(self, name):
        self.parameters = None
        self.is_initialized = False

    def load_parameters(self, parameters):
        self.parameters = {}
        for k in parameters.keys():
            self.parameters[k] = parameters[k]

    def load_hyperparameters(self, hyperparameters):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def get_individual_variable_name(self):
        """
        Return list of names of the individual variables from the model
        :return:
        """

        individual_variable_name = []

        infos = self.random_variable_informations()
        for name, info in infos.items():
            if info['type'] == 'individual':
                individual_variable_name.append(name)

        return individual_variable_name

    def compute_sum_squared_tensorized(self, data, param_ind, attribute_type=None):
        res = self.compute_individual_tensorized(data.timepoints, param_ind, attribute_type)
        res *= data.mask
        return torch.sum((res * data.mask - data.values) ** 2, dim=(1, 2))

    def compute_individual_attachment_tensorized_mcmc(self, data, realizations):
        param_ind = self.get_param_from_real(realizations)
        attachment = self.compute_individual_attachment_tensorized(data, param_ind, attribute_type='MCMC')
        return attachment

    def compute_individual_attachment_tensorized(self, data, param_ind, attribute_type):
        res = self.compute_individual_tensorized(data.timepoints, param_ind, attribute_type)
        res *= data.mask
        squared_sum = torch.sum((res * data.mask - data.values) ** 2, dim=(1, 2))
        noise_var = self.parameters['noise_std'] ** 2
        attachment = 0.5 * (1 / noise_var) * squared_sum
        attachment += np.log(np.sqrt(2 * np.pi * noise_var))
        return attachment

    def update_model_parameters(self, data, suff_stats, burn_in_phase=True):
        # Memoryless part of the algorithm
        if burn_in_phase:
            self.update_model_parameters_burn_in(data, suff_stats)
        # Stochastic sufficient statistics used to update the parameters of the model
        else:
            self.update_model_parameters_normal(data, suff_stats)
        self.attributes.update(['all'], self.parameters)


    def update_model_parameters_burn_in(self, data, realizations):
        raise NotImplementedError

    def get_population_realization_names(self):
        return [name for name, value in self.random_variable_informations().items() if value['type'] == 'population']

    def get_individual_realization_names(self):
        return [name for name, value in self.random_variable_informations().items() if value['type'] == 'individual']

    def __str__(self):
        output = "=== MODEL ===\n"
        for key in self.parameters.keys():
            output += "{0} : {1}\n".format(key, self.parameters[key])
        return output


    def compute_regularity_realization(self, realization):
        # Instanciate torch distribution
        if realization.variable_type == 'population':
            mean = self.parameters[realization.name]
            # TODO : Sure it is only MCMC_toolbox?
            std = self.MCMC_toolbox['priors']['{0}_std'.format(realization.name)]
        elif realization.variable_type == 'individual':
            mean = self.parameters["{0}_mean".format(realization.name)]
            std = self.parameters["{0}_std".format(realization.name)]
        else:
            raise ValueError("Variable type not known")

        return self.compute_regularity_variable(realization.tensor_realizations,mean,std)

    def compute_regularity_variable(self, value, mean, std):
        # Instanciate torch distribution
        distribution = torch.distributions.normal.Normal(loc=mean,scale=std)
        return -distribution.log_prob(value)

    def get_realization_object(self, n_individuals):
        ### TODO : CollectionRealizations should probably get self.get_info_var rather than all self
        realizations = CollectionRealization()
        realizations.initialize(n_individuals, self)
        return realizations

    def random_variable_informations(self):
        raise NotImplementedError