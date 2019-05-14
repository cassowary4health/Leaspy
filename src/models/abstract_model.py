import json
import torch
import numpy as np
import os
from torch.autograd import Variable
from decimal import Decimal as D
import io
from src.utils.numpy_encoder import NumpyEncoder
from src.utils.random_variable.gaussian_random_variable import GaussianRandomVariable


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

        dumped = json.dumps(self.model_parameters, cls=NumpyEncoder)

        with open(path, 'w') as f:
            json.dump(dumped, f)


    def get_parameters(self):
        return self.model_parameters


    def initialize_realizations(self):
        raise NotImplementedError

    def simulate_individual_parameters(self):
        raise NotImplementedError

    def __str__(self):
        output = "=== MODEL ===\n"

        for key in self.model_parameters.keys():
            output += "{0} : {1}\n".format(key, self.model_parameters[key])

        return output


    def compute_attachment(self, data, reals_pop, reals_ind):
        return np.sum(
            [self.compute_individual_attachment(data[idx], reals_pop, reals_ind[idx]) for idx in data.indices])



    def initialize_realizations(self, data):
        """
        Initialize the realizations.
        All individual parameters, and population parameters that need to be considered as realizations.
        TODO : initialize settings + smart initialization
        :param data:
        :return:
        """

        reals_pop_name = self.reals_pop_name
        reals_ind_name = self.reals_ind_name

        # Population parameters
        reals_pop = dict.fromkeys(reals_pop_name)
        for pop_name in reals_pop_name:
            reals_pop[pop_name] = self.model_parameters[pop_name]


        # Instanciate individual realizations
        reals_ind = dict.fromkeys(data.indices)

        # For all patients
        for idx in data.indices:
            # Create dictionnary of individual random variables
            reals_ind[idx] = dict.fromkeys(reals_ind_name)
            # For all invididual random variables, initialize
            for ind_name in reals_ind_name:
                reals_ind[idx][ind_name] = np.random.normal(loc=self.model_parameters['{0}_mean'.format(ind_name)],
                                                            scale=np.sqrt(self.model_parameters['{0}_var'.format(ind_name)]))

        # To Torch
        for key in reals_pop.keys():
            reals_pop[key] = Variable(torch.tensor(reals_pop[key]).float(), requires_grad=True)

        for idx in reals_ind.keys():
            for key in reals_ind[idx]:
                reals_ind[idx][key] = Variable(torch.tensor(reals_ind[idx][key]).float(), requires_grad=True)

        return reals_pop, reals_ind

    def compute_sumsquared(self, data, reals_pop, reals_ind):
        return np.sum([self.compute_individual_sumsquared(data[idx], reals_pop, reals_ind[idx]) for idx in data.indices])

    def compute_individual_sumsquared(self, individual, reals_pop, real_ind):
         return torch.sum((self.compute_individual(individual, reals_pop, real_ind)-individual.tensor_observations)**2)

    def compute_individual_attachment(self, individual, reals_pop, real_ind):
        return self.compute_individual_sumsquared(individual, reals_pop, real_ind)*np.power(2*self.model_parameters['noise_var'], -1) + np.log(np.sqrt(2*np.pi*self.model_parameters['noise_var']))



    def compute_regularity(self, data, reals_pop, reals_ind):
        #TODO only reg on reals_ind for now
        regularity_ind = np.sum([self.compute_individual_regularity(reals_ind[idx]) for idx in data.indices])
        #regularity_pop = self.compute_individual_regularity(reals_pop)
        return regularity_ind#+regularity_pop

    def compute_individual_regularity(self, real_ind):
        return np.sum([self.compute_regularity_variable(real, key) for key, real in real_ind.items()])

    def compute_regularity_variable(self, real, key):
        return self.random_variables[key].compute_negativeloglikelihood(real)

    def _update_random_variables(self):

        # TODO float for torch operations

        for real_pop_name in self.reals_pop_name:
            self.random_variables[real_pop_name].mu = float(self.model_parameters[real_pop_name])

        for real_ind_name in self.reals_ind_name:
            self.random_variables[real_ind_name].mu = float(self.model_parameters["{0}_mean".format(real_ind_name)])
            self.random_variables[real_ind_name].variance = float(self.model_parameters["{0}_var".format(real_ind_name)])

    def _initialize_random_variables(self):

        self.random_variables = dict.fromkeys(self.reals_pop_name+self.reals_ind_name)

        for real_pop_name in self.reals_pop_name:
            self.random_variables[real_pop_name] = GaussianRandomVariable(name=real_pop_name,
                                                                          mu=self.model_parameters[real_pop_name],
                                                                          variance=0.00001)

        for real_ind_name in self.reals_ind_name:
            self.random_variables[real_ind_name] = GaussianRandomVariable(name=real_ind_name,
                                                                          mu=self.model_parameters["{0}_mean".format(real_ind_name)],
                                                                          variance=self.model_parameters["{0}_var".format(real_ind_name)])


