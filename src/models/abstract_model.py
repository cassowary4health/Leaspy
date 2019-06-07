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
        # TODO Use it
        self.model_parameters = {}

    ###########################
    ## Initialization
    ###########################

    def load_parameters(self, model_parameters):
        for k, v in model_parameters.items():
            if k in self.model_parameters.keys():
                # TODO problem type list as np array
                #if type(self.model_parameters[k]) in [list]:
                #   self.model_parameters[k] = np.array(self.model_parameters[k])
                previous_v = self.model_parameters[k]
                print("Replacing {} parameter from value {} to value {}".format(k, previous_v, v))
            self.model_parameters[k] = v

    def load_dimension(self, dimension):
        self.dimension = dimension
        print("Setting model dimension to : {0}".format(dimension))


    def save_parameters(self, path):
        raise NotImplementedError

        """

        #TODO check que c'est le bon format (IGOR)
        model_settings = {}

        model_settings['parameters'] = self.model_parameters
        model_settings['dimension'] = self.dimension
        model_settings['type'] = self.model_name


        dumped = json.dumps(model_settings, cls=NumpyEncoder)

        with open(path, 'w') as f:
            json.dump(dumped, f)
        """


    def initialize_realizations(self, data):
        """
        Initialize the realizations.
        All individual parameters, and population parameters that need to be considered as realizations.
        TODO : initialize settings + smart initialization
        :param data:
        :return:
        """

        print("Initialize realizations")

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

    def initialize_random_variables(self):
        print("Initialize random variables")

        self.random_variables = dict.fromkeys(self.reals_pop_name+self.reals_ind_name)

        for real_pop_name in self.reals_pop_name:
            self.random_variables[real_pop_name] = []
            for dim in range(self.dimension):
                # TODO variance arbitrary here
                self.random_variables[real_pop_name].append(GaussianRandomVariable(name=real_pop_name,
                                                                              mu=self.model_parameters[real_pop_name][dim],
                                                                              variance=0.001))

        for real_ind_name in self.reals_ind_name:
            self.random_variables[real_ind_name] = GaussianRandomVariable(name=real_ind_name,
                                                                          mu=self.model_parameters["{0}_mean".format(real_ind_name)],
                                                                          variance=self.model_parameters["{0}_var".format(real_ind_name)])


    ###########################
    ## Getters / Setters
    ###########################

    def get_parameters(self):
        return self.model_parameters

    def _update_random_variables(self):
        # TODO float for torch operations

        for real_pop_name in self.reals_pop_name:
            rv = self.random_variables[real_pop_name]
            for dim in range(len(rv)):
                self.random_variables[real_pop_name][dim].mu = float(self.model_parameters[real_pop_name].reshape(-1)[dim])

        for real_ind_name in self.reals_ind_name:
            self.random_variables[real_ind_name].mu = float(self.model_parameters["{0}_mean".format(real_ind_name)])
            self.random_variables[real_ind_name].variance = float(
                self.model_parameters["{0}_var".format(real_ind_name)])

    def __str__(self):
        output = "=== MODEL ===\n"

        for key in self.model_parameters.keys():
            output += "{0} : {1}\n".format(key, self.model_parameters[key])

        return output

    ###########################
    ## Core
    ###########################

    # Attachment
    def compute_attachment(self, data, reals_pop, reals_ind):
        return np.sum(
            [self.compute_individual_attachment(data[idx], reals_pop, reals_ind[idx]) for idx in data.indices])

    def compute_sumsquared(self, data, reals_pop, reals_ind):
        return np.sum([self.compute_individual_sumsquared(data[idx], reals_pop, reals_ind[idx]) for idx in data.indices])

    def compute_individual_sumsquared(self, individual, reals_pop, real_ind):
         return torch.sum((self.compute_individual(individual, reals_pop, real_ind)-individual.tensor_observations)**2)

    def compute_individual_attachment(self, individual, reals_pop, real_ind):
        #return self.compute_individual_sumsquared(individual, reals_pop, real_ind)*np.power(2*self.model_parameters['noise_var'], -1) + np.log(np.sqrt(2*np.pi*self.model_parameters['noise_var']))

        #TODO Remove constant terms ???
        constant_term = self.cache_variables['constant_fit_variable']
        noise_inverse = self.cache_variables['noise_inverse']

        sum_squared = self.compute_individual_sumsquared(individual, reals_pop, real_ind)

        fit = 0.5 * noise_inverse * sum_squared

        res = fit + constant_term


        return res

    # Regularity
    def compute_regularity(self, data, reals_pop, reals_ind):
        #TODO only reg on reals_ind for now
        regularity_ind = np.sum([self.compute_individual_regularity(reals_ind[idx]) for idx in data.indices])
        #regularity_pop = self.compute_individual_regularity(reals_pop)
        return regularity_ind#+regularity_pop

    def compute_individual_regularity(self, real_ind):
        return np.sum([self.compute_regularity_variable(real, key) for key, real in real_ind.items()])

    def compute_regularity_variable(self, real, key):
        return self.random_variables[key].compute_negativeloglikelihood(real)

    def compute_regularity_arrayvariable(self, real, key, dim):
        return self.random_variables[key][dim].compute_negativeloglikelihood(real)


    def simulate_individual_parameters(self):
        raise NotImplementedError

    def update_cache_variables(self):
        """
        Has to implement the inverse of noise
        :return:
        """
        raise NotImplementedError

    def _initialize_cache_variables(self):
        self.cache_variables = {}
        self.cache_variables['noise_inverse'] = 1 / self.model_parameters['noise_var']
        self.cache_variables['constant_fit_variable'] = np.log(np.sqrt(2 * np.pi * self.model_parameters['noise_var']))

    def smart_initialization(self, data):
        raise NotImplementedError