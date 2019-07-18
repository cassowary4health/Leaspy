import json
import torch
import numpy as np
import warnings
from src.utils.realizations.collection_realization import CollectionRealization
import os
from torch.autograd import Variable
from decimal import Decimal as D
import io
from src.utils.numpy_encoder import NumpyEncoder
from src.utils.random_variable.gaussian_random_variable import GaussianRandomVariable
from src.utils.random_variable.random_variable_factory import RandomVariableFactory


class AbstractModel():
    def __init__(self, name):
        self.name = name
        self.dimension = None
        self.parameters = None
        self.is_initialized = False

    def load_parameters(self, parameters):
        self.parameters = {}
        for k in parameters.keys():
            self.parameters[k] = parameters[k]

    def load_hyperparameters(self, hyperparameters):
        raise NotImplementedError

    def initialize(self, dataset):
        raise NotImplementedError

    def compute_individual(self, individual, reals_pop, real_ind):
        raise NotImplementedError

    def compute_average(self, tensor_timepoints):
        raise NotImplementedError


    def save_parameters(self, parameters):
        #raise NotImplementedError
        return 0

    def random_variable_informations(self):
        raise NotImplementedError

    def initialize_realizations(self, data):
        ### TODO : Initialize or just simulate?

        realizations = CollectionRealization(data, self)
        return realizations

    def compute_sum_squared_tensorized(self, data, realizations):
        # Compute model
        res = self.compute_individual_tensorized(data, realizations)
        # Compute the attachment
        return torch.sum((res * data.mask - data.values) ** 2, dim=(1, 2))


    def get_population_realization_names(self):
        return [name for name, value in self.random_variable_informations().items() if value['type'] == 'population']

    def get_individual_realization_names(self):
        return [name for name, value in self.random_variable_informations().items() if value['type'] == 'individual']


    def initialize_random_variables(self, data):
        print("Initialize random variables")


        random_variable_factory = RandomVariableFactory()

        # Get the info variables
        info_variables = self.random_variable_informations()
        self.random_variables = dict.fromkeys(info_variables.keys())

        for name, info in info_variables.items():
            # Create the random variable
            self.random_variables[name] = random_variable_factory.random_variable(info)

            # TODO
            # Initialize the random variable to parameters
            # /!\ We need a convention here for rv_parameters e.g. loc, var,
            # ie rv to parameters and parameters to rv, for now only text with append the parameter name
            self.random_variables[name].initialize(self.parameters)



            """
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

            """


    ###########################
    ## Getters / Setters
    ###########################

    def get_parameters(self):
        return self.parameters

    def _update_random_variables(self):
        # TODO float for torch operations

        infos_variables = self.random_variable_informations()

        reals_pop_name = [infos_variables[key]["name"] for key in infos_variables.keys() if
                          infos_variables[key]["type"] == "population"]

        reals_ind_name = [infos_variables[key]["name"] for key in infos_variables.keys() if
                          infos_variables[key]["type"] == "individual"]

        for real_pop_name in reals_pop_name:
            self.random_variables[real_pop_name].mu = self.parameters[real_pop_name]

        for real_ind_name in reals_ind_name:
            self.random_variables[real_ind_name].mu = float(self.parameters["{0}_mean".format(real_ind_name)])
            self.random_variables[real_ind_name].variance = float(
                self.parameters["{0}_var".format(real_ind_name)])

    def __str__(self):
        output = "=== MODEL ===\n"

        for key in self.parameters.keys():
            output += "{0} : {1}\n".format(key, self.parameters[key])

        return output

    ###########################
    ## Core
    ###########################

    # Attachment
    def compute_attachment(self, data, reals_pop, reals_ind):
        return torch.stack([self.compute_individual_attachment(data[idx], reals_pop, reals_ind[idx]) for idx in data.indices]).sum()

    def compute_sumsquared(self, data, reals_pop, reals_ind):
        return torch.stack([self.compute_individual_sumsquared(data[idx], reals_pop, reals_ind[idx]) for idx in data.indices]).sum()

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
        self.cache_variables['noise_inverse'] = 1 / self.parameters['noise_var']
        self.cache_variables['constant_fit_variable'] = np.log(np.sqrt(2 * np.pi * self.parameters['noise_var']))


    def update_variable_info(self, key, reals_pop):
        """
        Check according to the key, if some intermediary parameters need to be re-computed.
        :param key:
        :return:
        """
        pass

    def compute_regularity_variable(self, realization):
        # Instanciate torch distribution
        if realization.variable_type == 'population':
            distribution = torch.distributions.normal.Normal(loc=torch.Tensor([self.parameters[realization.name]]).reshape(realization.shape),
                                                            scale=self.MCMC_toolbox['priors']['{0}_std'.format(realization.name)])
        elif realization.variable_type == 'individual':
            distribution = torch.distributions.normal.Normal(loc=self.parameters["{0}_mean".format(realization.name)],
                                                            scale=self.parameters["{0}_std".format(realization.name)])
        else:
            raise ValueError("Variable type not known")


        return -distribution.log_prob(realization.tensor_realizations)


    '''
    def adapt_shapes(self):

    shapes = self.get_pop_shapes()

    for pop_var, pop_shape in shapes.items():
        print(pop_var)
        self.model_parameters[pop_var] = np.array(self.model_parameters[pop_var]).reshape(pop_shape)
    
    
    
    def initialize_realizations(self, data):
        reals_pop = self.initialize_population_realizations()
        reals_ind = self.initialize_individual_realizations(data)
        return reals_pop, reals_ind

    def initialize_individual_realizations(self, data):
        """
        Initialize the realizations.
        All individual parameters, and population parameters that need to be considered as realizations.
        TODO : initialize settings + smart initialization
        :param data:
        :return:
        """

        print("Initialize realizations")

        # TODO Change here from get_info
        #reals_pop_name = self.reals_pop_name
        #reals_ind_name = self.reals_ind_name

        infos_variables = self.random_variable_informations()
        reals_ind_name = [infos_variables[key]["name"] for key in infos_variables.keys() if
                          infos_variables[key]["type"] == "individual"]

        # Instanciate individual realizations
        reals_ind = dict.fromkeys(data.indices)

        # For all patients
        for idx in data.indices:
            # Create dictionnary of individual random variables
            reals_ind[idx] = dict.fromkeys(reals_ind_name)
            # For all invididual random variables, initialize
            for ind_name in reals_ind_name:
                reals_ind[idx][ind_name] = np.random.normal(loc=self.parameters['{0}_mean'.format(ind_name)],
                                                            scale=np.sqrt(self.parameters['{0}_var'.format(ind_name)]),
                                                            size=(1, infos_variables[ind_name]["shape"][1]))

        # To Torch
        for idx in reals_ind.keys():
            for key in reals_ind[idx]:
                reals_ind[idx][key] = torch.tensor(reals_ind[idx][key]).float()
                    #reals_ind[idx][key] = Variable(torch.tensor(reals_ind[idx][key]).float(), requires_grad=True)


        return reals_ind


    def initialize_population_realizations(self):
        infos_variables = self.random_variable_informations()

        reals_pop_name = [infos_variables[key]["name"] for key in infos_variables.keys() if
                          infos_variables[key]["type"] == "population"]

        # Population parameters
        reals_pop = dict.fromkeys(reals_pop_name)
        for pop_name in reals_pop_name:
            print(pop_name)
            reals_pop[pop_name] = np.array(self.parameters[pop_name]).reshape(infos_variables[pop_name]["shape"])

        # To Torch
        for key in reals_pop.keys():
            reals_pop[key] = torch.tensor(reals_pop[key]).float()
            #reals_pop[key] = Variable(torch.tensor(reals_pop[key]).float(), requires_grad=True)


        # initialize intermediary variables
        for key in reals_pop.keys():
            self.update_variable_info(key, reals_pop)

        return reals_pop
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
        return self.random_variables[key].compute_negativeloglikelihood(real, dim)
    
    '''