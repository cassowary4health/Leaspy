import numpy as np

from src.models.utils.attributes import Attributes

class MultivariateModelNew:
    def __init__(self):
        self.dimension = None
        self.source_dimension = None
        self.is_initialized = False
        self.parameters = {
            "g": None,
            "mean_tau": None,
            "sigma_tau": None,
            "mean_xi": None,
            "sigma_xi": None,
            "sigma_noise": None,
            "betas": None,
            "deltas": None
        }
        self.bayesian_priors = None
        self.attributes = None


        ### MCMC related "parameters"
        self.MCMC_toolbox = {
            'population_random_effects': None,
            'individual_random_effects': None,
            'attributes': None,
            'priors': {
                'sigma_g': None, # tq p0 = 1 / (1+g)
                'sigma_deltas': None, # tq deltas = user_delta_in_years * v0 / (p0(1-p0))
                'sigma_betas': None
            }
        }


    def load_parameters(self, parameters):
        for k in self.parameters.keys():
            if k != 'p0':
                # TODO Check that everything is in it
                self.parameters[k] = parameters[k]
            else:
                self.parameters['g'] = np.log(1/parameters['p0'] - 1)

    def save_parameters(self, parameters):
        #TODO
        return 0


    def initialize_parameters(self, dataset, smart_initialization):
        # TODO : have the else/if bifurcation in the mother class
        # TODO : Load the values of the parameters from a default file
        # TODO : The initialize parameters should not stay here
        if smart_initialization:
            if self.is_initialized:
                print("Your parameters were already initialized - they have been overwriten")
            #TODO TODO : Smart initialization
            self.dimension = dataset.dimension
            self.source_dimension = self.dimension - 1
            self.parameters = {'g': 0., 'mean_tau': 70, 'sigma_tau': 2, 'mean_xi': -1., 'sigma_xi': 0.1,
                               'sigma_noise': 0.1, 'deltas': [0] * (self.dimension - 1),
                               'betas': np.zeros((self.dimension - 1, self.source_dimension)).tolist()
                               }
        else:
            if self.is_initialized:
                pass
            else:
                self.dimension = dataset.dimension
                self.source_dimension = self.dimension - 1
                self.parameters = {'g': 0., 'mean_tau': 70, 'sigma_tau': 2, 'mean_xi': -1., 'sigma_xi': 0.1,
                               'sigma_noise': 0.1, 'deltas': [0] * (self.dimension - 1),
                               'betas': np.zeros((self.dimension - 1, self.source_dimension)).tolist()
                               }

        # TODO self.attributes.update(self.parameters)name_of_changed_realizations

        # TODO : check that the parameters have really been initialized and that they are all here!
        self.is_initialized = True
        self.attributes = Attributes(self.dimension, self.source_dimension)

    def initialize_MCMC_toolbox(self, dataset):

        self.MCMC_toolbox['priors'] = {
            'sigma_g': 0.01,
            'sigma_deltas': 0.01,
            'sigma_betas': 0.01
        }

        self.MCMC_toolbox['population_random_effects'] = {
            'g': np.random.normal(self.parameters['g'], self.MCMC_toolbox['priors']['sigma_g']),
            'deltas': np.random.normal(self.parameters['deltas'],
                                       self.MCMC_toolbox['priors']['sigma_deltas'],
                                       self.dimension-1),
            'betas': np.random.normal(self.parameters['betas'],
                                      self.MCMC_toolbox['priors']['sigma_betas'],
                                      (self.dimension - 1, self.source_dimension, ))
        }



        self.MCMC_toolbox['individual_random_effects'] = {
            'xi': np.random.normal(self.parameters['mean_xi'], self.parameters['sigma_xi'], dataset.n_individuals),
            'tau': np.random.normal(self.parameters['mean_tau'], self.parameters['sigma_tau'], dataset.n_individuals),
            'sources': np.random.normal(self.parameters['mean_xi'],
                                        self.parameters['sigma_xi'],
                                        (dataset.n_individuals, self.source_dimension))
        }

        self.MCMC_toolbox['attributes'] = Attributes(self.dimension, self.source_dimension)

        values = self.MCMC_toolbox['population_random_effects'].copy()
        values['mean_tau'] = self.parameters['mean_tau']
        values['mean_xi'] = self.parameters['mean_xi']
        self.MCMC_toolbox['attributes'].update(['all'], values)



    def update_MCMC_toolbox(self, new_realizations):
        """
        :param new_realizations: {('name', position) : new_scalar_value}
        :return:
        """

        # Changes the indiv and population realizations in the MCMC_toolbox
        for k, v in new_realizations.items():
            return 0

        # Updates the attributes of the MCMC_toolbox


    def compute_loglikelihood_MCMC(self, individual_parameters):
        ### TODO Compute the likelihood during the MCMC phase
        ### TODO It uses the realization of the random variables
        return 0

    def compute_loglikelihood_variational(self, individual_parameters):
        #p0 = Variable(self.parameters['p0'])
        return 0

    def compute_loglikelihood(self, individual_parameters):
        ###
        return 0