import numpy as np
import torch
from src.models.utils.attributes import Attributes
from src.utils.realizations.collection_realization import CollectionRealization

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
            "mean_sources": None,
            "sigma_sources": None,
            "sigma_noise": None,
            "betas": None,
            "deltas": None
        }
        self.bayesian_priors = None
        self.attributes = None


        ### MCMC related "parameters"
        #TODO : With Raphaël, be sure that the random effects
        #TODO : are not in the model but in the algorithm
        self.MCMC_toolbox = {
            'attributes': None,
            'priors': {
                'sigma_g': None, # tq p0 = 1 / (1+exp(g)) i.e. g = 1/p0 - 1
                'sigma_deltas': None, # tq deltas = user_delta_in_years * v0 / (p0(1-p0))
                'sigma_betas': None
            }
        }

        self.gradient_toolbox = {
            'priors': {}
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
                               'mean_sources': 0.0,
                               'sigma_sources': 1.0,
                               'sigma_noise': 0.1, 'deltas': [0] * (self.dimension - 1),
                               'mean_sources': [0]*self.source_dimension, 'sigma_sources': [1]*self.source_dimension,
                               }
        else:
            if self.is_initialized:
                pass
            else:
                self.dimension = dataset.dimension
                self.source_dimension = self.dimension - 1
                self.parameters = {'g': 0., 'mean_tau': 70, 'sigma_tau': 2, 'mean_xi': -1., 'sigma_xi': 0.1,
                               'sigma_noise': 0.1, 'deltas': [0] * (self.dimension - 1),
                                'mean_sources': 0.0,
                                   'sigma_sources': 1.0,
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
                                      (self.dimension - 1, self.source_dimension))
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

    def compute_individual_attachment_tensorized(self, data, realizations):

        print("coucou")
        a_matrix = self.MCMC_toolbox['attributes'].mixing_matrix
        wi = torch.nn.functional.linear(realizations['sources'].tensor_realizations, a_matrix, bias=None)

        reparametrized_time = torch.exp(realizations['xi'].tensor_realizations) * (data.timepoints - realizations['tau'].tensor_realizations)



        #wi = np.dot(, )

    def compute_regularity_variable(self, realization):
        return 0

    def compute_loglikelihood(self, individual_parameters):
        ###
        return 0

    def get_info_variables(self):
        # TODO : Change the name of this method

        ## Population variables
        p0_infos = {
            "name": "g",
            "shape": (1, 1),
            "type": "population",
            "rv_type": "multigaussian"
        }

        deltas_infos = {
            "name": "deltas",
            "shape": (1, self.dimension-1),
            "type": "population",
            "rv_type": "multigaussian"
        }


        betas_infos = {
            "name": "betas",
            "shape": (self.dimension - 1, self.source_dimension),
            "type": "population",
            "rv_type": "multigaussian"
        }

        ## Individual variables
        tau_infos = {
            "name": "tau",
            "shape": (1, 1),
            "type": "individual",
            "rv_type": "gaussian"
        }

        xi_infos = {
            "name": "xi",
            "shape": (1, 1),
            "type": "individual",
            "rv_type": "gaussian"
        }

        sources_infos = {
            "name": "sources",
            "shape": (1, self.source_dimension),
            "type": "individual",
            "rv_type": "gaussian"
        }

        variables_infos = {
            "p0": p0_infos,
            "deltas": deltas_infos,
            "betas": betas_infos,
            "tau": tau_infos,
            "xi": xi_infos,
            "sources": sources_infos
        }

        return variables_infos

    def initialize_realizations(self, data):
        ### TODO : Initialize or just simulate?

        realizations = CollectionRealization(data, self)
        return realizations


    def update_model(self, data, sufficient_statistics):
        raise NotImplementedError
        ### TODO : Change the parameters theta of the model AND the MCMC_toolbox
        ### TODO : according to the new sufficient statistics

