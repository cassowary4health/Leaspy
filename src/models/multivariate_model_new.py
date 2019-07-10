import numpy as np

class MultivariateModelNew:
    def __init__(self):
        self.dimension = None
        self.source_dimension = None
        self.is_initialized = False
        self.parameters = {
            "p0": None,
            "mean_tau": None,
            "sigma_tau": None,
            "mean_xi": None,
            "sigma_xi": None,
            "sigma_noise": None,
            "betas": None,
            "deltas": None
        }
        self.priors = {
            'sigma_p0': 0.01,
            'sigma_delta': 0.01,
            'sigma_beta': 0.01
        }
        self.population_random_effects = {}
        self.individual_random_effects = {}
        self.attributes = {}
        #self.maximized_variables = {}

    def load_parameters(self, parameters):
        for k in self.parameters.keys():
            # TODO Check that everything is in it
            self.parameters[k] = parameters[k]


    def initialize_parameters(self, dataset, smart_initialization):
        if smart_initialization:
            if self.is_initialized:
                print("Your parameters were already initialized - they have been overwriten")
            #TODO TODO : Smart initialization
            self.dimension = dataset.dimension
            self.source_dimension = self.dimension - 1
            self.parameters = {'p0': 0.5, 'mean_tau': 70, 'sigma_tau': 2, 'mean_xi': -1., 'sigma_xi': 0.1,
                               'sigma_noise': 0.1, 'betas': [0] * (self.dimension - 1) * self.source_dimension,
                               'deltas': [0] * (self.dimension - 1)
                               }
        else:
            if self.is_initialized:
                pass
            else:
                self.dimension = dataset.dimension
                self.source_dimension = self.dimension - 1
                self.parameters = {'p0': 0.5, 'mean_tau': 70, 'sigma_tau': 2, 'mean_xi': -1., 'sigma_xi': 0.1,
                                   'sigma_noise': 0.1, 'betas': [0]*(self.dimension - 1)*self.source_dimension,
                                   'deltas': [0]*(self.dimension-1)
                                   }
        # TODO : check that the parameters have really been initialized and that they are all here!
        self.is_initialized = True

    def initialize_random_effects(self, dataset):
        self.population_random_effects['p0'] = np.random.normal(self.parameters['p0'], self.priors['sigma_p0'])
        self.population_random_effects['deltas'] = np.random.normal(self.parameters['deltas'],
                                                                    self.priors['sigma_delta'],
                                                                    size=self.dimension-1)
        self.population_random_effects['betas'] = np.random.normal(self.parameters['betas'],
                                                                   self.priors['sigma_beta'],
                                                                   size=(self.dimension-1)*self.source_dimension)

        self.individual_random_effects['xi'] = np.random.normal(self.parameters['mean_xi'],
                                                                self.parameters['sigma_xi'],
                                                                dataset.n_individuals)
        self.individual_random_effects['tau'] = np.random.normal(self.parameters['mean_tau'],
                                                                self.parameters['sigma_tau'],
                                                                dataset.n_individuals)

