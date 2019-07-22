import numpy as np
import torch
import json

from src.models.abstract_model import AbstractModel
from src.models.utils.attributes.attributes_multivariateparallel import Attributes_MultivariateParallel


class MultivariateModelParallel(AbstractModel):
    def __init__(self, name):
        super(MultivariateModelParallel, self).__init__(name)
        self.source_dimension = None
        self.parameters = {
            "g": None, "betas": None, "deltas": None,
            "tau_mean": None, "tau_std": None,
            "xi_mean": None,  "xi_std": None,
            "sources_mean": None, "sources_std": None,
            "noise_std": None
        }
        self.bayesian_priors = None
        self.attributes = None

        # MCMC related "parameters"
        self.MCMC_toolbox = {
            'attributes': None,
            'priors': {
                'g_std': None, # tq p0 = 1 / (1+exp(g)) i.e. g = 1/p0 - 1
                'deltas_std': None, # tq deltas = user_delta_in_years * v0 / (p0(1-p0))
                'betas_std': None
            }
        }

    def load_hyperparameters(self, hyperparameters):
        self.dimension = hyperparameters['dimension']
        self.source_dimension = hyperparameters['source_dimension']

    def save(self, path):
        model_settings = {
            'name': 'multivariate_parallel',
            'dimension': self.dimension,
            'source_dimension': self.source_dimension,
            'parameters': self.parameters
        }

        with open(path, 'w') as fp:
            json.dump(model_settings, fp)

    def initialize(self, data):
        self.dimension = data.dimension
        self.source_dimension = int(data.dimension/2.)  # TODO : How to change it independently of the initialize?

        # "Smart" initialization : may be improved
        # TODO !
        self.parameters = {
            'g': 1, 'tau_mean': 70.0, 'tau_std': 2.0, 'xi_mean': -3., 'xi_std': 0.1,
            'sources_mean': 0.0, 'sources_std': 1.0,
            'noise_std': 0.1, 'deltas': [-3., -2., -2.],
            'betas': np.zeros((self.dimension - 1, self.source_dimension)).tolist()
        }
        self.attributes = Attributes_MultivariateParallel(self.dimension, self.source_dimension)
        self.is_initialized = True

    def initialize_MCMC_toolbox(self, data):
        self.MCMC_toolbox = {
            'priors': {'g_std': 1., 'deltas_std': 0.1, 'betas_std': 0.1 },
            'attributes': Attributes_MultivariateParallel(self.dimension, self.source_dimension)
        }

        realizations = self.get_realization_object(data)
        self.update_MCMC_toolbox(['all'], realizations)

    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        L = name_of_the_variables_that_have_been_changed
        values = {}
        if any(c in L for c in ('g', 'all')):
            values['g'] = realizations['g'].tensor_realizations
        if any(c in L for c in ('deltas', 'all')):
            values['deltas'] = realizations['deltas'].tensor_realizations
        if any(c in L for c in ('betas', 'all')):
            values['betas'] = realizations['betas'].tensor_realizations
        if any(c in L for c in ('xi_mean', 'all')):
            values['xi_mean'] = self.parameters['xi_mean']

        self.MCMC_toolbox['attributes'].update(L, values)

    def compute_individual_tensorized(self, data, realizations):
        # Population parameters
        g = self.MCMC_toolbox['attributes'].g
        deltas = self.MCMC_toolbox['attributes'].deltas
        deltas_exp = torch.exp(-deltas)
        a_matrix = self.MCMC_toolbox['attributes'].mixing_matrix

        # Individual parameters
        xi = realizations['xi'].tensor_realizations
        tau = realizations['tau'].tensor_realizations
        wi = torch.nn.functional.linear(realizations['sources'].tensor_realizations, a_matrix, bias=None)
        timepoints = data.timepoints.reshape(data.timepoints.shape[0], data.timepoints.shape[1], 1)
        reparametrized_time = torch.exp(xi) * (timepoints - tau)

        # Log likelihood computation
        LL = wi * (g * deltas_exp + 1) ** 2 / (g * deltas_exp)
        LL = -reparametrized_time - deltas - LL
        model = 1. / (1. + g*torch.exp(LL))

        return model * data.mask

    def compute_individual_attachment_tensorized(self, data, realizations):
        squared_sum = self.compute_sum_squared_tensorized(data, realizations)
        noise_var = self.parameters['noise_std']**2
        attachment = 0.5 * (1/noise_var) * squared_sum
        attachment += np.log(np.sqrt(2 * np.pi * noise_var))

        return attachment

    def compute_sufficient_statistics(self, data, realizations):
        sufficient_statistics = {}
        sufficient_statistics['g'] = realizations['g'].tensor_realizations.detach()
        sufficient_statistics['deltas'] = realizations['deltas'].tensor_realizations.detach()
        sufficient_statistics['betas'] = realizations['betas'].tensor_realizations.detach()
        sufficient_statistics['tau'] = realizations['tau'].tensor_realizations
        sufficient_statistics['tau_sqrd'] = torch.pow(realizations['tau'].tensor_realizations, 2)
        sufficient_statistics['xi'] = realizations['xi'].tensor_realizations
        sufficient_statistics['xi_sqrd'] = torch.pow(realizations['xi'].tensor_realizations, 2)

        #TODO : Optimize to compute the matrix multiplication only once for the reconstruction
        data_reconstruction = self.compute_individual_tensorized(data, realizations)
        norm_0 = data.values * data.values * data.mask
        norm_1 = data.values * data_reconstruction * data.mask
        norm_2 = data_reconstruction * data_reconstruction * data.mask
        sufficient_statistics['obs_x_obs'] = torch.sum(norm_0, dim=2)
        sufficient_statistics['obs_x_reconstruction'] = torch.sum(norm_1, dim=2)
        sufficient_statistics['reconstruction_x_reconstruction'] = torch.sum(norm_2, dim=2)

        return sufficient_statistics

    def update_model_parameters(self, data, suff_stats, burn_in_phase=True):
        # Memoryless part of the algorithm
        if burn_in_phase:
            self.parameters['g'] = suff_stats['g'].tensor_realizations.detach().tolist()[0]
            self.parameters['deltas'] = suff_stats['deltas'].tensor_realizations.detach().numpy()
            self.parameters['betas'] = suff_stats['betas'].tensor_realizations.detach().numpy()
            xi = suff_stats['xi'].tensor_realizations.detach()
            self.parameters['xi_mean'] = torch.mean(xi).tolist()
            self.parameters['xi_std'] = torch.std(xi).tolist()
            tau = suff_stats['tau'].tensor_realizations.detach()
            self.parameters['tau_mean'] = torch.mean(tau).tolist()
            self.parameters['tau_std'] = torch.std(tau).tolist()

            data_fit = self.compute_individual_tensorized(data, suff_stats)
            squared_diff = ((data_fit-data.values)**2).sum()
            self.parameters['noise_std'] = np.sqrt(squared_diff/(data.n_visits*data.dimension))

        # Stochastic sufficient statistics used to update the parameters of the model
        else:
            self.parameters['g'] = suff_stats['g'].tolist()[0][0]
            self.parameters['deltas'] = suff_stats['deltas'].tolist()[0]
            self.parameters['betas'] = suff_stats['betas'].tolist()

            tau_mean = self.parameters['tau_mean']
            tau_std_updt = tau_mean * tau_mean - tau_mean * torch.sum(suff_stats['tau'])
            self.parameters['tau_std'] = ((tau_std_updt + torch.sum(suff_stats['tau_sqrd']))/data.n_individuals).tolist()
            self.parameters['tau_mean'] = torch.mean(suff_stats['tau']).tolist()

            xi_mean = self.parameters['xi_mean']
            tau_std_updt = xi_mean * xi_mean - xi_mean * torch.sum(suff_stats['xi'])
            self.parameters['xi_std'] = ((tau_std_updt + torch.sum(suff_stats['xi_sqrd']))/data.n_individuals).tolist()
            self.parameters['xi_mean'] = torch.mean(suff_stats['xi']).tolist()

            S1 = torch.sum(suff_stats['obs_x_obs'])
            S2 = torch.sum(suff_stats['obs_x_reconstruction'])
            S3 = torch.sum(suff_stats['reconstruction_x_reconstruction'])

            self.parameters['noise_std'] = torch.sqrt((S1 - 2.*S2 + S3)/(data.dimension * data.n_visits)).tolist()


    def random_variable_informations(self):
        ## Population variables
        g_infos = {
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
            "g": g_infos,
            "deltas": deltas_infos,
            "betas": betas_infos,
            "tau": tau_infos,
            "xi": xi_infos,
            "sources": sources_infos
        }

        return variables_infos

    def compute_parallel_curve(self, age, xi, tau, sources, attributes=False):
        ### TODO IMPORTANT : There is a need - for the other important function such as leaspy.predict or leaspy.simulate
        ### TODO : to have a fit(age, xi, tau, sources) function that internally uses the self.attributes parameters
        ### TODO : as this need to be a UNIQUE function to do that everywhere
        ### TODO HINT : there is probably a need for all the "attributes" that are in the realizations, parameters, RV, ...
        ### TODO HINT : to be stored in the attributes object. This way, the function fit can have a default attribute value
        ### TODO HINT : at False that uses the internal attibutes. Otherwise, it uses the attributes that are passed as
        ### TODO HINT : parameters of the fit function.
        ### TODO : This function should be vectorized so that it can be computed on a single individual or multiple individuals
        if attributes == False:
            attributes = self.attributes

        g = attributes['g']
        deltas = attributes['deltas']
        mixing_matrix = attributes['mixing_matrix']

        _unique_indiv = True
        if _unique_indiv:
            reparametrized_time = np.exp(xi)* (age - tau)
            wi = np.dot(mixing_matrix, sources)
            eta = - (g * np.exp(-deltas) + 1)**2 / (g*np.exp(-deltas))
            eta = -eta * wi - deltas - reparametrized_time
            eta = 1./(1. + g * np.exp(eta))
        else:
            ### TODO : Here goes the tensorized version
            pass

        return eta