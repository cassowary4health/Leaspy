from .abstract_model import AbstractModel
from .utils.attributes.attributes_multivariate import Attributes_Multivariate

import torch
import numpy as np
import json
from scipy import stats



class MultivariateModel(AbstractModel):
    def __init__(self, name):
        super(MultivariateModel, self).__init__(name)

        self.source_dimension = None
        self.parameters = {
            "g": None, "betas": None, "v0": None,
            "tau_mean": None, "tau_std": None,
            "xi_mean": None, "xi_std": None,
            "sources_mean": None, "sources_std": None,
            "noise_std": None
        }
        self.bayesian_priors = None
        self.attributes = None

        ### MCMC related "parameters"
        self.MCMC_toolbox = {
            'attributes': None,
            'priors': {
                'g_std': None,  # tq p0 = 1 / (1+exp(g)) i.e. g = 1/p0 - 1
                'v0_std': None,
                'betas_std': None
            }
        }

    def load_hyperparameters(self, hyperparameters):
        self.dimension = hyperparameters['dimension']
        self.source_dimension = hyperparameters['source_dimension']

    def save(self, path):
        model_parameters_save = self.parameters.copy()
        for key, value in model_parameters_save.items():
            if type(value) in [torch.Tensor]:
                model_parameters_save[key] = value.tolist()
        model_settings = {
            'name': 'multivariate_parallel',
            'dimension': self.dimension,
            'source_dimension': self.source_dimension,
            'parameters': model_parameters_save
        }

        with open(path, 'w') as fp:
            json.dump(model_settings, fp)

    def initialize(self, data):

        self.dimension = data.dimension
        self.source_dimension = int(data.dimension / 2.)  # TODO : How to change it independently of the initialize?

        ### TODO : Have a better initialization with the new G and exp(v0) parameters
        # Linear Regression on each feature
        p0_array = [None] * self.dimension
        v0_array = [None] * self.dimension
        noise_array = [None] * self.dimension

        df = data.to_pandas()
        x = df.index.get_level_values('TIMES').values

        for dim in range(self.dimension):
            y = df.iloc[:, dim].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            p0_array[dim], v0_array[dim] = intercept, slope
            noise_array[dim] = np.mean((intercept + slope * x - y) ** 2) ** 2

            # V0 array minimum value
            v0_array[dim] = max(v0_array[dim], -3)

        SMART_INITIALIZATION = {
            'g': torch.tensor(p0_array),
            'v0': torch.tensor(v0_array),
            'betas': torch.zeros((self.dimension - 1, self.source_dimension)),
            'tau_mean': df.index.values.mean(), 'tau_std': 1.0,
            'xi_mean': -3., 'xi_std': 0.05,
            'sources_mean': 0.0, 'sources_std': 1.0,
            'noise_std': 0.1
        }

        # Initializes Parameters
        for parameter_key in self.parameters.keys():
            if self.parameters[parameter_key] is None:
                self.parameters[parameter_key] = SMART_INITIALIZATION[parameter_key]

        self.attributes = Attributes_Multivariate(self.dimension, self.source_dimension)
        self.is_initialized = True

    def initialize_MCMC_toolbox(self, data):
        self.MCMC_toolbox = {
            'priors': {'g_std': 0.01, 'v0_std': 0.01, 'betas_std': 0.01},
            'attributes': Attributes_Multivariate(self.dimension, self.source_dimension)
        }
        realizations = self.get_realization_object(data.n_individuals)
        self.update_MCMC_toolbox(['all'], realizations)

    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        L = name_of_the_variables_that_have_been_changed
        values = {}
        if any(c in L for c in ('g', 'all')):
            values['g'] = realizations['g'].tensor_realizations
        if any(c in L for c in ('v0', 'all')):
            values['v0'] = realizations['v0'].tensor_realizations
        if any(c in L for c in ('betas', 'all')):
            values['betas'] = realizations['betas'].tensor_realizations

        self.MCMC_toolbox['attributes'].update(name_of_the_variables_that_have_been_changed, values)

    def _get_attributes(self,MCMC):
        if MCMC:
            g = self.MCMC_toolbox['attributes'].g
            v0 = self.MCMC_toolbox['attributes'].v0
            a_matrix = self.MCMC_toolbox['attributes'].mixing_matrix
        else:
            g = self.attributes.g
            v0 = self.attributes.v0
            a_matrix = self.attributes.mixing_matrix
        return g, v0, a_matrix

    def compute_sum_squared_tensorized(self,data,param_ind,MCMC):
        res = self.compute_individual_tensorized(data.timepoints, param_ind, MCMC)
        res *= data.mask
        return torch.sum((res * data.mask - data.values) ** 2, dim=(1, 2))

    def compute_individual_tensorized(self, timepoints, ind_parameters, MCMC=False):
        # Population parameters
        g, v0, a_matrix = self._get_attributes(MCMC)
        b = g / ((1.+g)*(1.+g))

        # Individual parameters
        xi,tau,sources = ind_parameters
        wi = torch.nn.functional.linear(sources, a_matrix, bias=None)
        # TODO change timepoints dimension in data structure
        reparametrized_time = self.time_reparametrization(timepoints,xi,tau)
        # Log likelihood computation
        a = tuple([1]*reparametrized_time.ndimension())
        v0 = v0.unsqueeze(0).repeat(*tuple(reparametrized_time.shape),1)
        reparametrized_time = reparametrized_time.unsqueeze(-1).repeat(*a,v0.shape[-1])

        LL = v0* reparametrized_time + wi.unsqueeze(-2)
        LL = 1. + g * torch.exp(-LL / b)
        model = 1. / LL

        return model


    def compute_sufficient_statistics(self, data, realizations):
        sufficient_statistics = {
            'g': realizations['g'].tensor_realizations.detach(),
            'v0': realizations['v0'].tensor_realizations.detach(),
            'betas': realizations['betas'].tensor_realizations.detach(),
            'tau': realizations['tau'].tensor_realizations,
            'tau_sqrd': torch.pow(realizations['tau'].tensor_realizations, 2),
            'xi': realizations['xi'].tensor_realizations,
            'xi_sqrd': torch.pow(realizations['xi'].tensor_realizations, 2)
        }

        # TODO : Optimize to compute the matrix multiplication only once for the reconstruction
        xi, tau, sources = self.get_param_from_real(realizations)
        data_reconstruction = self.compute_individual_tensorized(data.timepoints, (xi,tau,sources),MCMC=True)
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
            self.update_model_parameters_burn_in(data, suff_stats)
        # Stochastic sufficient statistics used to update the parameters of the model
        else:
            self.update_model_parameters_normal(data, suff_stats)
        self.attributes.update(['all'],self.parameters)

    def update_model_parameters_burn_in(self, data, suff_stats):
        # Memoryless part of the algorithm
        self.parameters['g'] = suff_stats['g'].tensor_realizations.detach()
        self.parameters['v0'] = suff_stats['v0'].tensor_realizations.detach()
        self.parameters['betas'] = suff_stats['betas'].tensor_realizations.detach()
        xi = suff_stats['xi'].tensor_realizations.detach()
        self.parameters['xi_mean'] = torch.mean(xi)
        self.parameters['xi_std'] = torch.std(xi)
        tau = suff_stats['tau'].tensor_realizations.detach()
        self.parameters['tau_mean'] = torch.mean(tau)
        self.parameters['tau_std'] = torch.std(tau)

        param_ind = self.get_param_from_real(suff_stats)
        squared_diff = self.compute_sum_squared_tensorized(data, param_ind,MCMC=True).sum()
        self.parameters['noise_std'] = np.sqrt(squared_diff / (data.n_visits * data.dimension))

        # Stochastic sufficient statistics used to update the parameters of the model
    def update_model_parameters_normal(self, data, suff_stats):
        # TODO with Raphael : check the SS, especially the issue with mean(xi) and v_k
        # TODO : 1. Learn the mean of xi and v_k
        # TODO : 2. Set the mean of xi to 0 and add it to the mean of V_k
        self.parameters['g'] = suff_stats['g']
        self.parameters['v0'] = suff_stats['v0']
        self.parameters['betas'] = suff_stats['betas']

        tau_mean = self.parameters['tau_mean']
        tau_std_updt = torch.mean(suff_stats['tau_sqrd']) - 2 * tau_mean * torch.mean(suff_stats['tau'])
        self.parameters['tau_std'] = torch.sqrt(tau_std_updt + self.parameters['tau_mean'] ** 2)
        self.parameters['tau_mean'] = torch.mean(suff_stats['tau'])

        xi_mean = self.parameters['xi_mean']
        xi_std_updt = torch.mean(suff_stats['xi_sqrd']) - 2 * xi_mean * torch.mean(suff_stats['xi'])
        self.parameters['xi_std'] = torch.sqrt(xi_std_updt + self.parameters['xi_mean'] ** 2)
        self.parameters['xi_mean'] = torch.mean(suff_stats['xi'])

        S1 = torch.sum(suff_stats['obs_x_obs'])
        S2 = torch.sum(suff_stats['obs_x_reconstruction'])
        S3 = torch.sum(suff_stats['reconstruction_x_reconstruction'])

        self.parameters['noise_std'] = torch.sqrt((S1 - 2. * S2 + S3) / (data.dimension * data.n_visits))

    def random_variable_informations(self):

        ## Population variables
        g_infos = {
            "name": "g",
            "shape": torch.Size([self.dimension]),
            "type": "population",
            "rv_type": "multigaussian"
        }

        v0_infos = {
            "name": "v0",
            "shape": torch.Size([self.dimension]),
            "type": "population",
            "rv_type": "multigaussian"
        }

        betas_infos = {
            "name": "betas",
            "shape": torch.Size([self.dimension - 1, self.source_dimension]),
            "type": "population",
            "rv_type": "multigaussian"
        }

        ## Individual variables
        tau_infos = {
            "name": "tau",
            "shape": torch.Size([1]),
            "type": "individual",
            "rv_type": "gaussian"
        }

        xi_infos = {
            "name": "xi",
            "shape": torch.Size([1]),
            "type": "individual",
            "rv_type": "gaussian"
        }

        sources_infos = {
            "name": "sources",
            "shape": torch.Size([self.source_dimension]),
            "type": "individual",
            "rv_type": "gaussian"
        }

        variables_infos = {
            "g": g_infos,
            "v0": v0_infos,
            "betas": betas_infos,
            "tau": tau_infos,
            "xi": xi_infos,
            "sources": sources_infos
        }

        return variables_infos

    '''
    def update_a_matrix(self, real_beta, real_v0):

        # TODO better this
        """

        ## Alex method

        v0 = torch.Tensor(self.model_parameters['v0']).reshape(-1, 1)

        # Compute projection
        scalar_product = torch.mm(self.a_matrix.t(), v0)
        num = torch.mm(scalar_product, v0.t())
        den = torch.sum(v0**2)

        a = self.a_matrix - (num/den).t()
        # Assigns
        self.a_matrix = a

        ## Householder
        """
        # Update the Q
        self.update_Q_matrix(real_v0)

        # Product with Beta
        self.a_matrix = torch.mm(self.Q_matrix, real_beta)
        
    def update_Q_matrix(self, real_v0):
        self.Q_matrix = torch.tensor(self.householder(real_v0)).type(torch.FloatTensor)

    def householder(self, real_v0):
        s = real_v0.detach().numpy()
        e1 = np.repeat(0, self.dimension)
        e1[0] = 1
        a = (s + np.sign(s[0]) * np.linalg.norm(s) * e1).reshape(1, -1)
        q_matrix = np.identity(self.dimension) - 2 * a.T.dot(a) / (a.dot(a.T))
        orthogonal_matrix = q_matrix[:, 1:]
        return orthogonal_matrix  
        
    
    def update_model(self, data, sufficient_statistics):

        # TODO parameters, automatic initialization of these parameters
        m_xi = data.n_individuals / 20
        sigma2_xi_0 = 0.05

        m_tau = data.n_individuals / 20
        sigma2_tau_0 = 1

        self.model_parameters['g'] = sufficient_statistics['g']
        self.model_parameters['v0'] = sufficient_statistics['v0']
        self.model_parameters['beta'] = sufficient_statistics['beta']

        # Tau
        self.model_parameters['tau_mean'] = sufficient_statistics['tau_mean']
        tau_var_update = (1 / (data.n_individuals + m_tau)) * (
                    data.n_individuals * sufficient_statistics['tau_var'] + m_tau * sigma2_tau_0)
        self.model_parameters['tau_var'] = tau_var_update

        self.model_parameters['xi_mean'] = sufficient_statistics['xi_mean']
        tau_var_update = (1 / (data.n_individuals + m_xi)) * (
                    data.n_individuals * sufficient_statistics['xi_var'] + m_tau * sigma2_xi_0)
        self.model_parameters['xi_var'] = tau_var_update

        # Sources
        self.model_parameters['empirical_sources_var'] = sufficient_statistics['empirical_sources_var']
        self.model_parameters['sources_var'] = 1.0

        # Noise
        self.model_parameters['noise_var'] = sufficient_statistics['sum_squared'] / (data.n_observations)

        # Update the Random Variables
        # TODO :  Is it useful to have the random variables here? Haven't they disappeared now?
        self._update_random_variables()

        # Update Cached Variables
        # TODO : Same question as above? Are they still used?
        self.cache_variables['noise_inverse'] = 1 / self.model_parameters['noise_var']
        self.cache_variables['constant_fit_variable'] = np.log(np.sqrt(2 * np.pi * self.model_parameters['noise_var']))
  
    '''
