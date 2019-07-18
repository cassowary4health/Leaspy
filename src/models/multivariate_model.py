import os
from src.utils.realizations.collection_realization import CollectionRealization
from src import default_data_dir
from src.models.abstract_model import AbstractModel
from src.inputs.model_settings import ModelSettings
import torch
from torch.autograd import Variable
import numpy as np
import json
from scipy import stats
from src.models.utils.attributes.attributes_multivariate import Attributes_Multivariate


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
            'g': p0_array,
            'v0': v0_array,
            'betas': np.zeros(shape=(self.dimension - 1, self.source_dimension)),
            'tau_mean': df.index.values.mean(), 'tau_std': 1.0,
            'xi_mean': 0., 'xi_std': 0.5,
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
        realizations = self.get_realization_object(data)
        self.update_MCMC_toolbox('all', realizations)

    def update_MCMC_toolbox(self, name_of_the_variable_that_has_been_changed, realizations):
        values = {
            'g': realizations['g'].tensor_realizations.detach().numpy(),
            'v0': realizations['v0'].tensor_realizations.detach().numpy(),
            'betas': realizations['betas'].tensor_realizations.detach().numpy()
        }
        self.MCMC_toolbox['attributes'].update([name_of_the_variable_that_has_been_changed], values)

    def compute_individual_tensorized(self, data, realizations):

        # Population parameters
        # TODO : Change the attributes!
        a_matrix = self.MCMC_toolbox['attributes'].mixing_matrix
        v0 = torch.exp(realizations['v0'].tensor_realizations)
        g = torch.exp(realizations['g'].tensor_realizations)
        b = g / ((1.+g)*(1.+g))

        # Individual parameters
        xi = realizations['xi'].tensor_realizations
        tau = realizations['tau'].tensor_realizations
        wi = torch.nn.functional.linear(realizations['sources'].tensor_realizations, a_matrix, bias=None)
        timepoints = data.timepoints.reshape(data.timepoints.shape[0], data.timepoints.shape[1], 1)  # TODO change timepoints dimension in data structure
        reparametrized_time = torch.exp(xi) * (timepoints - tau)

        # Log likelihood computation
        LL = v0 * reparametrized_time + wi
        LL = 1. + g * torch.exp(-LL / b)
        model = 1. / LL

        return model * data.mask

    def compute_individual_attachment_tensorized(self, data, realizations):
        squared_sum = self.compute_sum_squared_tensorized(data, realizations)
        noise_var = self.parameters['noise_std'] ** 2
        attachment = 0.5 * (1 / noise_var) * squared_sum
        attachment += np.log(np.sqrt(2 * np.pi * noise_var))

        return attachment

    def update_model_parameters(self, data, sufficient_statistics, burn_in_phase=True):
        # Memoryless part of the algorithm
        if burn_in_phase:
            self.parameters['g'] = sufficient_statistics['g'].tensor_realizations.detach().numpy()
            self.parameters['v0'] = sufficient_statistics['v0'].tensor_realizations.detach().numpy()
            self.parameters['betas'] = sufficient_statistics['betas'].tensor_realizations.detach().numpy()
            xi = sufficient_statistics['xi'].tensor_realizations.detach().numpy()
            self.parameters['xi_mean'] = np.mean(xi)
            self.parameters['xi_std'] = np.std(xi)
            tau = sufficient_statistics['tau'].tensor_realizations.detach().numpy()
            self.parameters['tau_mean'] = np.mean(tau)
            self.parameters['tau_std'] = np.std(tau)

            squared_diff = self.compute_sum_squared_tensorized(data, sufficient_statistics).sum().numpy()
            self.parameters['noise_std'] = np.sqrt(squared_diff / (data.n_visits * data.dimension))

        # Stochastic sufficient statistics used to update the parameters of the model
        else:
            # TODO : To finish
            return


    def save_parameters(self, path):
        # TODO later
        return 0

        # TODO, shouldnt be this be in the output manager ???
        # TODO check que c'est le bon format (IGOR)
        model_settings = {}

        model_settings['parameters'] = self.parameters
        model_settings['dimension'] = self.dimension
        model_settings['source_dimension'] = self.dimension
        model_settings['type'] = self.model_name

        if type(model_settings['parameters']['g']) not in [list]:
            model_settings['parameters']['g'] = model_settings['parameters']['g'].tolist()

        if type(model_settings['parameters']['v0']) not in [list]:
            model_settings['parameters']['v0'] = model_settings['parameters']['v0'].tolist()

        if type(model_settings['parameters']['betas']) not in [list]:
            model_settings['parameters']['betas'] = model_settings['parameters']['betas'].tolist()
            # beta = model_settings['parameters']['beta']
            # beta_n_columns = beta.shape[1]
            # model_settings['parameters'].pop('beta')
            # Save per column
            # for beta_dim in range(beta_n_columns):
            #    beta_column = beta[beta_dim].tolist()
            #    model_settings['parameters']['beta_'+ str(beta_dim)] = beta_column

        with open(path, 'w') as fp:
            json.dump(model_settings, fp)


    def compute_sufficient_statistics(self, data, realizations):
        # Tau
        tau_array = realizations['tau'].tensor_realizations
        tau_mean = np.mean(tau_array.detach().numpy()).tolist()
        tau_var = np.var(tau_array.detach().numpy()).tolist()

        # Ksi
        xi_array = realizations['xi'].tensor_realizations
        xi_mean = np.mean(xi_array.detach().numpy()).tolist()
        xi_var = np.var(xi_array.detach().numpy()).tolist()

        # Sources
        sources_array = realizations['sources'].tensor_realizations
        sources_var = np.var(sources_array.detach().numpy()).tolist()

        # P0
        g = realizations['g'].tensor_realizations.detach().numpy()

        # V0
        v0 = realizations['v0'].tensor_realizations.detach().numpy()

        # Beta
        beta = realizations['beta'].tensor_realizations.detach().numpy()

        # Compute sufficient statistics
        sufficient_statistics = {}
        sufficient_statistics['g'] = g
        sufficient_statistics['v0'] = v0
        sufficient_statistics['beta'] = beta
        sufficient_statistics['tau_mean'] = tau_mean
        sufficient_statistics['tau_var'] = tau_var
        sufficient_statistics['xi_mean'] = xi_mean
        sufficient_statistics['xi_var'] = xi_var
        sufficient_statistics['sum_squared'] = float(
            torch.sum(self.compute_sum_squared_tensorized(data, realizations).detach()).numpy())
        sufficient_statistics['empirical_sources_var'] = sources_var

        # TODO : non identifiable here with the xi, but how do we update each xi ?
        sufficient_statistics['v0'][sufficient_statistics['v0'] < 0.01] = 0.01
        self.model_parameters['v0'] = np.exp(sufficient_statistics['xi_mean']) * np.array(self.model_parameters['v0'])
        realizations['xi'].tensor_realizations = realizations['xi'].tensor_realizations - sufficient_statistics[
            'xi_mean']
        sufficient_statistics['xi_mean'] = sufficient_statistics['xi_mean']

        return sufficient_statistics

    def random_variable_informations(self):

        ## Population variables
        g_infos = {
            "name": "g",
            "shape": (1, self.dimension),
            "type": "population",
            "rv_type": "multigaussian"
        }

        v0_infos = {
            "name": "v0",
            "shape": (1, self.dimension),
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
