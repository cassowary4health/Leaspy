import os

from src import default_data_dir
from src.models.abstract_model import AbstractModel
from src.inputs.model_settings import ModelSettings
import torch
from torch.autograd import Variable
import numpy as np
import src.utils.conformity.Profiler
import json
from scipy import stats

class MultivariateModel(AbstractModel):
    # TODO dimension in multivariate model parameters initialization ???
    def __init__(self):
        data_dir = os.path.join(default_data_dir, "default_multivariate_parameters.json")
        reader = ModelSettings(data_dir)
        self.model_parameters = reader.parameters
        self.dimension = None

        if reader.model_type != 'multivariate':
            raise ValueError("The default multivariate parameters are not of multivariate type")




        self.model_name = 'multivariate'

        # TODO Change hyperparameter
        self.source_dimension = 1
        self.a_matrix = torch.Tensor(np.random.normal(loc=0, scale=1e-2, size=(4, self.source_dimension)))

        self.reals_pop_name = ['p0','v0']
        self.reals_ind_name = ['xi','tau']+['s'+str(i) for i in range(self.source_dimension)]


    ###########################
    ## Core
    ###########################

    # TODO Numba this
    #@src.utils.conformity.Profiler.do_profile()
    """
    def compute_individual(self, individual, reals_pop, real_ind):

        # Load from dict
        v0 = reals_pop['v0']
        #g = self.cache_variables['g']
        #b = self.cache_variables['b']
        p0 = reals_pop['p0']

        g = torch.pow(p0, -1) - 1
        b = torch.pow(p0, 2) * g

        # Time reparametrized
        a = v0*torch.exp(real_ind['xi'])*(individual.tensor_timepoints-real_ind['tau'])
        #a += torch.mm(self.a_matrix, real_ind['s0']).t()
        #TODO LAter
        #a += (self.a_matrix*real_ind['s0']).t()

        # Compute parallel curve
        parallel_curve = torch.pow(1 + g * torch.exp(-a / b), -1)

        return parallel_curve"""

    def compute_individual(self,  individual, reals_pop, real_ind):
        p0 = reals_pop['p0']
        v0 = reals_pop['v0']
        reparametrized_time = v0*torch.exp(real_ind['xi'])*(individual.tensor_timepoints-real_ind['tau'])
        return torch.pow(1 + (1 / p0 - 1) * torch.exp(-reparametrized_time / (p0 * (1 - p0))), -1)


    def compute_average(self, tensor_timepoints):
        p0 = torch.Tensor(self.model_parameters['p0'])
        v0 = torch.Tensor(self.model_parameters['v0'])
        reparametrized_time = v0*np.exp(self.model_parameters['xi_mean'])*(tensor_timepoints-self.model_parameters['tau_mean'])
        return torch.pow(1 + (1 / p0 - 1) * torch.exp(-reparametrized_time / (p0 * (1 - p0))), -1)

    def compute_sufficient_statistics(self, data, realizations):

        reals_pop, reals_ind = realizations

        # Tau
        tau_array = []
        for idx in reals_ind.keys():
            tau_array.append(reals_ind[idx]['tau'])
        tau_array = torch.Tensor(tau_array)

        tau_mean = np.mean(tau_array.detach().numpy()).tolist()
        tau_var = np.var(tau_array.detach().numpy()).tolist()

        # Ksi
        xi_array = []
        for idx in reals_ind.keys():
            xi_array.append(reals_ind[idx]['xi'])
        xi_array = torch.Tensor(xi_array)

        xi_mean = np.mean(xi_array.detach().numpy()).tolist()
        xi_var = np.var(xi_array.detach().numpy()).tolist()

        # P0
        p0 = reals_pop['p0'].detach().numpy()

        # V0
        v0 = reals_pop['v0'].detach().numpy()

        # Sources
        sources_array = []
        for idx in reals_ind.keys():
            sources_array.append(reals_ind[idx]['s0'].detach().numpy())
        sources_array = np.array(sources_array)

        sources_var = np.var(sources_array).tolist()


        sufficient_statistics = {}
        sufficient_statistics['p0'] = p0
        sufficient_statistics['v0'] = v0
        sufficient_statistics['tau_mean'] = tau_mean
        sufficient_statistics['tau_var'] = tau_var
        sufficient_statistics['xi_mean'] = xi_mean
        sufficient_statistics['xi_var'] = xi_var
        sufficient_statistics['sum_squared'] = float(self.compute_sumsquared(data, reals_pop, reals_ind).detach().numpy())
        sufficient_statistics['s0_var'] = sources_var

        # TODO : non identifiable here with the xi, but how do we update each xi ?
        #sufficient_statistics['v0'][sufficient_statistics['v0'] < 0] = 0.001
        #elf.model_parameters['v0'] = np.exp(sufficient_statistics['xi_mean']) * np.array(self.model_parameters['v0'])
        #for idx in reals_ind.keys():
        #    reals_ind[idx]['xi'] = reals_ind[idx]['xi']-sufficient_statistics['xi_mean']
        #sufficient_statistics['xi_mean'] = 0.0


        return sufficient_statistics



    def update_model(self, data, sufficient_statistics):

        #TODO parameters, automatic initialization of these parameters
        m_xi = data.n_individuals/20
        sigma2_xi_0 = 0.05

        m_tau = data.n_individuals / 20
        sigma2_tau_0 = 1

        self.model_parameters['p0'] = sufficient_statistics['p0']
        self.model_parameters['v0'] = sufficient_statistics['v0']

        # Tau
        self.model_parameters['tau_mean'] = sufficient_statistics['tau_mean']
        tau_var_update = (1/(data.n_individuals+m_tau))*(data.n_individuals * sufficient_statistics['tau_var']+m_tau*sigma2_tau_0)
        self.model_parameters['tau_var'] = tau_var_update

        self.model_parameters['xi_mean'] = sufficient_statistics['xi_mean']
        tau_var_update = (1/(data.n_individuals+m_xi))*(data.n_individuals * sufficient_statistics['xi_var']+m_tau*sigma2_xi_0)
        self.model_parameters['xi_var'] = tau_var_update

        # Sources
        #self.model_parameters['s0_var'] = sufficient_statistics['s0_var']

        # Noise
        self.model_parameters['noise_var'] = sufficient_statistics['sum_squared']/data.n_observations

        # Update the Random Variables
        self._update_random_variables()

        # Update Cached Variables
        self.cache_variables['noise_inverse'] = 1/self.model_parameters['noise_var']
        self.cache_variables['constant_fit_variable'] = np.log(np.sqrt(2 * np.pi * self.model_parameters['noise_var']))

        # Compute the a_matrix
        self.update_a_matrix()

    def update_a_matrix(self):

        # TODO better this

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

        # Householer to get Q
        Q = self.householder()

        # Product with Beta
        A = Q.dot(self.model_parameters['beta'])"""



    def householder(self):

        s = self.model_parameters['v0']
        e1 = np.repeat(0, self.dimension)
        e1[0] = 1
        a = (s+np.sign(s[0])*np.linalg.norm(s)*e1).reshape(1, -1)
        q_matrix = np.identity(self.dimension)-2*a.T.dot(a)/(a.dot(a.T))
        orthogonal_matrix = q_matrix[:, 1:]

        return orthogonal_matrix



    def update_cache_variables(self, reals, keys):
        # Update the p0
        if 'p0' in keys:
            self.cache_variables['g'] = torch.pow(reals['p0'], -1) - 1
            self.cache_variables['b'] = torch.pow(reals['p0'], 2) * self.cache_variables['g']



    def smart_initialization(self, data):

        # Initializes Dimension
        self.dimension = data.dimension

        ## Linear Regression on each feature
        p0_array = [None] * self.dimension
        v0_array = [None] * self.dimension
        noise_array = [None] * self.dimension

        df = data.to_pandas()
        x = df.index.get_level_values('TIMES').values

        for dim in range(self.dimension):
            y = df.iloc[:, dim].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            p0_array[dim], v0_array[dim] = intercept, slope
            noise_array[dim] = np.mean((intercept+slope*x-y)**2)**2


        # Pre-Initialize from dimension
        SMART_INITIALIZATION = {
            'p0': np.repeat(np.mean(p0_array), self.dimension),
            'v0' : np.repeat(np.mean(v0_array), self.dimension),
            'tau_mean': 0., 'tau_var': 1.0,
            'xi_mean': 0., 'xi_var': 0.5,
            'noise_var': 0.005
        }

        SMART_INITIALIZATION = {
            'p0': p0_array,
            'v0' : v0_array,
            'tau_mean': 0., 'tau_var': 1.0,
            'xi_mean': 0., 'xi_var': 0.5,
            'noise_var': 0.005
        }

        # Initializes Parameters
        for parameter_key in self.model_parameters.keys():
            if self.model_parameters[parameter_key] is None:
                self.model_parameters[parameter_key] = SMART_INITIALIZATION[parameter_key]

        # Initialize Cache
        self._initialize_cache_variables()


    def save_parameters(self, path):


        #TODO check que c'est le bon format (IGOR)
        model_settings = {}

        model_settings['parameters'] = self.model_parameters
        model_settings['dimension'] = self.dimension
        model_settings['type'] = self.model_name

        if type(model_settings['parameters']['p0']) not in [list]:
            model_settings['parameters']['p0'] = model_settings['parameters']['p0'].tolist()

        if type(model_settings['parameters']['v0']) not in [list]:
            model_settings['parameters']['v0'] = model_settings['parameters']['v0'].tolist()


        with open(path, 'w') as fp:
            json.dump(model_settings, fp)
