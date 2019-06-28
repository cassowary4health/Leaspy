import os

from src import default_data_dir
from src.models.abstract_model import AbstractModel
from src.inputs.model_settings import ModelSettings
import torch
from torch.autograd import Variable
import numpy as np
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
        #self.source_dimension = 2

        self.reals_pop_name = ['p0', 'v0', 'beta']
        self.reals_ind_name = ['xi', 'tau', 'sources']



    def get_pop_shapes(self):
        p0_shape = (1, self.dimension)
        v0_shape = (1, self.dimension)
        beta_shape = (self.dimension-1, self.source_dimension)

        return {"p0": p0_shape,
                "v0": v0_shape,
                "beta": beta_shape}


    def get_info_variables(self):


            ## Population variables

            p0_infos = {
                "name": "p0",
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

            beta_infos = {
                "name": "beta",
                "shape": (self.dimension-1, self.source_dimension),
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
                "v0": v0_infos,
                "beta": beta_infos,
                "tau": tau_infos,
                "xi": xi_infos,
                "sources": sources_infos
            }

            return variables_infos

    #TODO better this
    def update_variable_info(self, key, reals_pop):
        if key in ['v0']:
            self.update_Q_matrix()
        if key in ['v0','beta']:
            self.update_a_matrix(reals_pop['beta'])



    ###########################
    ## Core
    ###########################

    # TODO Numba this
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

        # Problem with the case multi...
        a += torch.mm(self.a_matrix, real_ind['sources'].t()).t()
        #TODO LAter
        #a += (self.a_matrix*real_ind['s0']).t()

        # Compute parallel curve
        parallel_curve = torch.pow(1 + g * torch.exp(-a / b), -1)

        return parallel_curve

    """
    def compute_individual(self,  individual, reals_pop, real_ind):
        p0 = reals_pop['p0']
        v0 = reals_pop['v0']
        reparametrized_time = v0*torch.exp(real_ind['xi'])*(individual.tensor_timepoints-real_ind['tau'])
        return torch.pow(1 + (1 / p0 - 1) * torch.exp(-reparametrized_time / (p0 * (1 - p0))), -1)"""


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

        # Beta
        beta = reals_pop['beta'].detach().numpy()

        # Sources
        sources_array = []
        for idx in reals_ind.keys():
            sources_array.append(reals_ind[idx]['sources'].detach().numpy())
        sources_array = np.array(sources_array)

        sources_var = np.var(sources_array).tolist()


        sufficient_statistics = {}
        sufficient_statistics['p0'] = p0
        sufficient_statistics['v0'] = v0
        sufficient_statistics['beta'] = beta
        sufficient_statistics['tau_mean'] = tau_mean
        sufficient_statistics['tau_var'] = tau_var
        sufficient_statistics['xi_mean'] = xi_mean
        sufficient_statistics['xi_var'] = xi_var
        sufficient_statistics['sum_squared'] = float(self.compute_sumsquared(data, reals_pop, reals_ind).detach().numpy())
        sufficient_statistics['empirical_sources_var'] = sources_var

        # TODO : non identifiable here with the xi, but how do we update each xi ?
        #sufficient_statistics['v0'][sufficient_statistics['v0'] < 0.01] = 0.01
        #self.model_parameters['v0'] = np.exp(sufficient_statistics['xi_mean']/2) * np.array(self.model_parameters['v0'])
        #for idx in reals_ind.keys():
        #    reals_ind[idx]['xi'] = reals_ind[idx]['xi']-sufficient_statistics['xi_mean']/2
        #ufficient_statistics['xi_mean'] = sufficient_statistics['xi_mean']/2


        return sufficient_statistics



    def update_model(self, data, sufficient_statistics):

        #TODO parameters, automatic initialization of these parameters
        m_xi = data.n_individuals/20
        sigma2_xi_0 = 0.05

        m_tau = data.n_individuals / 20
        sigma2_tau_0 = 1

        self.model_parameters['p0'] = sufficient_statistics['p0']
        self.model_parameters['v0'] = sufficient_statistics['v0']
        self.model_parameters['beta'] = sufficient_statistics['beta']

        # Tau
        self.model_parameters['tau_mean'] = sufficient_statistics['tau_mean']
        tau_var_update = (1/(data.n_individuals+m_tau))*(data.n_individuals * sufficient_statistics['tau_var']+m_tau*sigma2_tau_0)
        self.model_parameters['tau_var'] = tau_var_update

        self.model_parameters['xi_mean'] = sufficient_statistics['xi_mean']
        tau_var_update = (1/(data.n_individuals+m_xi))*(data.n_individuals * sufficient_statistics['xi_var']+m_tau*sigma2_xi_0)
        self.model_parameters['xi_var'] = tau_var_update

        # Sources
        self.model_parameters['empirical_sources_var'] = sufficient_statistics['empirical_sources_var']
        self.model_parameters['sources_var'] = 1.0

        # Noise
        self.model_parameters['noise_var'] = sufficient_statistics['sum_squared']/data.n_observations

        # Update the Random Variables
        self._update_random_variables()

        # Update Cached Variables
        self.cache_variables['noise_inverse'] = 1/self.model_parameters['noise_var']
        self.cache_variables['constant_fit_variable'] = np.log(np.sqrt(2 * np.pi * self.model_parameters['noise_var']))

        # Compute the a_matrix
        #self.update_a_matrix()

    def update_Q_matrix(self):
        self.Q_matrix = torch.tensor(self.householder()).type(torch.FloatTensor)

    def update_a_matrix(self, real_beta):

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

        # Product with Beta
        self.a_matrix = torch.mm(self.Q_matrix, real_beta)



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
        """
        SMART_INITIALIZATION = {
            'p0': np.repeat(np.mean(p0_array), self.dimension),
            'v0' : np.repeat(np.mean(v0_array), self.dimension),
            'beta': np.zeros(shape=(self.dimension - 1, self.source_dimension)),
            'tau_mean': 0., 'tau_var': 1.0,
            'xi_mean': 0., 'xi_var': 0.5,
            'noise_var': 0.005
        }"""


        SMART_INITIALIZATION = {
            'p0': p0_array,
            'v0' : v0_array,
            'beta': np.zeros(shape=(self.dimension-1, self.source_dimension)),
            'tau_mean': 0., 'tau_var': 1.0,
            'xi_mean': 0., 'xi_var': 0.5,
            'noise_var': None
        }

        # Initializes Parameters
        for parameter_key in self.model_parameters.keys():
            if self.model_parameters[parameter_key] is None:
                self.model_parameters[parameter_key] = SMART_INITIALIZATION[parameter_key]




    def save_parameters(self, path):

        #TODO, shouldnt be this be in the output manager ???
        #TODO check que c'est le bon format (IGOR)
        model_settings = {}

        model_settings['parameters'] = self.model_parameters
        model_settings['dimension'] = self.dimension
        model_settings['type'] = self.model_name

        if type(model_settings['parameters']['p0']) not in [list]:
            model_settings['parameters']['p0'] = model_settings['parameters']['p0'].tolist()

        if type(model_settings['parameters']['v0']) not in [list]:
            model_settings['parameters']['v0'] = model_settings['parameters']['v0'].tolist()

        if type(model_settings['parameters']['beta']) not in [list]:
            model_settings['parameters']['beta'] = model_settings['parameters']['beta'].tolist()
            #beta = model_settings['parameters']['beta']
            #beta_n_columns = beta.shape[1]
            #model_settings['parameters'].pop('beta')
            # Save per column
            #for beta_dim in range(beta_n_columns):
            #    beta_column = beta[beta_dim].tolist()
            #    model_settings['parameters']['beta_'+ str(beta_dim)] = beta_column

        with open(path, 'w') as fp:
            json.dump(model_settings, fp)
