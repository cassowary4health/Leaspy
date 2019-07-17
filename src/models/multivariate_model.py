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
    def __init__(self):

        self.model_name = 'Multivariate'
        self.dimension = None
        self.source_dimension = None
        self.is_initialized = False
        self.parameters = {
            "p0": None, "betas": None, "v0": None,
            "mean_tau": None, "sigma_tau": None,
            "mean_xi": None,  "sigma_xi": None,
            "mean_sources": None, "sigma_sources": None,
            "sigma_noise": None

        }
        self.bayesian_priors = None
        self.attributes = None

        ### MCMC related "parameters"
        self.MCMC_toolbox = {
            'attributes': None,
            'priors': {
                'sigma_p0': None, # tq p0 = 1 / (1+exp(g)) i.e. g = 1/p0 - 1
                'sigma_v0': None, # tq deltas = user_delta_in_years * v0 / (p0(1-p0))
                'sigma_betas': None
            }
        }

    ###########################
    ## Initialization
    ###########################

    def initialize_MCMC_toolbox(self, dataset):

        self.MCMC_toolbox['priors'] = {
            'sigma_p0': 0.01,
            'sigma_v0': 0.01,
            'sigma_betas': 0.01
        }

        self.MCMC_toolbox['attributes'] = Attributes_Multivariate(self.dimension, self.source_dimension)

        values = {
            'p0': self.parameters['p0'],
            'v0': self.parameters['v0'],
            'betas': self.parameters['betas']
        }
        values['mean_tau'] = self.parameters['mean_tau']
        values['mean_xi'] = self.parameters['mean_xi']
        self.MCMC_toolbox['attributes'].update(['all'], values)


    def update_MCMC_toolbox(self, name_of_the_variable_that_has_been_changed, realizations):
        """
        :param new_realizations: {('name', position) : new_scalar_value}
        :return:
        """
        ### TODO : Check if it is possible / usefull to have multiple variables sampled

        # Updates the attributes of the MCMC_toolbox

        ### TODO : Probably convert all the variables to torch tensors
        values = {
            'p0': realizations['p0'].tensor_realizations.detach().numpy(),
            'v0': realizations['v0'].tensor_realizations.detach().numpy(),
            'betas': realizations['betas'].tensor_realizations.detach().numpy()
        }

        self.MCMC_toolbox['attributes'].update([name_of_the_variable_that_has_been_changed],
                                               values)




    def load_parameters(self, parameters):
        for k in self.parameters.keys():
                self.parameters[k] = parameters[k]


    def initialize_parameters(self, data, smart_initialization):

        # Initializes Dimension
        self.dimension = data.dimension

        # TODO change it
        self.source_dimension = 2

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

            # V0 array minimum value
            v0_array[dim] = max(v0_array[dim], 0.05)

        SMART_INITIALIZATION = {
            'p0': p0_array,
            'v0' : v0_array,
            'betas': np.zeros(shape=(self.dimension-1, self.source_dimension)),
            'mean_tau': df.index.values.mean(), 'sigma_tau': 1.0,
            'mean_xi': 0., 'sigma_xi': 0.5,
            'mean_sources' : 0.0, 'sigma_sources' : 1.0,
            'sigma_noise': 0.1
        }

        # Initializes Parameters
        for parameter_key in self.parameters.keys():
            if self.parameters[parameter_key] is None:
                self.parameters[parameter_key] = SMART_INITIALIZATION[parameter_key]

        self.is_initialized = True
        self.attributes = Attributes_Multivariate(self.dimension, self.source_dimension)



    ###########################
    ## Getters/Setters
    ###########################

    def get_pop_shapes(self):
        p0_shape = (1, self.dimension)
        v0_shape = (1, self.dimension)
        beta_shape = (self.dimension-1, self.source_dimension)

        return {"p0": p0_shape,
                "v0": v0_shape,
                "beta": beta_shape}


    def random_variable_informations(self):

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

            betas_infos = {
                "name": "betas",
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
                "betas": betas_infos,
                "tau": tau_infos,
                "xi": xi_infos,
                "sources": sources_infos
            }

            return variables_infos

    def update_variable_info(self, key, realizations):
        """
        If either v0 or beta is changed, update the a matrix
        :param key:
        :param reals_pop:
        :return:
        """
        if key in ['v0', 'beta']:
            self.update_a_matrix(realizations['beta'].tensor_realizations, realizations['v0'].tensor_realizations)

    ###########################
    ## Core
    ###########################

    def compute_individual(self, individual, reals_pop, real_ind):
        # Load from dict
        v0 = reals_pop['v0']
        p0 = reals_pop['p0']

        # TODO : cache these variables ???
        g = torch.pow(p0, -1) - 1
        b = torch.pow(p0, 2) * g

        # Time reparametrized
        a = v0*torch.exp(real_ind['xi'])*(individual.tensor_timepoints-real_ind['tau'])

        # Add sources
        a += torch.mm(self.a_matrix, real_ind['sources'].t()).t()

        # Compute parallel curve
        parallel_curve = torch.pow(1 + g * torch.exp(-a / b), -1)

        return parallel_curve


    def compute_individual_tensorized(self, data, realizations):

        # Load from dict
        v0 = realizations['v0'].tensor_realizations
        p0 = realizations['p0'].tensor_realizations

        # TODO : cache these variables ???
        g = torch.pow(p0, -1) - 1
        b = torch.pow(p0, 2) * g

        # TODO change timepoints dimension in data structure
        timepoints = data.timepoints
        timepoints = timepoints.reshape(timepoints.shape[0],
                                        timepoints.shape[1],
                                        1)

        a_matrix = self.MCMC_toolbox['attributes'].mixing_matrix
        wi = torch.nn.functional.linear(realizations['sources'].tensor_realizations, a_matrix, bias=None)

        # Time reparametrized
        a = v0 * torch.exp(realizations['xi'].tensor_realizations) * (timepoints - realizations['tau'].tensor_realizations)

        # Add sources
        a += wi

        # Compute parallel curve
        parallel_curve = torch.pow(1 + g * torch.exp(-a / b), -1)

        return parallel_curve

    def compute_individual_attachment_tensorized(self, data,
                                      realizations):

        squared_sum = self.compute_sum_squared_tensorized(data,
                                     realizations)

        noise_variance = self.parameters['sigma_noise']**2

        individual_attachments = 0.5 * (1/noise_variance) * squared_sum
        individual_attachments += np.log(np.sqrt(2 * np.pi * noise_variance))


        #individual_attachments = 0.5 * self.cache_variables['noise_inverse'] * squared_sum
        #individual_attachments += self.cache_variables['constant_fit_variable']
        return individual_attachments

    def update_model(self, data, sufficient_statistics):

        #TODO parameters, automatic initialization of these parameters
        m_xi = data.n_individuals/20
        sigma2_xi_0 = 0.05

        m_tau = data.n_individuals/20
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
        self.model_parameters['noise_var'] = sufficient_statistics['sum_squared']/(data.n_observations)

        # Update the Random Variables
        self._update_random_variables()

        # Update Cached Variables
        self.cache_variables['noise_inverse'] = 1/self.model_parameters['noise_var']
        self.cache_variables['constant_fit_variable'] = np.log(np.sqrt(2 * np.pi * self.model_parameters['noise_var']))



    def update_model_parameters(self, data, sufficient_statistics, burn_in_phase=True):

        ## Probably optimize some common parts

        # Memoryless part of the algorithm
        if burn_in_phase:
            self.parameters['p0'] = sufficient_statistics['p0'].tensor_realizations.detach().numpy()
            self.parameters['v0'] = sufficient_statistics['v0'].tensor_realizations.detach().numpy()
            self.parameters['betas'] = sufficient_statistics['betas'].tensor_realizations.detach().numpy()
            xi = sufficient_statistics['xi'].tensor_realizations.detach().numpy()
            self.parameters['mean_xi'] = np.mean(xi)
            self.parameters['sigma_xi'] = np.std(xi)
            tau = sufficient_statistics['tau'].tensor_realizations.detach().numpy()
            self.parameters['mean_tau'] = np.mean(tau)
            self.parameters['sigma_tau'] = np.std(tau)

            squared_diff = self.compute_sum_squared_tensorized(data, sufficient_statistics).sum().numpy()
            self.parameters['sigma_noise'] = np.sqrt(squared_diff/(data.n_visits*data.dimension))



    ###########################
    ## In / Out
    ###########################


    def save_parameters(self, path):
        # TODO later
        return 0

        #TODO, shouldnt be this be in the output manager ???
        #TODO check que c'est le bon format (IGOR)
        model_settings = {}

        model_settings['parameters'] = self.parameters
        model_settings['dimension'] = self.dimension
        model_settings['source_dimension'] = self.dimension
        model_settings['type'] = self.model_name

        if type(model_settings['parameters']['p0']) not in [list]:
            model_settings['parameters']['p0'] = model_settings['parameters']['p0'].tolist()

        if type(model_settings['parameters']['v0']) not in [list]:
            model_settings['parameters']['v0'] = model_settings['parameters']['v0'].tolist()

        if type(model_settings['parameters']['betas']) not in [list]:
            model_settings['parameters']['betas'] = model_settings['parameters']['betas'].tolist()
            #beta = model_settings['parameters']['beta']
            #beta_n_columns = beta.shape[1]
            #model_settings['parameters'].pop('beta')
            # Save per column
            #for beta_dim in range(beta_n_columns):
            #    beta_column = beta[beta_dim].tolist()
            #    model_settings['parameters']['beta_'+ str(beta_dim)] = beta_column

        with open(path, 'w') as fp:
            json.dump(model_settings, fp)




    def update_Q_matrix(self, real_v0):
        self.Q_matrix = torch.tensor(self.householder(real_v0)).type(torch.FloatTensor)

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

    def householder(self, real_v0):
        s = real_v0.detach().numpy()
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


    def compute_average(self, tensor_timepoints):
        p0 = torch.Tensor(self.model_parameters['p0'])
        v0 = torch.Tensor(self.model_parameters['v0'])
        reparametrized_time = v0*np.exp(self.model_parameters['xi_mean'])*(tensor_timepoints-self.model_parameters['tau_mean'])
        return torch.pow(1 + (1 / p0 - 1) * torch.exp(-reparametrized_time / (p0 * (1 - p0))), -1)

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
        p0 = realizations['p0'].tensor_realizations.detach().numpy()

        # V0
        v0 = realizations['v0'].tensor_realizations.detach().numpy()

        # Beta
        beta = realizations['beta'].tensor_realizations.detach().numpy()

        # Compute sufficient statistics
        sufficient_statistics = {}
        sufficient_statistics['p0'] = p0
        sufficient_statistics['v0'] = v0
        sufficient_statistics['beta'] = beta
        sufficient_statistics['tau_mean'] = tau_mean
        sufficient_statistics['tau_var'] = tau_var
        sufficient_statistics['xi_mean'] = xi_mean
        sufficient_statistics['xi_var'] = xi_var
        sufficient_statistics['sum_squared'] = float(torch.sum(self.compute_sum_squared_tensorized(data, realizations).detach()).numpy())
        sufficient_statistics['empirical_sources_var'] = sources_var

        # TODO : non identifiable here with the xi, but how do we update each xi ?
        sufficient_statistics['v0'][sufficient_statistics['v0'] < 0.01] = 0.01
        self.model_parameters['v0'] = np.exp(sufficient_statistics['xi_mean']) * np.array(self.model_parameters['v0'])
        realizations['xi'].tensor_realizations = realizations['xi'].tensor_realizations-sufficient_statistics['xi_mean']
        sufficient_statistics['xi_mean'] = sufficient_statistics['xi_mean']

        return sufficient_statistics
