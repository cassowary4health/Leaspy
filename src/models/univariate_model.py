import os

from src import default_data_dir
from src.models.abstract_model import AbstractModel
from src.inputs.model_settings import ModelSettings
import torch
from torch.autograd import Variable
import numpy as np
import json
from scipy.optimize import minimize



#@torch.jit.script
#def compute_individual_torch(timepoints, p0, xi, tau):
#    reparametrized_time = torch.exp(xi) * (timepoints - tau)
#    return torch.pow(1 + (1 / p0 - 1) * torch.exp(-reparametrized_time / (p0 * (1 - p0))), -1)

class UnivariateModel(AbstractModel):
    def __init__(self):
        data_dir = os.path.join(default_data_dir, "default_univariate_parameters.json")
        reader = ModelSettings(data_dir)
        self.model_parameters = reader.parameters

        if reader.model_type != 'univariate':
            raise ValueError("The default univariate parameters are not of univariate type")


        self.reals_pop_name = ['p0']
        self.reals_ind_name = ['xi','tau']


        self.model_name = 'univariate'

        ###########################
        ## Initialization
        ###########################

    def get_pop_shapes(self):
        p0_shape = (1, 1)
        return {"p0": p0_shape}

    def get_info_variables(self, data):

            n_individuals = data.n_individuals

            p0_infos = {
                "name": "p0",
                "shape": (1, 1),
                "type": "population",
                "rv_type": "multigaussian"
            }

            tau_infos = {
                "name": "tau",
                "shape": (n_individuals, 1),
                "type": "individual",
                "rv_type": "gaussian"
            }

            xi_infos = {
                "name": "xi",
                "shape": (n_individuals, 1),
                "type": "individual",
                "rv_type": "gaussian"
            }

            variables_infos = {
                "p0" : p0_infos,
                "tau" : tau_infos,
                "xi" : xi_infos
            }

            return variables_infos


    ###########################
    ## Core
    ###########################


    def compute_individual(self, individual, reals_pop, real_ind):
        p0 = reals_pop['p0']
        reparametrized_time = torch.exp(real_ind['xi'])*(individual.tensor_timepoints-real_ind['tau'])
        return torch.pow(1+(1/p0-1)*torch.exp(-reparametrized_time/(p0*(1-p0))), -1)
        #return compute_individual_torch(individual.tensor_timepoints,
        #                                    reals_pop['p0'],
        #                                     real_ind['xi'],
        #                                     real_ind['tau'])




    def compute_average(self, tensor_timepoints):
        p0 = torch.Tensor(self.model_parameters['p0'])
        # TODO better
        #p0 = p0[0][0]
        reparametrized_time = np.exp(self.model_parameters['xi_mean'])*(tensor_timepoints.reshape(-1,1)-self.model_parameters['tau_mean'])
        return torch.pow(1 + (1 / p0 - 1) * torch.exp(-reparametrized_time / (p0 * (1 - p0))), -1)
        #return compute_individual_torch(tensor_timepoints,
        #                         torch.tensor(self.model_parameters['p0']),
        #                         torch.tensor([0]),
        #                         torch.tensor([0]))

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

        sufficient_statistics = {}
        sufficient_statistics['p0'] = p0
        sufficient_statistics['tau_mean'] = tau_mean
        sufficient_statistics['tau_var'] = tau_var
        sufficient_statistics['xi_mean'] = xi_mean
        sufficient_statistics['xi_var'] = xi_var
        sufficient_statistics['sum_squared'] = float(self.compute_sumsquared(data, reals_pop, reals_ind).detach().numpy())

        return sufficient_statistics



    def update_model(self, data, sufficient_statistics):

        #TODO parameters, automatic initialization of these parameters
        m_xi = data.n_individuals / 20
        sigma2_xi_0 = 1.0

        m_tau = data.n_individuals / 20
        sigma2_tau_0 = 1.0


        self.model_parameters['p0'] = sufficient_statistics['p0']

        # Tau
        self.model_parameters['tau_mean'] = sufficient_statistics['tau_mean']
        tau_var_update = (1/(data.n_individuals+m_tau))*(data.n_individuals * sufficient_statistics['tau_var']+m_tau*sigma2_tau_0)
        self.model_parameters['tau_var'] = tau_var_update

        # Xi
        self.model_parameters['xi_mean'] = sufficient_statistics['xi_mean']
        tau_var_update = (1/(data.n_individuals+m_xi))*(data.n_individuals * sufficient_statistics['xi_var']+m_tau*sigma2_xi_0)
        self.model_parameters['xi_var'] = tau_var_update

        # Noise
        self.model_parameters['noise_var'] = sufficient_statistics['sum_squared']/data.n_observations

        # Update the Random Variables
        self._update_random_variables()

        # Update Cached Variables
        self.cache_variables['noise_inverse'] = 1/self.model_parameters['noise_var']
        self.cache_variables['constant_fit_variable'] = np.log(np.sqrt(2 * np.pi * self.model_parameters['noise_var']))


    def smart_initialization(self, data):
        """
        Assigns dimension + model_parameters

        model parameters from the data
        :param data:
        :return:
        """

        # Initializes Dimension
        self.dimension = data.dimension

        # Find a P0
        p0 = 0
        for indices in data.indices:
            p0 += data[indices].tensor_observations.mean()
        p0 /= data.n_individuals
        p0 = p0.detach().numpy()

        """

        # Optimize for alpha/tau
        # TODO Torch/float/numpy
        def cost_function(x, *args):
            xi, tau = x
            individual, p0 = args
            reals_pop_dummy = {'p0': p0}
            real_ind_dummy = {'xi': torch.Tensor([xi]), 'tau':tau}
            squared_diff = torch.sum((self.compute_individual(individual, reals_pop_dummy, real_ind_dummy)-individual.tensor_timepoints)**2)
            squared_diff = float(squared_diff.detach().numpy())

            return squared_diff

        results = []

        for idx in data.indices:
            res = minimize(cost_function, x0=(-2.0, 0.0),
                           args = (data[idx], p0),
                           method='Powell',
                           options={'xtol': 1e-15, 'disp': True})

            if res.success and res.x[0] > -4 and res.x[0] < 4:
                results.append(res.x)
            else:
                print(res.x, data[idx].tensor_observations)

        xi_mean, tau_mean = np.mean(results, axis=0)
        xi_var, tau_var = np.var(results, axis=0)/10"""

        # Pre-Initialize from dimension
        SMART_INITIALIZATION = {
            'p0': p0.reshape(1,1), 'tau_mean': 0.0, 'tau_var': 1.0,
            'xi_mean': -1.0, 'xi_var': 0.1, 'noise_var': 0.5
        }

        # Initializes Parameters
        for parameter_key in self.model_parameters.keys():
            if self.model_parameters[parameter_key] is None:
                print("Changing value of {0} from {1} to {2}".format(parameter_key, "None", SMART_INITIALIZATION[parameter_key]))
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


        with open(path, 'w') as fp:
            json.dump(model_settings, fp)
