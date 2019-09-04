import numpy as np
import torch
import json

from leaspy.utils.realizations.realization import Realization

from .abstract_multivariate_model import AbstractMultivariateModel
from .utils.attributes.attributes_logistic_parallel import Attributes_LogisticParallel
from .utils.initialization.initialization_logistic import initialize_logistic_parallel

class LogisticParallelModel(AbstractMultivariateModel):
    ###############
    #INITITALISATION
    ###############

    def __init__(self, name):
        super(LogisticParallelModel, self).__init__(name)
        self.parameters["deltas"] = None
        self.MCMC_toolbox['priors']['deltas_std'] = None

    def load_parameters(self, parameters):
        self.parameters = {}
        for k in parameters.keys():
            self.parameters[k] = torch.tensor(parameters[k])
        self.attributes = Attributes_LogisticParallel(self.dimension, self.source_dimension)
        self.attributes.update(['all'],self.parameters)



    def initialize(self, data):
        # TODO check why yhis changes tests in personalize ? seed ? or other ?
        self = initialize_logistic_parallel(self, data)

        """
        self.dimension = data.dimension
        if self.source_dimension is None:
            self.source_dimension = int(data.dimension/2.)

        # "Smart" initialization : may be improved
        # TODO !
        self.parameters = {
            'g': torch.tensor([1.]), 'tau_mean': 70.0, 'tau_std': 2.0, 'xi_mean': -3., 'xi_std': 0.1,
            'sources_mean': 0.0, 'sources_std': 1.0,
            'noise_std': 0.1, 'deltas': torch.tensor([0.0]*(self.dimension-1)),
            'betas': torch.zeros((self.dimension - 1, self.source_dimension))
        }
        self.attributes = Attributes_LogisticParallel(self.dimension, self.source_dimension)
        self.is_initialized = True"""

    def initialize_MCMC_toolbox(self):
        self.MCMC_toolbox = {
            'priors': {'g_std': 1., 'deltas_std': 0.1, 'betas_std': 0.1 },
            'attributes': Attributes_LogisticParallel(self.dimension, self.source_dimension)
        }

        # TODO we created an ad-hoc dictionnary
        pop_dictionnary = {}
        for name_variable, info_variable in self.random_variable_informations().items():
            if info_variable['type'] == "population":
                pop_dictionnary[name_variable] = Realization.from_tensor(name_variable,
                                                                         info_variable['shape'],
                                                                         info_variable['type'],
                                                                         self.parameters[name_variable])

        self.update_MCMC_toolbox(["all"], pop_dictionnary)


    ############
    #CORE
    ############

    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        L = name_of_the_variables_that_have_been_changed
        values = {}
        if any(c in L for c in ('g', 'all')):
            values['g'] = realizations['g'].tensor_realizations
        if any(c in L for c in ('deltas', 'all')):
            values['deltas'] = realizations['deltas'].tensor_realizations
        if any(c in L for c in ('betas', 'all')) and self.source_dimension != 0:
            values['betas'] = realizations['betas'].tensor_realizations
        if any(c in L for c in ('xi_mean', 'all')):
            values['xi_mean'] = self.parameters['xi_mean']

        self.MCMC_toolbox['attributes'].update(L, values)

    def _get_attributes(self,MCMC):
        if MCMC:
            g = self.MCMC_toolbox['attributes'].g
            deltas = self.MCMC_toolbox['attributes'].deltas
            a_matrix = self.MCMC_toolbox['attributes'].mixing_matrix
        else:
            g = self.attributes.g
            deltas = self.attributes.deltas
            a_matrix = self.attributes.mixing_matrix
        return g, deltas, a_matrix



    def compute_individual_tensorized(self, timepoints, ind_parameters, MCMC=False):



        # Population parameters
        g, deltas, a_matrix = self._get_attributes(MCMC)
        deltas_exp = torch.exp(-deltas)

        # Individual parameters
        xi, tau, sources = ind_parameters
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        #print(xi.shape, tau.shape, sources.shape, timepoints.shape)

        # Log likelihood computation
        LL = deltas.unsqueeze(0).repeat(timepoints.shape[0], 1)
        if self.source_dimension != 0:
            wi = torch.nn.functional.linear(sources, a_matrix, bias=None)
            LL += wi * (g * deltas_exp + 1) ** 2 / (g * deltas_exp)
        LL = -reparametrized_time.unsqueeze(-1) - LL.unsqueeze(-2)
        model = 1. / (1. + g*torch.exp(LL))

        #print(model.shape)
        return model


    def compute_sufficient_statistics(self, data, realizations):
        sufficient_statistics = {}
        sufficient_statistics['g'] = realizations['g'].tensor_realizations.detach()[0]
        sufficient_statistics['deltas'] = realizations['deltas'].tensor_realizations.detach()
        if self.source_dimension != 0:
            sufficient_statistics['betas'] = realizations['betas'].tensor_realizations.detach()
        sufficient_statistics['tau'] = realizations['tau'].tensor_realizations
        sufficient_statistics['tau_sqrd'] = torch.pow(realizations['tau'].tensor_realizations, 2)
        sufficient_statistics['xi'] = realizations['xi'].tensor_realizations
        sufficient_statistics['xi_sqrd'] = torch.pow(realizations['xi'].tensor_realizations, 2)

        #TODO : Optimize to compute the matrix multiplication only once for the reconstruction
        xi, tau, sources = self.get_param_from_real(realizations)
        data_reconstruction = self.compute_individual_tensorized(data.timepoints, (xi,tau,sources),MCMC=True)
        data_reconstruction *= data.mask
        norm_0 = data.values * data.values * data.mask
        norm_1 = data.values * data_reconstruction * data.mask
        norm_2 = data_reconstruction * data_reconstruction * data.mask
        sufficient_statistics['obs_x_obs'] = torch.sum(norm_0, dim=2)
        sufficient_statistics['obs_x_reconstruction'] = torch.sum(norm_1, dim=2)
        sufficient_statistics['reconstruction_x_reconstruction'] = torch.sum(norm_2, dim=2)

        return sufficient_statistics


    def update_model_parameters_burn_in(self, data, realizations):
        self.parameters['g'] = realizations['g'].tensor_realizations.detach()[0]
        self.parameters['deltas'] = realizations['deltas'].tensor_realizations.detach()
        if self.source_dimension != 0:
            self.parameters['betas'] = realizations['betas'].tensor_realizations.detach()
        xi = realizations['xi'].tensor_realizations.detach()
        self.parameters['xi_mean'] = torch.mean(xi)
        self.parameters['xi_std'] = torch.std(xi)
        tau = realizations['tau'].tensor_realizations.detach()
        self.parameters['tau_mean'] = torch.mean(tau)
        self.parameters['tau_std'] = torch.std(tau)

        param_ind = self.get_param_from_real(realizations)
        data_fit = self.compute_individual_tensorized(data.timepoints, param_ind, MCMC=True)
        data_fit *= data.mask
        squared_diff = ((data_fit - data.values) ** 2).sum()
        squared_diff = squared_diff.detach()  # Remove the gradients
        self.parameters['noise_std'] = torch.sqrt(squared_diff / (data.n_visits * data.dimension))

    def update_model_parameters_normal(self, data, suff_stats):

        self.parameters['g'] = suff_stats['g']
        self.parameters['deltas'] = suff_stats['deltas']
        if self.source_dimension != 0:
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
            "shape": torch.Size([1]),
            "type": "population",
            "rv_type": "multigaussian"
        }
        deltas_infos = {
            "name": "deltas",
            "shape": torch.Size([self.dimension-1]),
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
            "deltas": deltas_infos,
            "tau": tau_infos,
            "xi": xi_infos,
        }

        if self.source_dimension != 0:
            variables_infos['sources'] = sources_infos
            variables_infos['betas'] = betas_infos

        return variables_infos



######### NOT USED YET

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