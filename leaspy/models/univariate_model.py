import json

import torch

from leaspy import __version__

from leaspy.models.abstract_model import AbstractModel
from leaspy.models.utils.attributes import AttributesFactory
from leaspy.models.utils.initialization.model_initialization import initialize_parameters
from leaspy.utils.docs import doc_with_super, doc_with_

# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either model values or model values + jacobians wrt individual parameters
# TODO refact? subclass or other proper code technique to extract model's concrete formulation depending on if linear, logistic, mixed log-lin, ...

@doc_with_super()
class UnivariateModel(AbstractModel):
    """
    Univariate (logistic or linear) model for a single variable of interest.
    """
    def __init__(self, name, **kwargs):
        super().__init__(name)
        self.dimension = 1
        self.source_dimension = 0  # TODO, None ???
        self.parameters = {
            "g": None,
            "tau_mean": None, "tau_std": None,
            "xi_mean": None, "xi_std": None,
            "noise_std": None
        }
        self.bayesian_priors = None
        self.attributes = None

        # MCMC related "parameters"
        self.MCMC_toolbox = {
            'attributes': None,
            'priors': {
                # for logistic: "p0" = 1 / (1+exp(g)) i.e. exp(g) = 1/p0 - 1
                # for linear: "p0" = g
                'g_std': None,
            }
        }

        # load hyperparameters
        self.load_hyperparameters(kwargs)

    def save(self, path, **kwargs):

        model_parameters_save = self.parameters.copy()
        for key, value in model_parameters_save.items():
            if type(value) in [torch.Tensor]:
                model_parameters_save[key] = value.tolist()
        model_settings = {
            'leaspy_version': __version__,
            'name': self.name,
            'features': self.features,
            #'dimension': 1,
            'loss': self.loss,
            'parameters': model_parameters_save
        }
        # TODO : in leaspy models there should be a method to only return the dict describing the model
        # and then another generic method (inherited) should save this dict
        # (with extra standard fields such as 'leaspy_version' for instance)
        with open(path, 'w') as fp:
            json.dump(model_settings, fp, **kwargs)

    def load_hyperparameters(self, hyperparameters):

        if 'features' in hyperparameters.keys():
            self.features = hyperparameters['features']
        if 'loss' in hyperparameters.keys():
            self.loss = hyperparameters['loss']
        if any([key not in ('features', 'loss') for key in hyperparameters.keys()]):
            raise ValueError("Only <features> and <loss> are valid hyperparameters for an UnivariateModel!"
                             f"You gave {hyperparameters}.")

    def initialize(self, dataset, method="default"):

        self.features = dataset.headers

        self.parameters = initialize_parameters(self, dataset, method)

        self.attributes = AttributesFactory.attributes(self.name, dimension=1)
        self.attributes.update(['all'], self.parameters)

        self.is_initialized = True

    def load_parameters(self, parameters):
        self.parameters = {}
        for k in parameters.keys():
            self.parameters[k] = torch.tensor(parameters[k], dtype=torch.float32)
        self.attributes = AttributesFactory.attributes(self.name, dimension=1)
        self.attributes.update(['all'], self.parameters)

    def initialize_MCMC_toolbox(self):
        """
        Initialize Monte-Carlo Markov-Chain toolbox for calibration of model

        TODO to move in a "MCMC-model interface"
        """
        self.MCMC_toolbox = {
            'priors': {'g_std': 0.01}, # population parameter
            'attributes': AttributesFactory.attributes(self.name, dimension=1)
        }

        population_dictionary = self._create_dictionary_of_population_realizations()
        self.update_MCMC_toolbox(["all"], population_dictionary)

    ##########
    # CORE
    ##########
    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        """
        Update the MCMC toolbox with a collection of realizations of model population parameters.

        TODO to move in a "MCMC-model interface"

        Parameters
        ----------
        name_of_the_variables_that_have_been_changed: container[str] (list, tuple, ...)
            Names of the population parameters to update in MCMC toolbox
        realizations : :class:`.CollectionRealization`
            All the realizations to update MCMC toolbox with
        """
        L = name_of_the_variables_that_have_been_changed
        values = {}
        if any(c in L for c in ('g', 'all')):
            values['g'] = realizations['g'].tensor_realizations
        if any(c in L for c in ('xi_mean', 'all')):
            values['xi_mean'] = self.parameters['xi_mean']

        self.MCMC_toolbox['attributes'].update(L, values)

    def _get_attributes(self, MCMC):
        if MCMC:
            g = self.MCMC_toolbox['attributes'].positions
        else:
            g = self.attributes.positions
        return g

    def compute_mean_traj(self, timepoints):
        """
        Compute trajectory of the model with individual parameters being the group-average ones.

        TODO check dimensions of io?
        TODO generalize in abstract manifold model

        Parameters
        ----------
        timepoints : :class:`torch.Tensor` [1, n_timepoints]

        Returns
        -------
        :class:`torch.Tensor` [1, n_timepoints, dimension]
            The group-average values at given timepoints
        """
        individual_parameters = {
            'xi': torch.tensor([self.parameters['xi_mean']], dtype=torch.float32),
            'tau': torch.tensor([self.parameters['tau_mean']], dtype=torch.float32),
        }

        return self.compute_individual_tensorized(timepoints, individual_parameters)

    def compute_individual_tensorized(self, timepoints, ind_parameters, attribute_type=None):
        if self.name == 'univariate_logistic':
            return self.compute_individual_tensorized_logistic(timepoints, ind_parameters, attribute_type)
        elif self.name == 'univariate_linear':
            return self.compute_individual_tensorized_linear(timepoints, ind_parameters, attribute_type)
        else:
            raise ValueError(f"UnivariateModel: only `univariate_linear` and `univariate_logistic` are supported. You gave {self.name}")

    def compute_individual_tensorized_logistic(self, timepoints, ind_parameters, attribute_type=False):

        # Population parameters
        g = self._get_attributes(attribute_type)
        # Individual parameters
        xi, tau = ind_parameters['xi'], ind_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        LL = -reparametrized_time.unsqueeze(-1)
        model = 1. / (1. + g * torch.exp(LL))

        return model

    def compute_individual_tensorized_linear(self, timepoints, ind_parameters, attribute_type=False):

        # Population parameters
        positions = self._get_attributes(attribute_type)
        # Individual parameters
        xi, tau = ind_parameters['xi'], ind_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)
        LL = -reparametrized_time.unsqueeze(-1)
        model = positions - LL

        return model

    def compute_jacobian_tensorized(self, timepoints, ind_parameters, attribute_type=None):
        if self.name in ['logistic', 'univariate_logistic']:
            return self.compute_jacobian_tensorized_logistic(timepoints, ind_parameters, attribute_type)
        elif self.name in ['linear', 'univariate_linear']:
            return self.compute_jacobian_tensorized_linear(timepoints, ind_parameters, attribute_type)
        else:
            raise ValueError(f"UnivariateModel: only `univariate_linear` and `univariate_logistic` are supported. You gave {self.name}")

    def compute_jacobian_tensorized_linear(self, timepoints, ind_parameters, attribute_type=None):

        # Population parameters
        positions = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = ind_parameters['xi'], ind_parameters['tau']

        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Log likelihood computation
        reparametrized_time = reparametrized_time.unsqueeze(-1)

        LL = reparametrized_time + positions

        alpha = torch.exp(xi).unsqueeze(-1)

        derivatives = {
            'xi' : (reparametrized_time).unsqueeze(-1),
            'tau' : (-alpha * torch.ones_like(reparametrized_time)).unsqueeze(-1),
        }
        return derivatives

    def compute_jacobian_tensorized_logistic(self, timepoints, ind_parameters, MCMC=False):

        # Population parameters
        g = self._get_attributes(MCMC)

        # Individual parameters
        xi, tau = ind_parameters['xi'], ind_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Log likelihood computation
        reparametrized_time = reparametrized_time.unsqueeze(-1) # (n_individuals, n_timepoints, n_features==1)

        model = 1. / (1. + g * torch.exp(-reparametrized_time))

        c = model * (1. - model)
        alpha = torch.exp(xi).reshape(-1, 1, 1)

        derivatives = {
            'xi': (c * reparametrized_time).unsqueeze(-1),
            'tau': (c * -alpha).unsqueeze(-1),
        }

        # dict[param_name: str, torch.Tensor of shape(n_ind, n_tpts, n_fts, n_dims_param)]
        return derivatives

    def compute_sufficient_statistics(self, data, realizations):
        sufficient_statistics = {}
        sufficient_statistics['g'] = realizations['g'].tensor_realizations.detach() # avoid 0D / 1D tensors mix
        sufficient_statistics['tau'] = realizations['tau'].tensor_realizations
        sufficient_statistics['tau_sqrd'] = torch.pow(realizations['tau'].tensor_realizations, 2)
        sufficient_statistics['xi'] = realizations['xi'].tensor_realizations
        sufficient_statistics['xi_sqrd'] = torch.pow(realizations['xi'].tensor_realizations, 2)

        # TODO : Optimize to compute the matrix multiplication only once for the reconstruction
        ind_parameters = self.get_param_from_real(realizations)
        data_reconstruction = self.compute_individual_tensorized(data.timepoints, ind_parameters, attribute_type=True)

        data_reconstruction *= data.mask.float() # speed-up computations
        #norm_0 = data.values * data.values * data.mask.float()
        norm_1 = data.values * data_reconstruction #* data.mask.float()
        norm_2 = data_reconstruction * data_reconstruction #* data.mask.float()
        #sufficient_statistics['obs_x_obs'] = torch.sum(norm_0, dim=2)
        sufficient_statistics['obs_x_reconstruction'] = norm_1 #.sum(dim=2)
        sufficient_statistics['reconstruction_x_reconstruction'] = norm_2 #.sum(dim=2)

        if self.loss == 'crossentropy':
            sufficient_statistics['crossentropy'] = self.compute_individual_attachment_tensorized(data, ind_parameters, attribute_type=True)

        return sufficient_statistics

    def update_model_parameters_burn_in(self, data, realizations):

        # Memoryless part of the algorithm
        self.parameters['g'] = realizations['g'].tensor_realizations.detach()
        xi = realizations['xi'].tensor_realizations.detach()
        self.parameters['xi_mean'] = torch.mean(xi)
        self.parameters['xi_std'] = torch.std(xi)
        tau = realizations['tau'].tensor_realizations.detach()
        self.parameters['tau_mean'] = torch.mean(tau)
        self.parameters['tau_std'] = torch.std(tau)

        param_ind = self.get_param_from_real(realizations)
        squared_diff = self.compute_sum_squared_tensorized(data, param_ind, attribute_type=True).sum()
        self.parameters['noise_std'] = torch.sqrt(squared_diff / data.n_observations)

        if self.loss == 'crossentropy':
            crossentropy = self.compute_individual_attachment_tensorized(data, param_ind, attribute_type=True).sum()
            self.parameters['crossentropy'] = crossentropy
        # Stochastic sufficient statistics used to update the parameters of the model

    def update_model_parameters_normal(self, data, suff_stats):
        self.parameters['g'] = suff_stats['g']

        tau_mean = self.parameters['tau_mean']
        tau_std_updt = torch.mean(suff_stats['tau_sqrd']) - 2 * tau_mean * torch.mean(suff_stats['tau'])
        self.parameters['tau_std'] = torch.sqrt(tau_std_updt + self.parameters['tau_mean'] ** 2)
        self.parameters['tau_mean'] = torch.mean(suff_stats['tau'])

        xi_mean = self.parameters['xi_mean']
        xi_std_updt = torch.mean(suff_stats['xi_sqrd']) - 2 * xi_mean * torch.mean(suff_stats['xi'])
        self.parameters['xi_std'] = torch.sqrt(xi_std_updt + self.parameters['xi_mean'] ** 2)
        self.parameters['xi_mean'] = torch.mean(suff_stats['xi'])

        #S1 = torch.sum(suff_stats['obs_x_obs'])
        S1 = data.L2_norm
        S2 = suff_stats['obs_x_reconstruction'].sum()
        S3 = suff_stats['reconstruction_x_reconstruction'].sum()

        self.parameters['noise_std'] = torch.sqrt((S1 - 2. * S2 + S3) / data.n_observations)

        if self.loss == 'crossentropy':
            self.parameters['crossentropy'] = suff_stats['crossentropy'].sum()


    def random_variable_informations(self):

        ## Population variables
        g_infos = {
            "name": "g",
            "shape": torch.Size([1]),
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

        variables_infos = {
            "g": g_infos,
            "tau": tau_infos,
            "xi": xi_infos,
        }

        return variables_infos

# document some methods (we cannot decorate them at method creation since they are not yet decorated from `doc_with_super`)
doc_with_(UnivariateModel.compute_individual_tensorized_linear, UnivariateModel.compute_individual_tensorized, mapping={'the model': 'the model (linear)'})
doc_with_(UnivariateModel.compute_individual_tensorized_logistic, UnivariateModel.compute_individual_tensorized, mapping={'the model': 'the model (logistic)'})

doc_with_(UnivariateModel.compute_jacobian_tensorized_linear, UnivariateModel.compute_jacobian_tensorized, mapping={'the model': 'the model (linear)'})
doc_with_(UnivariateModel.compute_jacobian_tensorized_logistic, UnivariateModel.compute_jacobian_tensorized, mapping={'the model': 'the model (logistic)'})
