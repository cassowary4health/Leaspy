import json

import torch

from leaspy import __version__

from leaspy.models.abstract_model import AbstractModel
from leaspy.models.utils.attributes import AttributesFactory
from leaspy.models.utils.initialization.model_initialization import initialize_parameters
from leaspy.models.utils.noise_model import NoiseModel

from leaspy.utils.typing import Optional
from leaspy.utils.docs import doc_with_super, doc_with_
from leaspy.utils.subtypes import suffixed_method
from leaspy.exceptions import LeaspyModelInputError

# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either model values or model values + jacobians wrt individual parameters
# TODO refact? subclass or other proper code technique to extract model's concrete formulation depending on if linear, logistic, mixed log-lin, ...


@doc_with_super()
class UnivariateModel(AbstractModel):
    """
    Univariate (logistic or linear) model for a single variable of interest.

    Parameters
    ----------
    name : str
        Name of the model
    **kwargs
        Hyperparameters of the model

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        * If `name` is not one of allowed sub-type: 'univariate_linear' or 'univariate_logistic'
        * If hyperparameters are inconsistent
    """

    SUBTYPES_SUFFIXES = {
        'univariate_linear': '_linear',
        'univariate_logistic': '_logistic'
    }

    def __init__(self, name: str, **kwargs):

        super().__init__(name)

        self.dimension = 1
        self.source_dimension = 0  # TODO, None ???
        self.noise_model = 'gaussian_scalar'

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

        # subtype of univariate model
        self._subtype_suffix = self._check_subtype()

        # Load hyperparameters at end to overwrite default for new hyperparameters
        self.load_hyperparameters(kwargs)

    def _check_subtype(self):
        if self.name not in self.SUBTYPES_SUFFIXES.keys():
            raise LeaspyModelInputError(f'Univariate model name should be among these valid sub-types: '
                                        f'{list(self.SUBTYPES_SUFFIXES.keys())}.')

        return self.SUBTYPES_SUFFIXES[self.name]

    def save(self, path: str, **kwargs):

        model_parameters_save = self.parameters.copy()
        for key, value in model_parameters_save.items():
            if isinstance(value, torch.Tensor):
                model_parameters_save[key] = value.tolist()
        model_settings = {
            'leaspy_version': __version__,
            'name': self.name,
            'features': self.features,
            #'dimension': 1,
            'noise_model': self.noise_model,
            'parameters': model_parameters_save
        }

        # TODO : in leaspy models there should be a method to only return the dict describing the model
        # and then another generic method (inherited) should save this dict
        # (with extra standard fields such as 'leaspy_version' for instance)

        # Default json.dump kwargs:
        kwargs = {'indent': 2, **kwargs}

        with open(path, 'w') as fp:
            json.dump(model_settings, fp, **kwargs)

    def load_hyperparameters(self, hyperparameters: dict):

        expected_hyperparameters = ('features',)
        if 'features' in hyperparameters.keys():
            self.features = hyperparameters['features']

        # load new `noise_model` directly in-place & add the recognized hyperparameters to known tuple
        # TODO? forbid the usage of `gaussian_diagonal` noise for such model?
        expected_hyperparameters += NoiseModel.set_noise_model_from_hyperparameters(self, hyperparameters)

        self._raise_if_unknown_hyperparameters(expected_hyperparameters, hyperparameters)

    def initialize(self, dataset, method="default"):

        self.features = dataset.headers

        self.parameters = initialize_parameters(self, dataset, method)

        self.attributes = AttributesFactory.attributes(self.name, dimension=1)

        # Postpone the computation of attributes when really needed!
        #self.attributes.update(['all'], self.parameters)

        self.is_initialized = True

    def load_parameters(self, parameters):
        self.parameters = {}
        for k in parameters.keys():
            self.parameters[k] = torch.tensor(parameters[k])

        # derive the model attributes from model parameters upon reloading of model
        self.attributes = AttributesFactory.attributes(self.name, dimension=1)
        self.attributes.update(['all'], self.parameters)

    def initialize_MCMC_toolbox(self):
        """
        Initialize Monte-Carlo Markov-Chain toolbox for calibration of model
        """
        # TODO to move in the MCMC-fit algorithm
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

        TODO to move in the MCMC-fit algorithm

        Parameters
        ----------
        name_of_the_variables_that_have_been_changed : container[str] (list, tuple, ...)
            Names of the population parameters to update in MCMC toolbox
        realizations : :class:`.CollectionRealization`
            All the realizations to update MCMC toolbox with
        """
        L = name_of_the_variables_that_have_been_changed
        values = {}
        if any(c in L for c in ('g', 'all')):
            values['g'] = realizations['g'].tensor_realizations

        self.MCMC_toolbox['attributes'].update(L, values)


    def _get_attributes(self, attribute_type: Optional[str]):
        if attribute_type is None:
            return self.attributes.get_attributes()
        elif attribute_type == 'MCMC':
            return self.MCMC_toolbox['attributes'].get_attributes()
        else:
            raise LeaspyModelInputError(f"The specified attribute type does not exist: {attribute_type}. "
                                        "Should be None or 'MCMC'.")

    def compute_mean_traj(self, timepoints, *, attribute_type: Optional[str] = None):
        """
        Compute trajectory of the model with individual parameters being the group-average ones.

        TODO check dimensions of io?
        TODO generalize in abstract manifold model

        Parameters
        ----------
        timepoints : :class:`torch.Tensor` [1, n_timepoints]
        attribute_type : 'MCMC' or None

        Returns
        -------
        :class:`torch.Tensor` [1, n_timepoints, dimension]
            The group-average values at given timepoints
        """
        individual_parameters = {
            'xi': torch.tensor([self.parameters['xi_mean']]),
            'tau': torch.tensor([self.parameters['tau_mean']]),
        }

        return self.compute_individual_tensorized(timepoints, individual_parameters, attribute_type=attribute_type)

    @suffixed_method
    def compute_individual_tensorized(self, timepoints, individual_parameters, *, attribute_type=None):
        pass

    def compute_individual_tensorized_logistic(self, timepoints, individual_parameters, *, attribute_type=None):

        # Population parameters
        g = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # TODO? more efficient & accurate to compute `torch.exp(-t + log_g)` since we directly sample & stored log_g
        t = reparametrized_time.unsqueeze(-1)
        model = 1. / (1. + g * torch.exp(-t))

        return model

    def compute_individual_tensorized_linear(self, timepoints, individual_parameters, *, attribute_type=None):

        # Population parameters
        positions = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        return positions + reparametrized_time.unsqueeze(-1)

    @suffixed_method
    def compute_individual_ages_from_biomarker_values_tensorized(self, value: torch.Tensor,
                                                                 individual_parameters: dict, feature: str):
        pass

    def compute_individual_ages_from_biomarker_values_tensorized_logistic(self, value: torch.Tensor,
                                                                          individual_parameters: dict, feature: str):
        # avoid division by zero:
        value = value.masked_fill((value == 0) | (value == 1), float('nan'))

        # get tensorized attributes
        g = self._get_attributes(None)
        xi, tau = individual_parameters['xi'], individual_parameters['tau']

        # compute age
        ages = torch.exp(-xi) * torch.log(g/(1 / value - 1)) + tau
        assert ages.shape == value.shape

        return ages

    @suffixed_method
    def compute_jacobian_tensorized(self, timepoints, individual_parameters, *, attribute_type=None):
        pass

    def compute_jacobian_tensorized_linear(self, timepoints, individual_parameters, *, attribute_type=None):

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1)
        alpha = torch.exp(xi).unsqueeze(-1)

        # Jacobian of model expected value w.r.t. individual parameters
        derivatives = {
            'xi': reparametrized_time.unsqueeze(-1),
            'tau': (-alpha * torch.ones_like(reparametrized_time)).unsqueeze(-1),
        }

        return derivatives

    def compute_jacobian_tensorized_logistic(self, timepoints, individual_parameters, *, attribute_type=None):

        # Population parameters
        g = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1) # (n_individuals, n_timepoints, n_features==1)
        alpha = torch.exp(xi).reshape(-1, 1, 1)

        # Jacobian of model expected value w.r.t. individual parameters
        model = 1. / (1. + g * torch.exp(-reparametrized_time))
        c = model * (1. - model)

        derivatives = {
            'xi': (c * reparametrized_time).unsqueeze(-1),
            'tau': (c * -alpha).unsqueeze(-1),
        }

        # dict[param_name: str, torch.Tensor of shape(n_ind, n_tpts, n_fts, n_dims_param)]
        return derivatives

    def compute_sufficient_statistics(self, data, realizations):

        # unlink all sufficient statistics from updates in realizations!
        realizations = realizations.clone_realizations()

        sufficient_statistics = {}
        sufficient_statistics['g'] = realizations['g'].tensor_realizations
        sufficient_statistics['tau'] = realizations['tau'].tensor_realizations
        sufficient_statistics['tau_sqrd'] = torch.pow(realizations['tau'].tensor_realizations, 2)
        sufficient_statistics['xi'] = realizations['xi'].tensor_realizations
        sufficient_statistics['xi_sqrd'] = torch.pow(realizations['xi'].tensor_realizations, 2)

        # TODO : Optimize to compute the matrix multiplication only once for the reconstruction
        individual_parameters = self.get_param_from_real(realizations)
        data_reconstruction = self.compute_individual_tensorized(data.timepoints, individual_parameters, attribute_type='MCMC')

        data_reconstruction *= data.mask.float() # speed-up computations

        norm_1 = data.values * data_reconstruction #* data.mask.float()
        norm_2 = data_reconstruction * data_reconstruction #* data.mask.float()

        sufficient_statistics['obs_x_reconstruction'] = norm_1 #.sum(dim=2)
        sufficient_statistics['reconstruction_x_reconstruction'] = norm_2 #.sum(dim=2)

        if self.noise_model == 'bernoulli':
            sufficient_statistics['crossentropy'] = self.compute_individual_attachment_tensorized(data, individual_parameters, attribute_type='MCMC')

        return sufficient_statistics

    def update_model_parameters_burn_in(self, data, realizations):
        # Memoryless part of the algorithm

        # unlink model parameters from updates in realizations!
        realizations = realizations.clone_realizations()

        self.parameters['g'] = realizations['g'].tensor_realizations
        xi = realizations['xi'].tensor_realizations
        self.parameters['xi_mean'] = torch.mean(xi)
        self.parameters['xi_std'] = torch.std(xi)
        tau = realizations['tau'].tensor_realizations
        self.parameters['tau_mean'] = torch.mean(tau)
        self.parameters['tau_std'] = torch.std(tau)

        param_ind = self.get_param_from_real(realizations)
        self.parameters['noise_std'] = NoiseModel.rmse_model(self, data, param_ind, attribute_type='MCMC')

        if self.noise_model == 'bernoulli':
            crossentropy = self.compute_individual_attachment_tensorized(data, param_ind, attribute_type='MCMC').sum()
            self.parameters['crossentropy'] = crossentropy

    def update_model_parameters_normal(self, data, suff_stats):
        # Stochastic sufficient statistics used to update the parameters of the model

        self.parameters['g'] = suff_stats['g']

        tau_mean = self.parameters['tau_mean']
        tau_var_updt = torch.mean(suff_stats['tau_sqrd']) - 2. * tau_mean * torch.mean(suff_stats['tau'])
        tau_var = tau_var_updt + tau_mean ** 2
        self.parameters['tau_std'] = self._compute_std_from_var(tau_var, varname='tau_std')
        self.parameters['tau_mean'] = torch.mean(suff_stats['tau'])

        xi_mean = self.parameters['xi_mean']
        xi_var_updt = torch.mean(suff_stats['xi_sqrd']) - 2. * xi_mean * torch.mean(suff_stats['xi'])
        xi_var = xi_var_updt + xi_mean ** 2
        self.parameters['xi_std'] = self._compute_std_from_var(xi_var, varname='xi_std')
        self.parameters['xi_mean'] = torch.mean(suff_stats['xi'])

        S1 = data.L2_norm
        S2 = suff_stats['obs_x_reconstruction'].sum()
        S3 = suff_stats['reconstruction_x_reconstruction'].sum()

        noise_var = (S1 - 2. * S2 + S3) / data.n_observations
        self.parameters['noise_std'] = self._compute_std_from_var(noise_var, varname='noise_std')

        if self.noise_model == 'bernoulli':
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
doc_with_(UnivariateModel.compute_individual_tensorized_linear,
          UnivariateModel.compute_individual_tensorized,
          mapping={'the model': 'the model (linear)'})
doc_with_(UnivariateModel.compute_individual_tensorized_logistic,
          UnivariateModel.compute_individual_tensorized,
          mapping={'the model': 'the model (logistic)'})

doc_with_(UnivariateModel.compute_jacobian_tensorized_linear,
          UnivariateModel.compute_jacobian_tensorized,
          mapping={'the model': 'the model (linear)'})
doc_with_(UnivariateModel.compute_jacobian_tensorized_logistic,
          UnivariateModel.compute_jacobian_tensorized,
          mapping={'the model': 'the model (logistic)'})

doc_with_(UnivariateModel.compute_individual_ages_from_biomarker_values_tensorized_logistic,
          UnivariateModel.compute_individual_ages_from_biomarker_values_tensorized,
          mapping={'the model': 'the model (logistic)'})
