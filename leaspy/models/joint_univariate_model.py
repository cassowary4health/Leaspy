import json
import torch
import math


from leaspy import __version__

from leaspy.models.abstract_model import AbstractModel
from leaspy.models.utils.attributes import AttributesFactory
from leaspy.models.utils.initialization.model_initialization import initialize_parameters
from leaspy.models.utils.noise_model import NoiseModel
from leaspy.models.utils.ordinal import OrdinalModelMixin

from leaspy.utils.typing import DictParamsTorch


from leaspy.utils.typing import Optional
from leaspy.utils.docs import doc_with_super, doc_with_
from leaspy.utils.subtypes import suffixed_method
from leaspy.exceptions import LeaspyModelInputError


TWO_PI = torch.tensor(2 * math.pi)


# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either model values or model values + jacobians wrt individual parameters
# TODO refact? subclass or other proper code technique to extract model's concrete formulation depending on if linear, logistic, mixed log-lin, ...


@doc_with_super()
class JointUnivariateModel(AbstractModel, OrdinalModelMixin):
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
        #'univariate_linear': '_linear',
        'joint_univariate_logistic': '_logistic'
    }

    ## --- INITIALIZATION FUNCTIONS ---
    def __init__(self, name: str, **kwargs):

        super().__init__(name)

        self.dimension = 1
        self.source_dimension = 0  # TODO, None ???
        self.noise_model = 'joint'

        self.parameters = {
            "g": None,
            "v0": None,
            "rho": None,
            "nu": None,
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
                'v0_std': None,
                'rho_std': None,
                'nu_std': None,
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

    def load_hyperparameters(self, hyperparameters: dict):

        expected_hyperparameters = ('features',)
        if 'features' in hyperparameters.keys():
            self.features = hyperparameters['features']

        # load new `noise_model` directly in-place & add the recognized hyperparameters to known tuple
        # TODO? forbid the usage of `gaussian_diagonal` noise for such model?
        expected_hyperparameters += NoiseModel.set_noise_model_from_hyperparameters(self, hyperparameters)

        # special hyperparameter(s) for ordinal model
        expected_hyperparameters += self._handle_ordinal_hyperparameters(hyperparameters)

        self._raise_if_unknown_hyperparameters(expected_hyperparameters, hyperparameters)

    ## --- WHAT IS THIS INITIALIZATION? ---
    def initialize(self, dataset, method="default"):
        self.features = dataset.headers
        self.parameters = initialize_parameters(self, dataset, method)
        self.attributes = AttributesFactory.attributes(self.name, dimension=1,
                                                       **self._attributes_factory_ordinal_kws)

        # Postpone the computation of attributes when really needed!
        #self.attributes.update(['all'], self.parameters)

        self.is_initialized = True

    def initialize_MCMC_toolbox(self):
        """
        Initialize Monte-Carlo Markov-Chain toolbox for calibration of model
        """
        # TODO to move in the MCMC-fit algorithm
        self.MCMC_toolbox = {
            'priors': {'g_std': 0.01,
                       'v0_std': 0.01,
                       'rho_std': 0.01,
                       'nu_std': 0.01,}, # population parameter
        }

        # specific priors for ordinal models
        self._initialize_MCMC_toolbox_ordinal_priors()

        self.MCMC_toolbox['attributes'] = AttributesFactory.attributes(self.name, dimension=1,
                                                                       **self._attributes_factory_ordinal_kws)
        population_dictionary = self._create_dictionary_of_population_realizations()
        self.update_MCMC_toolbox(["all"], population_dictionary)

    def random_variable_informations(self):

        ## Population variables
        g_infos = {
            "name": "g",
            "shape": torch.Size([1]),
            "type": "population",
            "rv_type": "multigaussian"
        }
        v0_infos = {
            "name": "v0",
            "shape": torch.Size([1]),
            "type": "population",
            "rv_type": "multigaussian"
        }
        rho_infos = {
            "name": "rho",
            "shape": torch.Size([1]),
            "type": "population",
            "rv_type": "multigaussian"
        }
        nu_infos = {
            "name": "nu",
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
            "rho": rho_infos,
            "nu": nu_infos,
            "tau": tau_infos,
            "xi": xi_infos,
            "v0": v0_infos
        }

        self._add_ordinal_random_variables(variables_infos)

        return variables_infos

    ## --- LOAD & SAVE FUNCTIONS ---
    def load_parameters(self, parameters):
        self.parameters = {}

        for k in parameters.keys():
            self.parameters[k] = torch.tensor(parameters[k])

        # re-build the ordinal_infos if relevant
        self._rebuild_ordinal_infos_from_model_parameters()

        # derive the model attributes from model parameters upon reloading of model
        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension,
                                                       **self._attributes_factory_ordinal_kws)
        self.attributes.update(['all'], self.parameters)

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

        self._export_extra_ordinal_settings(model_settings)

        # TODO : in leaspy models there should be a method to only return the dict describing the model
        # and then another generic method (inherited) should save this dict
        # (with extra standard fields such as 'leaspy_version' for instance)

        # Default json.dump kwargs:
        kwargs = {'indent': 2, **kwargs}

        with open(path, 'w') as fp:
            json.dump(model_settings, fp, **kwargs)

    ## --- [WHAT IS THIS FUNCTIONS ---
    def update_MCMC_toolbox(self, vars_to_update, realizations):
        """
        Update the MCMC toolbox with a collection of realizations of model population parameters.

        TODO to move in the MCMC-fit algorithm

        Parameters
        ----------
        vars_to_update : container[str] (list, tuple, ...)
            Names of the population parameters to update in MCMC toolbox
        realizations : :class:`.CollectionRealization`
            All the realizations to update MCMC toolbox with
        """
        values = {}
        if any(c in vars_to_update for c in ('g', 'all')):
            values['g'] = realizations['g'].tensor_realizations

        if any(c in vars_to_update for c in ('v0', 'v0_collinear', 'all')):
            values['v0'] = realizations['v0'].tensor_realizations

        if any(c in vars_to_update for c in ('rho', 'all')):
            values['rho'] = realizations['rho'].tensor_realizations

        if any(c in vars_to_update for c in ('nu', 'all')):
            values['nu'] = realizations['nu'].tensor_realizations

        self._update_MCMC_toolbox_ordinal(vars_to_update, realizations, values)
        self.MCMC_toolbox['attributes'].update(vars_to_update, values)

    def _call_method_from_attributes(self, method_name: str, attribute_type: Optional[str], **call_kws):
        # TODO: move in a abstract parent class for univariate & multivariate models (like AbstractManifoldModel...)
        if attribute_type is None:
            return getattr(self.attributes, method_name)(**call_kws)
        elif attribute_type == 'MCMC':
            return getattr(self.MCMC_toolbox['attributes'], method_name)(**call_kws)
        else:
            raise LeaspyModelInputError(f"The specified attribute type does not exist: {attribute_type}. "
                                        "Should be None or 'MCMC'.")

    def _get_attributes(self, attribute_type: Optional[str]):
        return self._call_method_from_attributes('get_attributes', attribute_type)

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

    ## --- [BASIC] COMPUTATION OF LONGITUDINAL PATIENT VALUE FUNCTIONS ---
    @suffixed_method
    def compute_individual_tensorized(self, timepoints, individual_parameters, *, attribute_type=None):
        pass

    def compute_individual_tensorized_logistic(self, timepoints, individual_parameters, *, attribute_type=None):

        # Population parameters
        g, v0, rho, nu = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        LL = torch.exp(v0)*reparametrized_time.unsqueeze(-1)
        if self.is_ordinal:
            # add an extra dimension for the levels of the ordinal item
            LL = LL.unsqueeze(-1)
            g = g.unsqueeze(-1)
            deltas = self._get_deltas(attribute_type) # (features, max_level)
            deltas = deltas.unsqueeze(0).unsqueeze(0)  # add (ind, timepoints) dimensions
            LL = LL - deltas.cumsum(dim=-1) #TODO

        # TODO? more efficient & accurate to compute `torch.exp(-LL + log_g)` since we directly sample & stored log_g
        model = 1. / (1. + g * torch.exp(-LL))

        # Compute the pdf and not the sf
        if self.noise_model == 'ordinal':
            model = self.compute_ordinal_pdf_from_ordinal_sf(model)

        return model # (n_individuals, n_timepoints, n_features == 1 [, extra_dim_ordinal_models])

    def compute_individual_tensorized_survival(self, timepoints, individual_parameters, *, attribute_type=None):
        # Population parameters
        g, v0, rho, nu = self._get_attributes(attribute_type)

        # Individual parameters
        xi = individual_parameters['xi'].reshape(timepoints.shape)
        reparametrized_time = (torch.exp(xi)*timepoints)

        survival = torch.exp(-(reparametrized_time / nu) ** rho)

        return survival  # (n_individuals, 1, 1_features)

    def compute_individual_tensorized_hazard(self, timepoints, individual_parameters, *, attribute_type=None):
        # Population parameters
        g, v0, rho, nu = self._get_attributes(attribute_type)

        # Individual parameters
        xi = individual_parameters['xi'].reshape(timepoints.shape)
        reparametrized_time = (torch.exp(xi))*timepoints

        hazard = (rho/nu)*((reparametrized_time/nu)**(rho-1))

        return hazard  # (n_individuals, 1, 1_features)

    """def compute_individual_tensorized_linear(self, timepoints, individual_parameters, *, attribute_type=None):

        # Population parameters
        positions = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        return positions + reparametrized_time.unsqueeze(-1)"""

    ## --- [JACOBIAN] COMPUTATION OF LONGITUDINAL PATIENT VALUE FUNCTIONS ---
    @suffixed_method
    def compute_jacobian_tensorized(self, timepoints, individual_parameters, *, attribute_type=None):
        pass

    def compute_jacobian_tensorized_logistic(self, timepoints, individual_parameters, *, attribute_type=None):

        # Population parameters
        g, v0, rho, nu = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)
        alpha = torch.exp(xi).reshape(-1, 1, 1)

        # Log likelihood computation
        LL = torch.exp(v0)*reparametrized_time.unsqueeze(-1)  # (n_individuals, n_timepoints, n_features==1)

        if self.is_ordinal:
            # add an extra dimension for the levels of the ordinal item
            LL = LL.unsqueeze(-1)
            g = g.unsqueeze(-1)
            deltas = self._get_deltas(attribute_type)  # (features, max_level)
            deltas = deltas.unsqueeze(0).unsqueeze(0)  # add (ind, timepoints) dimensions
            LL = LL - deltas.cumsum(dim=-1)

        model = 1. / (1. + g * torch.exp(-LL))

        c = model * (1. - model)

        derivatives = {
            'xi': (reparametrized_time).unsqueeze(-1).unsqueeze(-1),
            'tau': (-alpha).unsqueeze(-1),
        }

        if self.is_ordinal:
            ordinal_lvls_shape = c.shape[-1]
            for param in derivatives:
                derivatives[param] = derivatives[param].unsqueeze(-2).repeat(1, 1, 1, ordinal_lvls_shape, 1)

        for param in derivatives:
            derivatives[param] = c.unsqueeze(-1) * derivatives[param]

        # Compute derivative of the pdf and not of the sf
        if self.noise_model == 'ordinal':
            for param in derivatives:
                derivatives[param] = self.compute_ordinal_pdf_from_ordinal_sf(derivatives[param])

        # dict[param_name: str, torch.Tensor of shape(n_ind, n_tpts, n_fts == 1 [, extra_dim_ordinal_models], n_dims_param)]
        return derivatives

    """def compute_jacobian_tensorized_linear(self, timepoints, individual_parameters, *, attribute_type=None):

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

            return derivatives"""
    ## --- WHERE ARE THEY CALL??? ---
    @suffixed_method
    def compute_individual_ages_from_biomarker_values_tensorized(self, value: torch.Tensor,
                                                                 individual_parameters: dict, feature: str):
        pass

    def compute_individual_ages_from_biomarker_values_tensorized_logistic(self, value: torch.Tensor,
                                                                          individual_parameters: dict, feature: str):

        if value.dim() != 2:
            raise LeaspyModelInputError(f"The biomarker value should be dim 2, not {value.dim()}!")

        if self.is_ordinal:
            return self._compute_individual_ages_from_biomarker_values_tensorized_logistic_ordinal(value, individual_parameters)

        # avoid division by zero:
        value = value.masked_fill((value == 0) | (value == 1), float('nan'))

        # get tensorized attributes
        g, rho, nu = self._get_attributes(None)
        xi, tau = individual_parameters['xi'], individual_parameters['tau']

        xi_std = xi-self.parameters['xi_mean']

        # compute age
        ages = torch.exp(xi_std) * torch.log(g/(1 / value - 1)) + tau
        assert ages.shape == value.shape

        return ages

    '''def _compute_individual_ages_from_biomarker_values_tensorized_logistic_ordinal(self, value: torch.Tensor,
                                                                          individual_parameters: dict):
        """
        For one individual, compute age(s) breakpoints at which the given features levels are the most likely (given the subject's
        individual parameters).

        Consistency checks are done in the main API layer.

        Parameters
        ----------
        value : :class:`torch.Tensor`
            Contains the biomarker level value(s) of the subject.

        individual_parameters : dict
            Contains the individual parameters.
            Each individual parameter should be a scalar or array_like

        Returns
        -------
        :class:`torch.Tensor`
            Contains the subject's ages computed at the given values(s)
            Shape of tensor is (1, n_values)

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            if computation is tried on more than 1 individual
        """

        # 1/ get attributes
        g = self._get_attributes(None)
        xi, tau = individual_parameters['xi'], individual_parameters['tau']

        # get feature value for g, v0 and wi
        feat_ind = 0  # univariate model
        g = torch.tensor([g[feat_ind]])  # g and v0 were shape: (n_features in the multivariate model)

        # 2/ compute age
        ages_0 = tau + (torch.exp(-xi)) * ((g / (g + 1) ** 2) * torch.log(g))
        deltas_ft = self._get_deltas(None)[feat_ind]
        delta_max = deltas_ft[torch.isfinite(deltas_ft)].sum()
        ages_max = tau + (torch.exp(-xi)) * ((g / (g + 1) ** 2) * torch.log(g) + delta_max)

        grid_timepoints = torch.linspace(ages_0.item(), ages_max.item(), 1000)

        return self._ordinal_grid_search_value(grid_timepoints, value,
                                               individual_parameters=individual_parameters,
                                               feat_index=feat_ind)'''

    ## --- SUFFICIENT STATISTICS & FOLLOWING PARAMETER UPDATE ---
    def compute_individual_attachment_event(self, data, param_ind, attribute_type=None):

        survival = self.compute_individual_tensorized_survival(data.event_time_min, param_ind, attribute_type=attribute_type)
        hazard = self.compute_individual_tensorized_hazard(data.event_time_max, param_ind, attribute_type=attribute_type)

        hazard = torch.nan_to_num(hazard, nan=1.)
        #print(data.mask_event*torch.log(hazard))
        indiv_ = -data.mask_event*torch.log(hazard)-1*torch.log(survival)
        #print(data.mask.squeeze().shape, indiv_.unsqueeze(-1).expand(data.timepoints.shape).shape)
        #per_vis = data.mask.squeeze()*indiv_.unsqueeze(-1).expand(data.timepoints.shape)
        #print(per_vis.sum(axis = 1).shape)
        return indiv_ # per_vis.sum(axis = 1)


    def compute_individual_attachment_tensorized(self, data, param_ind: DictParamsTorch, *,
                                                 attribute_type) -> torch.FloatTensor:
        """
        Compute attachment term (per subject)

        Parameters
        ----------
        data : :class:`.Dataset`
            Contains the data of the subjects, in particular the subjects' time-points and the mask for nan values & padded visits

        param_ind : dict
            Contain the individual parameters

        attribute_type : Any
            Flag to ask for MCMC attributes instead of model's attributes.

        Returns
        -------
        attachment : :class:`torch.Tensor`
            Negative Log-likelihood, shape = (n_subjects,)

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If invalid `noise_model` for model
        """

        # TODO: this snippet could be implemented directly in NoiseModel (or subclasses depending on noise structure)
        if self.noise_model is None:
            raise LeaspyModelInputError('`noise_model` was not set correctly set.')

        elif 'joint' in self.noise_model:
            # diagonal noise (squared) [same for all features if it's forced to be a scalar]
            # TODO? shouldn't 'noise_std' be part of the "MCMC_toolbox" to use the one we want??
            noise_var = self.parameters['noise_std'] * self.parameters['noise_std'] # slight perf improvement over ** 2, k tensor (or scalar tensor)
            noise_var = noise_var.expand((1, data.dimension)) # 1,k tensor (for scalar products just after) # <!> this formula works with scalar noise as well

            L2_res_per_ind_per_ft = self.compute_sum_squared_per_ft_tensorized(data, param_ind, attribute_type=attribute_type) # ik tensor

            attachment_visits = (0.5 / noise_var) @ L2_res_per_ind_per_ft.t()
            attachment_visits += 0.5 * torch.log(TWO_PI * noise_var) @ data.n_observations_per_ind_per_ft.float().t()
            attachment_visits = attachment_visits.reshape((data.n_individuals,))

        else:
            raise LeaspyModelInputError(f'`noise_model` should be in {NoiseModel.VALID_NOISE_STRUCTS}')

        attachment_events = self.compute_individual_attachment_event(data, param_ind, attribute_type=attribute_type)
        attachment_total = attachment_visits + attachment_events #data.timepoints.shape[1]*
        #print(attachment_visits.shape, attachment_events.shape)

        return attachment_total

    def compute_sufficient_statistics(self, data, realizations):

        # unlink all sufficient statistics from updates in realizations!
        realizations = realizations.clone_realizations()

        sufficient_statistics = {}
        sufficient_statistics['g'] = realizations['g'].tensor_realizations
        sufficient_statistics['v0'] = realizations['v0'].tensor_realizations
        sufficient_statistics['nu'] = realizations['nu'].tensor_realizations
        sufficient_statistics['rho'] = realizations['rho'].tensor_realizations
        sufficient_statistics['tau'] = realizations['tau'].tensor_realizations
        sufficient_statistics['tau_sqrd'] = torch.pow(realizations['tau'].tensor_realizations, 2)
        sufficient_statistics['xi'] = realizations['xi'].tensor_realizations
        sufficient_statistics['xi_sqrd'] = torch.pow(realizations['xi'].tensor_realizations, 2)

        self._add_ordinal_tensor_realizations(realizations, sufficient_statistics)

        # TODO : Optimize to compute the matrix multiplication only once for the reconstruction
        individual_parameters = self.get_param_from_real(realizations)
        data_reconstruction = self.compute_individual_tensorized(data.timepoints, individual_parameters, attribute_type='MCMC')

        if self.noise_model in ['joint']:
            sufficient_statistics['log-likelihood'] = self.compute_individual_attachment_tensorized(data, individual_parameters,
                                                                                                    attribute_type='MCMC').sum()
        return sufficient_statistics

    def update_model_parameters_burn_in(self, data, realizations):
        # Memoryless part of the algorithm

        # unlink model parameters from updates in realizations!
        realizations = realizations.clone_realizations()

        self.parameters['g'] = realizations['g'].tensor_realizations
        self.parameters['v0'] = realizations['v0'].tensor_realizations
        self.parameters['nu'] = realizations['nu'].tensor_realizations
        self.parameters['rho'] = realizations['rho'].tensor_realizations
        xi = realizations['xi'].tensor_realizations
        #self.parameters['xi_mean'] = torch.mean(xi)
        self.parameters['xi_std'] = torch.std(xi)
        tau = realizations['tau'].tensor_realizations
        self.parameters['tau_mean'] = torch.mean(tau)
        self.parameters['tau_std'] = torch.std(tau)

        self._add_ordinal_tensor_realizations(realizations, self.parameters)

        param_ind = self.get_param_from_real(realizations)
        if self.noise_model in ['joint']:
            total_attachment = self.compute_individual_attachment_tensorized(data, param_ind,
                                                                              attribute_type='MCMC').sum()
            self.parameters['log-likelihood'] = total_attachment

    def update_model_parameters_normal(self, data, suff_stats):
        # Stochastic sufficient statistics used to update the parameters of the model

        self.parameters['g'] = suff_stats['g']
        self.parameters['v0'] = suff_stats['v0']
        self.parameters['nu'] = suff_stats['nu']
        self.parameters['rho'] = suff_stats['rho']

        tau_mean = self.parameters['tau_mean']
        tau_var_updt = torch.mean(suff_stats['tau_sqrd']) - 2. * tau_mean * torch.mean(suff_stats['tau'])
        tau_var = tau_var_updt + tau_mean ** 2
        self.parameters['tau_std'] = self._compute_std_from_var(tau_var, varname='tau_std')
        self.parameters['tau_mean'] = torch.mean(suff_stats['tau'])

        xi_mean = self.parameters['xi_mean']
        xi_var_updt = torch.mean(suff_stats['xi_sqrd']) - 2. * xi_mean * torch.mean(suff_stats['xi'])
        xi_var = xi_var_updt + xi_mean ** 2
        self.parameters['xi_std'] = self._compute_std_from_var(xi_var, varname='xi_std')
        #self.parameters['xi_mean'] = torch.mean(suff_stats['xi'])

        self._add_ordinal_sufficient_statistics(suff_stats, self.parameters)

        if self.noise_model in ['joint']:
            self.parameters['log-likelihood'] = suff_stats['log-likelihood'].sum()
        else:
            raise()

# document some methods (we cannot decorate them at method creation since they are not yet decorated from `doc_with_super`)
#doc_with_(JointUnivariateModel.compute_individual_tensorized_linear,
#          JointUnivariateModel.compute_individual_tensorized,
#          mapping={'the model': 'the model (linear)'})
doc_with_(JointUnivariateModel.compute_individual_tensorized_logistic,
          JointUnivariateModel.compute_individual_tensorized,
          mapping={'the model': 'the model (logistic)'})

#doc_with_(JointUnivariateModel.compute_jacobian_tensorized_linear,
#          JointUnivariateModel.compute_jacobian_tensorized,
#          mapping={'the model': 'the model (linear)'})
doc_with_(JointUnivariateModel.compute_jacobian_tensorized_logistic,
          JointUnivariateModel.compute_jacobian_tensorized,
          mapping={'the model': 'the model (logistic)'})

doc_with_(JointUnivariateModel.compute_individual_ages_from_biomarker_values_tensorized_logistic,
          JointUnivariateModel.compute_individual_ages_from_biomarker_values_tensorized,
          mapping={'the model': 'the model (logistic)'})
