import torch

from leaspy.models.abstract_multivariate_model import AbstractMultivariateModel
from leaspy.models.utils.attributes import AttributesFactory
from leaspy.models.utils.noise_model import NoiseModel

from leaspy.utils.docs import doc_with_super, doc_with_
from leaspy.utils.subtypes import suffixed_method
from leaspy.exceptions import LeaspyModelInputError

# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either model values or model values + jacobians wrt individual parameters
# TODO refact? subclass or other proper code technique to extract model's concrete formulation depending on if linear, logistic, mixed log-lin, ...


@doc_with_super()
class MultivariateModel(AbstractMultivariateModel):
    """
    Manifold model for multiple variables of interest (logistic or linear formulation).

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
        'linear': '_linear',
        'logistic': '_logistic',
        'mixed_linear-logistic': '_mixed',
    }

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.parameters["v0"] = None
        self.MCMC_toolbox['priors']['v0_std'] = None  # Value, Coef

        self._subtype_suffix = self._check_subtype()

        # enforce a prior for v0_mean --> legacy / never used in practice
        self._set_v0_prior = False


    def _check_subtype(self):
        if self.name not in self.SUBTYPES_SUFFIXES.keys():
            raise LeaspyModelInputError(f'Multivariate model name should be among these valid sub-types: '
                                        f'{list(self.SUBTYPES_SUFFIXES.keys())}.')

        return self.SUBTYPES_SUFFIXES[self.name]

    def load_parameters(self, parameters):
        # TODO? Move this method in higher level class AbstractMultivariateModel? (<!> Attributes class)
        self.parameters = {}
        for k in parameters.keys():
            if k in ['mixing_matrix']:
                continue
            self.parameters[k] = torch.tensor(parameters[k])

        # derive the model attributes from model parameters upon reloading of model
        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension)
        self.attributes.update(['all'], self.parameters)

    @suffixed_method
    def compute_individual_tensorized(self, timepoints, individual_parameters, *, attribute_type=None):
        pass

    def compute_individual_tensorized_linear(self, timepoints, individual_parameters, *, attribute_type=None):

        # Population parameters
        positions, velocities, mixing_matrix = self._get_attributes(attribute_type)
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1)  # for automatic broadcast on n_features (last dim)

        # Model expected value
        model = positions + velocities * reparametrized_time

        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(mixing_matrix.t())
            model += wi.unsqueeze(-2)

        return model # (n_individuals, n_timepoints, n_features)

    def compute_individual_tensorized_logistic(self, timepoints, individual_parameters, *, attribute_type=None):

        # Population parameters
        g, v0, a_matrix = self._get_attributes(attribute_type)
        g_plus_1 = 1. + g
        b = g_plus_1 * g_plus_1 / g

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1) # (n_individuals, n_timepoints, n_features)

        # Model expected value
        t = v0 * reparametrized_time
        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(a_matrix.t())
            t += wi.unsqueeze(-2) # unsqueeze for (n_timepoints)

        # TODO? more efficient & accurate to compute `torch.exp(-t*b + log_g)` since we directly sample & stored log_g
        model = 1. / (1. + g * torch.exp(-t * b))

        return model # (n_individuals, n_timepoints, n_features)

    @suffixed_method
    def compute_individual_ages_from_biomarker_values_tensorized(self, value: torch.Tensor,
                                                                 individual_parameters: dict, feature: str):
        pass

    def compute_individual_ages_from_biomarker_values_tensorized_logistic(self, value: torch.Tensor,
                                                                          individual_parameters: dict, feature: str):
        if value.dim() != 2:
            raise LeaspyModelInputError(f"The biomarker value should be dim 2, not {value.dim()}!")

        # avoid division by zero:
        value = value.masked_fill((value == 0) | (value == 1), float('nan'))

        # 1/ get attributes
        g, v0, a_matrix = self._get_attributes(None)
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(a_matrix.t())
        else:
            wi = 0

        # get feature value for g, v0 and wi
        feat_ind = self.features.index(feature)  # all consistency checks were done in API layer
        g = torch.tensor([g[feat_ind]])  # g and v0 were shape: (n_features in the multivariate model)
        v0 = torch.tensor([v0[feat_ind]])
        if self.source_dimension != 0:
            wi = wi[0, feat_ind].item()  # wi was shape (1, n_features)

        # 2/ compute age
        ages = tau + (torch.exp(-xi) / v0) * ((g / (g + 1) ** 2) * torch.log(g/(1 / value - 1)) - wi)
        # assert ages.shape == value.shape

        return ages

    @suffixed_method
    def compute_jacobian_tensorized(self, timepoints, individual_parameters, *, attribute_type=None):
        pass

    def compute_jacobian_tensorized_linear(self, timepoints, individual_parameters, *, attribute_type=None):

        # Population parameters
        _, v0, mixing_matrix = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1) # (n_individuals, n_timepoints, n_features)
        alpha = torch.exp(xi).reshape(-1, 1, 1)
        dummy_to_broadcast_n_ind_n_tpts = torch.ones_like(reparametrized_time)

        # Jacobian of model expected value w.r.t. individual parameters
        derivatives = {
            'xi': (v0 * reparametrized_time).unsqueeze(-1), # add a last dimension for len param
            'tau': (v0 * -alpha * dummy_to_broadcast_n_ind_n_tpts).unsqueeze(-1), # same
        }

        if self.source_dimension > 0:
            derivatives['sources'] = mixing_matrix.expand((1,1,-1,-1)) * dummy_to_broadcast_n_ind_n_tpts.unsqueeze(-1)

        # dict[param_name: str, torch.Tensor of shape(n_ind, n_tpts, n_fts, n_dims_param)]
        return derivatives

    def compute_jacobian_tensorized_logistic(self, timepoints, individual_parameters, *, attribute_type=None):
        # TODO: refact highly inefficient (many duplicated code from `compute_individual_tensorized_logistic`)

        # Population parameters
        g, v0, a_matrix = self._get_attributes(attribute_type)
        g_plus_1 = 1. + g
        b = g_plus_1 * g_plus_1 / g

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1) # (n_individuals, n_timepoints, n_features)
        alpha = torch.exp(xi).reshape(-1, 1, 1)

        # Model expected value
        t = v0 * reparametrized_time
        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(a_matrix.t())
            t += wi.unsqueeze(-2)
        model = 1. / (1. + g * torch.exp(-t * b))

        # Jacobian of model expected value w.r.t. individual parameters
        c = model * (1. - model) * b

        derivatives = {
            'xi': (c * v0 * reparametrized_time).unsqueeze(-1),
            'tau': (c * v0 * -alpha).unsqueeze(-1),
        }
        if self.source_dimension > 0:
            derivatives['sources'] = c.unsqueeze(-1) * a_matrix.expand((1,1,-1,-1))

        # dict[param_name: str, torch.Tensor of shape(n_ind, n_tpts, n_fts, n_dims_param)]
        return derivatives

    ##############################
    ### MCMC-related functions ###
    ##############################

    def initialize_MCMC_toolbox(self):
        self.MCMC_toolbox = {
            'priors': {'g_std': 0.01, 'v0_std': 0.01, 'betas_std': 0.01}, # population parameters
            'attributes': AttributesFactory.attributes(self.name, self.dimension, self.source_dimension)
        }

        # Initialize a prior for v0_mean (legacy code / never used in practice)
        if self._set_v0_prior:
            self.MCMC_toolbox['priors']['v0_mean'] = self.parameters['v0'].clone().detach()
            self.MCMC_toolbox['priors']['s_v0'] = 0.1
            # TODO? same on g?

        # TODO? why not passing the ready-to-use collection realizations that is initialized at beginning of fit algo and use it here instead?
        population_dictionary = self._create_dictionary_of_population_realizations()
        self.update_MCMC_toolbox(["all"], population_dictionary)

    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        L = name_of_the_variables_that_have_been_changed
        values = {}
        if any(c in L for c in ('g', 'all')):
            values['g'] = realizations['g'].tensor_realizations
        if any(c in L for c in ('v0', 'v0_collinear', 'all')):
            values['v0'] = realizations['v0'].tensor_realizations
        if any(c in L for c in ('betas', 'all')) and self.source_dimension != 0:
            values['betas'] = realizations['betas'].tensor_realizations

        self.MCMC_toolbox['attributes'].update(name_of_the_variables_that_have_been_changed, values)

    def _center_xi_realizations(self, realizations):
        # This operation does not change the orthonormal basis
        # (since the resulting v0 is collinear to the previous one)
        # Nor all model computations (only v0 * exp(xi_i) matters),
        # it is only intended for model identifiability / `xi_i` regularization
        # <!> all operations are performed in "log" space (v0 is log'ed)
        mean_xi = torch.mean(realizations['xi'].tensor_realizations)
        realizations['xi'].tensor_realizations = realizations['xi'].tensor_realizations - mean_xi
        realizations['v0'].tensor_realizations = realizations['v0'].tensor_realizations + mean_xi

        self.update_MCMC_toolbox(['v0_collinear'], realizations)

        return realizations

    def compute_sufficient_statistics(self, data, realizations):

        # modify realizations in-place
        realizations = self._center_xi_realizations(realizations)

        # unlink all sufficient statistics from updates in realizations!
        realizations = realizations.clone_realizations()

        sufficient_statistics = {
            'g': realizations['g'].tensor_realizations,
            'v0': realizations['v0'].tensor_realizations,
            'tau': realizations['tau'].tensor_realizations,
            'tau_sqrd': torch.pow(realizations['tau'].tensor_realizations, 2),
            'xi': realizations['xi'].tensor_realizations,
            'xi_sqrd': torch.pow(realizations['xi'].tensor_realizations, 2)
        }
        if self.source_dimension != 0:
            sufficient_statistics['betas'] = realizations['betas'].tensor_realizations

        individual_parameters = self.get_param_from_real(realizations)

        data_reconstruction = self.compute_individual_tensorized(data.timepoints,
                                                                 individual_parameters,
                                                                 attribute_type='MCMC')

        data_reconstruction *= data.mask.float()  # speed-up computations

        norm_1 = data.values * data_reconstruction
        norm_2 = data_reconstruction * data_reconstruction

        sufficient_statistics['obs_x_reconstruction'] = norm_1  # .sum(dim=2) # no sum on features...
        sufficient_statistics['reconstruction_x_reconstruction'] = norm_2  # .sum(dim=2) # no sum on features...

        if self.noise_model == 'bernoulli':
            sufficient_statistics['crossentropy'] = self.compute_individual_attachment_tensorized(data, individual_parameters,
                                                                                                  attribute_type='MCMC')

        return sufficient_statistics

    def update_model_parameters_burn_in(self, data, realizations):
        # During the burn-in phase, we only need to store the following parameters (cf. !66 and #60)
        # - noise_std
        # - *_mean/std for regularization of individual variables
        # - others population parameters for regularization of population variables
        # We don't need to update the model "attributes" (never used during burn-in!)

        # TODO: refactorize?

        # modify realizations in-place!
        realizations = self._center_xi_realizations(realizations)

        # unlink model parameters from updates in realizations!
        realizations = realizations.clone_realizations()

        # Memoryless part of the algorithm
        self.parameters['g'] = realizations['g'].tensor_realizations

        v0_emp = realizations['v0'].tensor_realizations
        if self.MCMC_toolbox['priors'].get('v0_mean', None) is not None:
            v0_mean = self.MCMC_toolbox['priors']['v0_mean']
            s_v0 = self.MCMC_toolbox['priors']['s_v0']
            sigma_v0 = self.MCMC_toolbox['priors']['v0_std']
            self.parameters['v0'] = (1 / (1 / (s_v0 ** 2) + 1 / (sigma_v0 ** 2))) * (
                        v0_emp / (sigma_v0 ** 2) + v0_mean / (s_v0 ** 2))
        else:
            # new default
            self.parameters['v0'] = v0_emp

        if self.source_dimension != 0:
            self.parameters['betas'] = realizations['betas'].tensor_realizations

        xi = realizations['xi'].tensor_realizations
        # self.parameters['xi_mean'] = torch.mean(xi)  # fixed = 0 by design
        self.parameters['xi_std'] = torch.std(xi)
        tau = realizations['tau'].tensor_realizations
        self.parameters['tau_mean'] = torch.mean(tau)
        self.parameters['tau_std'] = torch.std(tau)

        # by design: sources_mean = 0., sources_std = 1.

        param_ind = self.get_param_from_real(realizations)
        self.parameters['noise_std'] = NoiseModel.rmse_model(self, data, param_ind, attribute_type='MCMC')

        if self.noise_model == 'bernoulli':
            self.parameters['crossentropy'] = self.compute_individual_attachment_tensorized(data, param_ind,
                                                                                            attribute_type='MCMC').sum()

    def update_model_parameters_normal(self, data, suff_stats):
        # TODO? add a true, configurable, validation for all parameters? (e.g.: bounds on tau_var/std but also on tau_mean, ...)

        # Stochastic sufficient statistics used to update the parameters of the model

        # TODO with Raphael : check the SS, especially the issue with mean(xi) and v_k
        # TODO : 1. Learn the mean of xi and v_k
        # TODO : 2. Set the mean of xi to 0 and add it to the mean of V_k
        self.parameters['g'] = suff_stats['g']
        self.parameters['v0'] = suff_stats['v0']
        if self.source_dimension != 0:
            self.parameters['betas'] = suff_stats['betas']

        tau_mean = self.parameters['tau_mean']
        tau_var_updt = torch.mean(suff_stats['tau_sqrd']) - 2. * tau_mean * torch.mean(suff_stats['tau'])
        tau_var = tau_var_updt + tau_mean ** 2
        self.parameters['tau_std'] = self._compute_std_from_var(tau_var, varname='tau_std')
        self.parameters['tau_mean'] = torch.mean(suff_stats['tau'])

        xi_mean = self.parameters['xi_mean']
        xi_var_updt = torch.mean(suff_stats['xi_sqrd']) - 2. * xi_mean * torch.mean(suff_stats['xi'])
        xi_var = xi_var_updt + xi_mean ** 2
        self.parameters['xi_std'] = self._compute_std_from_var(xi_var, varname='xi_std')
        # self.parameters['xi_mean'] = torch.mean(suff_stats['xi'])  # fixed = 0 by design

        if 'scalar' in self.noise_model:
            # scalar noise (same for all features)
            S1 = data.L2_norm
            S2 = suff_stats['obs_x_reconstruction'].sum()
            S3 = suff_stats['reconstruction_x_reconstruction'].sum()

            noise_var = (S1 - 2. * S2 + S3) / data.n_observations
        else:
            # keep feature dependence on feature to update diagonal noise (1 free param per feature)
            S1 = data.L2_norm_per_ft
            S2 = suff_stats['obs_x_reconstruction'].sum(dim=(0, 1))
            S3 = suff_stats['reconstruction_x_reconstruction'].sum(dim=(0, 1))

            # tensor 1D, shape (dimension,)
            noise_var = (S1 - 2. * S2 + S3) / data.n_observations_per_ft.float()

        self.parameters['noise_std'] = self._compute_std_from_var(noise_var, varname='noise_std')

        if self.noise_model == 'bernoulli':
            self.parameters['crossentropy'] = suff_stats['crossentropy'].sum()

    ###################################
    ### Random Variable Information ###
    ###################################

    def random_variable_informations(self):

        # --- Population variables
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
            "rv_type": "multigaussian",
            "scale": .5  # cf. GibbsSampler
        }

        # --- Individual variables
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
            "tau": tau_infos,
            "xi": xi_infos,
        }

        if self.source_dimension != 0:
            variables_infos['sources'] = sources_infos
            variables_infos['betas'] = betas_infos

        return variables_infos

# document some methods (we cannot decorate them at method creation since they are not yet decorated from `doc_with_super`)
doc_with_(MultivariateModel.compute_individual_tensorized_linear,
          MultivariateModel.compute_individual_tensorized,
          mapping={'the model': 'the model (linear)'})
doc_with_(MultivariateModel.compute_individual_tensorized_logistic,
          MultivariateModel.compute_individual_tensorized,
          mapping={'the model': 'the model (logistic)'})
#doc_with_(MultivariateModel.compute_individual_tensorized_mixed,
#          MultivariateModel.compute_individual_tensorized,
#          mapping={'the model': 'the model (mixed logistic-linear)'})

doc_with_(MultivariateModel.compute_jacobian_tensorized_linear,
          MultivariateModel.compute_jacobian_tensorized,
          mapping={'the model': 'the model (linear)'})
doc_with_(MultivariateModel.compute_jacobian_tensorized_logistic,
          MultivariateModel.compute_jacobian_tensorized,
          mapping={'the model': 'the model (logistic)'})
#doc_with_(MultivariateModel.compute_jacobian_tensorized_mixed,
#          MultivariateModel.compute_jacobian_tensorized,
#          mapping={'the model': 'the model (mixed logistic-linear)'})

#doc_with_(MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized_linear,
#          MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized,
#          mapping={'the model': 'the model (linear)'})
doc_with_(MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized_logistic,
          MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized,
          mapping={'the model': 'the model (logistic)'})
#doc_with_(MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized_mixed,
#          MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized,
#          mapping={'the model': 'the model (mixed logistic-linear)'})
