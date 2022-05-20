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
            if k in ('mixing_matrix',):
                # The mixing matrix will always be recomputed from `betas` and the other needed model parameters (g, v0)
                continue
            self.parameters[k] = torch.tensor(parameters[k])

        # re-build the ordinal_infos if relevant
        self._rebuild_ordinal_infos_from_model_parameters()

        # derive the model attributes from model parameters upon reloading of model
        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension,
                                                       **self._attributes_factory_ordinal_kws)
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

        if self.is_ordinal:
            # add an extra dimension for the levels of the ordinal item
            reparametrized_time = reparametrized_time.unsqueeze(-1)
            g = g.unsqueeze(-1)
            b = b.unsqueeze(-1)
            v0 = v0.unsqueeze(-1)
            deltas = self._get_deltas(attribute_type)  # (features, max_level)
            deltas = deltas.unsqueeze(0).unsqueeze(0)  # add (ind, timepoints) dimensions
            # infinite deltas (impossible ordinal levels) will induce model = 0 which is intended
            reparametrized_time = reparametrized_time - deltas.cumsum(dim=-1)

        LL = v0 * reparametrized_time

        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(a_matrix.t()).unsqueeze(-2) # unsqueeze for (n_timepoints)
            if self.is_ordinal:
                wi = wi.unsqueeze(-1)
            LL += wi

        # TODO? more efficient & accurate to compute `torch.exp(-t*b + log_g)` since we directly sample & stored log_g
        LL = 1. + g * torch.exp(-LL * b)
        model = 1. / LL

        # For ordinal loss, compute pdf instead of survival function
        if self.noise_model == 'ordinal':
            model = self.compute_ordinal_pdf_from_ordinal_sf(model)

        return model # (n_individuals, n_timepoints, n_features [, extra_dim_ordinal_models])

    @suffixed_method
    def compute_individual_ages_from_biomarker_values_tensorized(self, value: torch.Tensor,
                                                                 individual_parameters: dict, feature: str):
        pass

    def compute_individual_ages_from_biomarker_values_tensorized_logistic(self, value: torch.Tensor,
                                                                          individual_parameters: dict, feature: str):
        if value.dim() != 2:
            raise LeaspyModelInputError(f"The biomarker value should be dim 2, not {value.dim()}!")

        if self.is_ordinal:
            return self._compute_individual_ages_from_biomarker_values_tensorized_logistic_ordinal(value, individual_parameters, feature)

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

    def _compute_individual_ages_from_biomarker_values_tensorized_logistic_ordinal(self, value: torch.Tensor,
                                                                          individual_parameters: dict, feature: str):
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

        feature : str
            Name of the considered biomarker (optional for univariate models, compulsory for multivariate models).

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
        ages_0 = tau + (torch.exp(-xi) / v0) * ((g / (g + 1) ** 2) * torch.log(g) - wi)
        deltas_ft = self._get_deltas(None)[feat_ind]
        delta_max = deltas_ft[torch.isfinite(deltas_ft)].sum()
        ages_max = tau + (torch.exp(-xi) / v0) * ((g / (g + 1) ** 2) * torch.log(g) - wi + delta_max)

        grid_timepoints = torch.linspace(ages_0.item(), ages_max.item(), 1000)

        return self._ordinal_grid_search_value(grid_timepoints, value,
                                               individual_parameters=individual_parameters,
                                               feat_index=feat_ind)

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

        if self.is_ordinal:
            # add an extra dimension for the levels of the ordinal item
            LL = reparametrized_time.unsqueeze(-1)
            g = g.unsqueeze(-1)
            b = b.unsqueeze(-1)
            deltas = self._get_deltas(attribute_type)  # (features, max_level)
            deltas = deltas.unsqueeze(0).unsqueeze(0)  # add (ind, timepoints) dimensions
            LL = LL - deltas.cumsum(dim=-1)
            LL = v0.unsqueeze(-1) * LL

        else:
            LL = v0 * reparametrized_time

        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(a_matrix.t()).unsqueeze(-2) # unsqueeze for (n_timepoints)
            if self.is_ordinal:
                wi = wi.unsqueeze(-1)
            LL += wi
        LL = 1. + g * torch.exp(-LL * b)
        model = 1. / LL

        # Jacobian of model expected value w.r.t. individual parameters
        c = model * (1. - model) * b

        derivatives = {
            'xi': (v0 * reparametrized_time).unsqueeze(-1),
            'tau': (-v0 * alpha).unsqueeze(-1),
        }
        if self.source_dimension > 0:
            derivatives['sources'] = a_matrix.expand((1,1,-1,-1))

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

        # dict[param_name: str, torch.Tensor of shape(n_ind, n_tpts, n_fts [, extra_dim_ordinal_models], n_dims_param)]
        return derivatives

    ##############################
    ### MCMC-related functions ###
    ##############################

    def initialize_MCMC_toolbox(self):
        self.MCMC_toolbox = {
            'priors': {'g_std': 0.01, 'v0_std': 0.01, 'betas_std': 0.01}, # population parameters
        }

        # Initialize a prior for v0_mean (legacy code / never used in practice)
        if self._set_v0_prior:
            self.MCMC_toolbox['priors']['v0_mean'] = self.parameters['v0'].clone().detach()
            self.MCMC_toolbox['priors']['s_v0'] = 0.1
            # TODO? same on g?

        # specific priors for ordinal models
        self._initialize_MCMC_toolbox_ordinal_priors()

        self.MCMC_toolbox['attributes'] = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension,
                                                                       **self._attributes_factory_ordinal_kws)
        # TODO? why not passing the ready-to-use collection realizations that is initialized at beginning of fit algo and use it here instead?
        population_dictionary = self._create_dictionary_of_population_realizations()
        self.update_MCMC_toolbox(["all"], population_dictionary)

    def update_MCMC_toolbox(self, vars_to_update, realizations):
        values = {}
        if any(c in vars_to_update for c in ('g', 'all')):
            values['g'] = realizations['g'].tensor_realizations
        if any(c in vars_to_update for c in ('v0', 'v0_collinear', 'all')):
            values['v0'] = realizations['v0'].tensor_realizations
        if self.source_dimension != 0 and any(c in vars_to_update for c in ('betas', 'all')):
            values['betas'] = realizations['betas'].tensor_realizations

        self._update_MCMC_toolbox_ordinal(vars_to_update, realizations, values)

        self.MCMC_toolbox['attributes'].update(vars_to_update, values)

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

        self._add_ordinal_tensor_realizations(realizations, sufficient_statistics)

        individual_parameters = self.get_param_from_real(realizations)

        data_reconstruction = self.compute_individual_tensorized(data.timepoints, individual_parameters,
                                                                 attribute_type='MCMC')

        if self.noise_model in ['gaussian_scalar', 'gaussian_diagonal']:
            data_reconstruction *= data.mask.float()  # speed-up computations

            norm_1 = data.values * data_reconstruction
            norm_2 = data_reconstruction * data_reconstruction

            sufficient_statistics['obs_x_reconstruction'] = norm_1  # .sum(dim=2) # no sum on features...
            sufficient_statistics['reconstruction_x_reconstruction'] = norm_2  # .sum(dim=2) # no sum on features...

        if self.noise_model in ['bernoulli', 'ordinal', 'ordinal_ranking']:
            sufficient_statistics['log-likelihood'] = self.compute_individual_attachment_tensorized(data, individual_parameters,
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

        self._add_ordinal_tensor_realizations(realizations, self.parameters)

        xi = realizations['xi'].tensor_realizations
        # self.parameters['xi_mean'] = torch.mean(xi)  # fixed = 0 by design
        self.parameters['xi_std'] = torch.std(xi)
        tau = realizations['tau'].tensor_realizations
        self.parameters['tau_mean'] = torch.mean(tau)
        self.parameters['tau_std'] = torch.std(tau)

        # by design: sources_mean = 0., sources_std = 1.

        param_ind = self.get_param_from_real(realizations)

        # Should we really keep this ? cf #54 issue
        if self.noise_model in ['bernoulli', 'ordinal', 'ordinal_ranking']:
            self.parameters['log-likelihood'] = self.compute_individual_attachment_tensorized(data, param_ind,
                                                                                              attribute_type='MCMC').sum()
        else:
            self.parameters['noise_std'] = NoiseModel.rmse_model(self, data, param_ind, attribute_type='MCMC')

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

        self._add_ordinal_sufficient_statistics(suff_stats, self.parameters)

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

        if self.noise_model in ['bernoulli', 'ordinal', 'ordinal_ranking']:
            self.parameters['log-likelihood'] = suff_stats['log-likelihood'].sum()

        elif 'scalar' in self.noise_model:
            # scalar noise (same for all features)
            S1 = data.L2_norm
            S2 = suff_stats['obs_x_reconstruction'].sum()
            S3 = suff_stats['reconstruction_x_reconstruction'].sum()

            noise_var = (S1 - 2. * S2 + S3) / data.n_observations
            self.parameters['noise_std'] = self._compute_std_from_var(noise_var, varname='noise_std')
        else:
            # keep feature dependence on feature to update diagonal noise (1 free param per feature)
            S1 = data.L2_norm_per_ft
            S2 = suff_stats['obs_x_reconstruction'].sum(dim=(0, 1))
            S3 = suff_stats['reconstruction_x_reconstruction'].sum(dim=(0, 1))

            # tensor 1D, shape (dimension,)
            noise_var = (S1 - 2. * S2 + S3) / data.n_observations_per_ft.float()
            self.parameters['noise_std'] = self._compute_std_from_var(noise_var, varname='noise_std')

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

        self._add_ordinal_random_variables(variables_infos)

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
