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

    def _get_deltas(self, attribute_type):
        """

        Parameters
        ----------
        attribute_type: None or 'MCMC'

        Returns
        -------
        The deltas in the ordinal model
        """
        if attribute_type is None:
            return self.attributes.get_deltas()
        elif attribute_type == 'MCMC':
            return self.MCMC_toolbox['attributes'].get_deltas()
        else:
            raise ValueError("The specified attribute type does not exist : {}".format(attribute_type))

    def load_parameters(self, parameters):
        # TODO? Move this method in higher level class AbstractMultivariateModel? (<!> Attributes class)
        self.parameters = {}
        for k in parameters.keys():
            if k in ['mixing_matrix']:
                continue
            self.parameters[k] = torch.tensor(parameters[k])
            if 'deltas' in k:
                if not self.ordinal_infos['batch_deltas']:
                    self.ordinal_infos["features"].append({"name":k[7:], "nb_levels":self.parameters[k].shape[0] + 1}) #k[7:] removes the deltas_ to extract the feature's name
                else:
                    # Find ordinal infos from the delta values themselves
                    for i, feat in enumerate(self.features):
                        bool_array = (self.parameters['deltas'][i,:] != 0).int()
                        if 0 in bool_array:
                            lvl = bool_array.argmin().item() + 2
                        else:
                            lvl = bool_array.shape[0] + 1
                        self.ordinal_infos["features"].append({"name":feat, "nb_levels": lvl})
        if any('deltas' in c for c in parameters.keys()):
            self.ordinal_infos["max_level"] = max([feat["nb_levels"] for feat in self.ordinal_infos["features"]])
            # Mask for setting values > max_level per item to zero
            self.ordinal_infos["mask"] = torch.cat([
                torch.cat([
                    torch.ones((1,1,1,feat['nb_levels'])),
                    torch.zeros((1,1,1,self.ordinal_infos["max_level"] - feat['nb_levels'])),
                ], dim=-1) for feat in self.ordinal_infos["features"]
            ], dim=2)
            ord = self.ordinal_infos
        else:
            ord = None
        # derive the model attributes from model parameters upon reloading of model
        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension, ordinal_infos=ord)
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

    def compute_likelihood_from_cdf(self, model_values):
        """
        Computes the likelihood of an ordinal model assuming that the model_values are the CDF.

        Parameters
        ----------
        model_values : `torch.Tensor`
            Cumulative distribution values : model_values[..., l] is the proba to be superior or equal to l+1

        Returns
        -------
        likelihood : `torch.Tensor`
            likelihood[..., l] is the proba to be equal to l
        """

        s = list(model_values.shape)
        s[3] = 1
        mask = self.ordinal_infos["mask"]
        if len(s) == 5: # in the case of gradient we added a dimension
            mask = mask.unsqueeze(-1)
            first_row = torch.zeros(size=tuple(s)).float() #gradient(P>=0) = 0
        else:
            first_row = torch.ones(size=tuple(s)).float() #(P>=0) = 1
        model = model_values * mask
        cdf_sup = torch.cat([first_row, model], dim=3)
        last_row = torch.zeros(size=tuple(s)).float()
        cdf_inf = torch.cat([model, last_row], dim=3)
        likelihood = cdf_sup - cdf_inf

        return likelihood

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

        if self.noise_model == 'ordinal':
            # add an extra dimension for the levels of the ordinal item
            reparametrized_time = reparametrized_time.unsqueeze(-1)
            g = g.unsqueeze(-1)
            b = b.unsqueeze(-1)
            v0 = v0.unsqueeze(-1)
            deltas = self._get_deltas(attribute_type)
            ordinal_scale = deltas.shape[-1] + 1
            deltas_ = torch.cat([torch.zeros((deltas.shape[0], 1)), deltas], dim=1)  # (features, max_level)
            deltas_ = deltas_.unsqueeze(0).unsqueeze(0)  # add (ind, timepoints) dimensions
            reparametrized_time = reparametrized_time - deltas_.cumsum(dim=-1)

        LL = v0 * reparametrized_time

        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(a_matrix.t()).unsqueeze(-2) # unsqueeze for (n_timepoints)
            if self.noise_model == 'ordinal':
                wi = wi.unsqueeze(-1)
            LL += wi

        # TODO? more efficient & accurate to compute `torch.exp(-t*b + log_g)` since we directly sample & stored log_g
        LL = 1. + g * torch.exp(-LL * b)
        model = 1. / LL

        # For ordinal, compute likelihoods instead of cumulative distribution function
        if self.noise_model == 'ordinal':
            model = self.compute_likelihood_from_cdf(model)

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

    def compute_individual_ages_from_biomarker_values_tensorized_logistic_ordinal(self, value: torch.Tensor,
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

        feature : str (or None)
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
        if value.dim() != 2:
            raise LeaspyModelInputError(f"The biomarker value should be dim 2, not {value.dim()}!")

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
        deltas = self._get_deltas(None)
        delta_max = deltas[feat_ind].sum()
        ages_max = tau + (torch.exp(-xi) / v0) * ((g / (g + 1) ** 2) * torch.log(g) - wi + delta_max)

        timepoints = torch.linspace(ages_0.item(), ages_max.item(), 1000).unsqueeze(0)

        grid = self.compute_individual_tensorized_logistic(timepoints, individual_parameters, attribute_type=None)[:,:,feat_ind,:]

        MLE = grid.argmax(dim=-1)
        index_cross = (MLE.unsqueeze(1) >= value.unsqueeze(-1)).int().argmax(dim=-1)

        return timepoints[0,index_cross]

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

        if self.noise_model == 'ordinal':
            # add an extra dimension for the levels of the ordinal item
            LL = reparametrized_time.unsqueeze(-1)
            g = g.unsqueeze(-1)
            b = b.unsqueeze(-1)
            deltas = self._get_deltas(attribute_type)
            ordinal_scale = deltas.shape[-1] + 1
            deltas_ = torch.cat([torch.zeros((deltas.shape[0], 1)), deltas], dim=1)  # (features, max_level)
            deltas_ = deltas_.unsqueeze(0).unsqueeze(0)  # add (ind, timepoints) dimensions
            LL = LL - deltas_.cumsum(dim=-1)
            LL = v0.unsqueeze(-1) * LL

        else:
            LL = v0 * reparametrized_time

        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(a_matrix.t()).unsqueeze(-2) # unsqueeze for (n_timepoints)
            if self.noise_model == 'ordinal':
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

        if self.noise_model == 'ordinal':
            for param in derivatives:
                derivatives[param] = derivatives[param].unsqueeze(-2).repeat(1, 1, 1, self.ordinal_infos["max_level"], 1)

        for param in derivatives:
            derivatives[param] = c.unsqueeze(-1) * derivatives[param]

        # Compute derivative of the likelihood and not of the cdf
        if self.noise_model == 'ordinal':
            for param in derivatives:
                derivatives[param] = self.compute_likelihood_from_cdf(derivatives[param])

        # dict[param_name: str, torch.Tensor of shape(n_ind, n_tpts, n_fts, n_dims_param)]
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

        if self.noise_model == 'ordinal':
            if self.ordinal_infos['batch_deltas']:
                self.MCMC_toolbox['priors']['deltas_std'] = 0.1
            else:
                for feat in self.ordinal_infos["features"]:
                    self.MCMC_toolbox['priors'][f'deltas_{feat["name"]}_std'] = 0.1
            ord = self.ordinal_infos
        else:
            ord = None
        self.MCMC_toolbox['attributes'] = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension, ordinal_infos=ord)
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
        if self.noise_model == 'ordinal':
            if self.ordinal_infos['batch_deltas']:
                if any(c in L for c in ('deltas', 'all')):
                    values['deltas'] = realizations['deltas'].tensor_realizations
            else:
                for feat in self.ordinal_infos["features"]:
                    if any(c in L for c in ('deltas_'+feat["name"], 'all')) and self.source_dimension != 0:
                        values['deltas_'+feat["name"]] = realizations['deltas_'+feat["name"]].tensor_realizations

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

        if self.noise_model == 'ordinal':
            if self.ordinal_infos['batch_deltas']:
                sufficient_statistics['deltas'] = realizations['deltas'].tensor_realizations
            else:
                for feat in self.ordinal_infos["features"]:
                    sufficient_statistics['deltas_'+feat["name"]] = realizations['deltas_'+feat["name"]].tensor_realizations

        individual_parameters = self.get_param_from_real(realizations)

        data_reconstruction = self.compute_individual_tensorized(data.timepoints,
                                                                 individual_parameters,
                                                                 attribute_type='MCMC')

        if self.noise_model in ['gaussian_scalar', 'gaussian_diagonal']:
            data_reconstruction *= data.mask.float()  # speed-up computations

            norm_1 = data.values * data_reconstruction
            norm_2 = data_reconstruction * data_reconstruction

            sufficient_statistics['obs_x_reconstruction'] = norm_1  # .sum(dim=2) # no sum on features...
            sufficient_statistics['reconstruction_x_reconstruction'] = norm_2  # .sum(dim=2) # no sum on features...

        if self.noise_model in ['bernoulli', 'ordinal']:
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

        if self.noise_model == 'ordinal':
            if self.ordinal_infos['batch_deltas']:
                self.parameters['deltas'] = realizations['deltas'].tensor_realizations
            else:
                for feat in self.ordinal_infos["features"]:
                    self.parameters['deltas_'+feat["name"]] = realizations['deltas_'+feat["name"]].tensor_realizations

        xi = realizations['xi'].tensor_realizations
        # self.parameters['xi_mean'] = torch.mean(xi)  # fixed = 0 by design
        self.parameters['xi_std'] = torch.std(xi)
        tau = realizations['tau'].tensor_realizations
        self.parameters['tau_mean'] = torch.mean(tau)
        self.parameters['tau_std'] = torch.std(tau)

        # by design: sources_mean = 0., sources_std = 1.

        param_ind = self.get_param_from_real(realizations)

        # Should we really keep this ? cf #54 issue
        if self.noise_model in ['bernoulli', 'ordinal']:
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

        if self.noise_model == 'ordinal':
            if self.ordinal_infos['batch_deltas']:
                self.parameters['deltas'] = suff_stats['deltas']
            else:
                for feat in self.ordinal_infos["features"]:
                    self.parameters['deltas_'+feat["name"]] = suff_stats['deltas_'+feat["name"]]

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

        if self.noise_model in ['bernoulli', 'ordinal']:
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

        if self.noise_model == 'ordinal':
            variables_infos['v0']['scale'] = 0.1
            if self.ordinal_infos['batch_deltas']:
                # For each feature : create a sampler for deltas of size (nb_levels_of_the_feature)
                max_level = self.ordinal_infos["max_level"]
                deltas_infos = {
                    "name": "deltas",
                    "shape": torch.Size([self.dimension, max_level - 1]),
                    "type": "population",
                    "rv_type": "multigaussian",
                    "scale": .5,
                    "mask": self.ordinal_infos["mask"][0,0,:,1:], # cut the zero level
                }
                variables_infos['deltas'] = deltas_infos
            else:
                # Instead of a sampler for each feature, sample deltas for all features in one sampler class
                for feat in self.ordinal_infos["features"]:
                    deltas_infos = {
                        "name": "deltas_"+feat["name"],
                        "shape": torch.Size([feat["nb_levels"] - 1]),
                        "type": "population",
                        "rv_type": "gaussian",
                        "scale": .5,
                    }
                    variables_infos['deltas_'+feat["name"]] = deltas_infos

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
