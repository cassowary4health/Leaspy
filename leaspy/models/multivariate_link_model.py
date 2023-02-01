import torch

from .abstract_multivariate_link_model import AbstractMultivariateLinkModel
from .utils.attributes import AttributesFactory

from leaspy.utils.docs import doc_with_super, doc_with_
from leaspy.utils.subtypes import suffixed_method
from leaspy.exceptions import LeaspyModelInputError

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either model values or model values + jacobians wrt individual parameters
# TODO refact? subclass or other proper code technique to extract model's concrete formulation depending on if linear, logistic, mixed log-lin, ...


@doc_with_super()
class MultivariateLinkModel(AbstractMultivariateLinkModel):
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
        'logistic_link': '_logistic_link',
    }

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.MCMC_toolbox['priors']['v0_std'] = None  # Value, Coef

        self._subtype_suffix = self._check_subtype()

    def _check_subtype(self):
        if self.name not in self.SUBTYPES_SUFFIXES.keys():
            raise LeaspyModelInputError(f'Multivariate model name should be among these valid sub-types: '
                                        f'{list(self.SUBTYPES_SUFFIXES.keys())}.')

        return self.SUBTYPES_SUFFIXES[self.name]

    def load_parameters(self, parameters):
        self.parameters = {}
        for k in parameters.keys():
            if k in ['mixing_matrix']:
                continue
            self.parameters[k] = torch.tensor(parameters[k], dtype=torch.float32)
        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension, self.device)
        self.attributes.update(['all'], self.parameters)

    @suffixed_method
    def compute_individual_tensorized(self, timepoints, ind_parameters, attribute_type=None):
        pass

    @staticmethod
    def time_reparametrization_link(timepoints: torch.FloatTensor, xi: torch.FloatTensor, tau: torch.FloatTensor, tau_mean) -> torch.FloatTensor:
        """
        Tensorized time reparametrization formula

        <!> Shapes of tensors must be compatible between them.

        Parameters
        ----------
        timepoints : :class:`torch.Tensor`
            Timepoints to reparametrize
        xi : :class:`torch.Tensor`
            Log-acceleration of individual(s)
        tau : :class:`torch.Tensor`
            Time-shift(s)

        Returns
        -------
        :class:`torch.Tensor` of same shape as `timepoints`
        """
        #print(f"Shapes xi {xi.shape}, timepoints {timepoints.shape}, tau_mean {tau_mean.shape}, tau {tau.shape}")
        return torch.exp(xi) * (timepoints - tau_mean - tau)

    def compute_individual_tensorized_logistic_link(self, timepoints, ind_parameters, attribute_type=None):

        # Individual parameters
        xi, tau, tau_mean, v0, g = ind_parameters['xi'], ind_parameters['tau'], ind_parameters['tau_mean'], ind_parameters['v0'], ind_parameters['g']

        g_plus_1 = 1. + g
        b = g_plus_1 * g_plus_1 / g

        reparametrized_time = self.time_reparametrization_link(timepoints, xi, tau, tau_mean)
        # Log likelihood computation
        reparametrized_time = reparametrized_time.unsqueeze(-1) # (n_individuals, n_timepoints, n_features)
        #v0 = v0.reshape(1, 1, -1) # not needed, automatic broadcast on last dim (n_features)
        #v0 = self.compute_individual_speeds(self.cofactors.transpose(0,1))

        #print("run")
        #print(v0.shape)
        #print(reparametrized_time.shape)
        LL = v0[:,None,:] * reparametrized_time

        # compute orthonormal basis and mixing matrix for every subject
        a_matrix = torch.zeros((self.dimension, self.source_dimension))
        if self.source_dimension != 0:
            sources = ind_parameters['sources'].reshape(-1, self.source_dimension)
            betas = self.get_beta(attribute_type)

            ortho_basis = torch.stack(self.compute_ortho_basis_indiv(g, v0, attribute_type))
            a_matrix = ortho_basis @ betas

            wi = (a_matrix @ sources[:,:,None]).squeeze()
            #wi = sources.matmul(a_matrix.t())
            #print(f"Shape of v0 {v0.shape}, sources {sources.shape}, betas {betas.shape}, ortho_basis {ortho_basis.shape}, a_matrix {a_matrix.shape}, wi {wi.shape}, LL {LL.shape}", flush=True) 
            LL += wi.unsqueeze(-2) # unsqueeze for (n_timepoints)
        LL = 1. + g.unsqueeze(-2) * torch.exp(-LL * b.unsqueeze(-2))
        model = 1. / LL

        return model # (n_individuals, n_timepoints, n_features)


    def get_beta(self, attribute_type=None):
        if attribute_type is None:
            return self.attributes.betas
        elif attribute_type == 'MCMC':
            return self.MCMC_toolbox['attributes'].betas

    def compute_ortho_basis_indiv(self, pos, v0, attribute_type=None):
        if attribute_type is None:
            return self.attributes._compute_orthonormal_basis_indiv(pos, v0)
        elif attribute_type == 'MCMC':
            return self.MCMC_toolbox['attributes']._compute_orthonormal_basis_indiv(pos, v0)

    @suffixed_method
    def compute_individual_ages_from_biomarker_values_tensorized(self, value: torch.Tensor,
                                                                 individual_parameters: dict, feature: str):
        pass

    @suffixed_method
    def compute_jacobian_tensorized(self, timepoints, ind_parameters, attribute_type=None):
        pass

    def compute_jacobian_tensorized_logistic_link(self, timepoints, ind_parameters, attribute_type=None):
        # Individual parameters
        xi, tau, tau_mean, v0, g = ind_parameters['xi'], ind_parameters['tau'], ind_parameters['tau_mean'], ind_parameters['v0'], ind_parameters['g']

        g_plus_1 = 1. + g
        b = g_plus_1 * g_plus_1 / g

        reparametrized_time = self.time_reparametrization_link(timepoints, xi, tau, tau_mean)
        reparametrized_time = reparametrized_time.unsqueeze(-1) # (n_individuals, n_timepoints, n_features)

        LL = v0[:,None,:] * reparametrized_time

        a_matrix = torch.zeros((self.dimension, self.source_dimension))
        if self.source_dimension != 0:
            sources = ind_parameters['sources'].reshape(-1, self.source_dimension)
            betas = self.get_beta(attribute_type)

            ortho_basis = torch.stack(self.compute_ortho_basis_indiv(g, v0, attribute_type))
            a_matrix = ortho_basis @ betas

            wi = (a_matrix @ sources[:,:,None]).squeeze()
            LL += wi.unsqueeze(-2) # unsqueeze for (n_timepoints)
        LL = 1. + g.unsqueeze(-2) * torch.exp(-LL * b.unsqueeze(-2))
        model = 1. / LL

        c = model * (1. - model) * b
        alpha = torch.exp(xi).reshape(-1, 1, 1)

        derivatives = {
            'xi': (c * v0 * reparametrized_time).unsqueeze(-1),
            'tau': (c * -v0 * alpha).unsqueeze(-1),
        }
        if self.source_dimension > 0:
            derivatives['sources'] = c.unsqueeze(-1) * a_matrix.expand((1,1,-1,-1))

        # dict[param_name: str, torch.Tensor of shape(n_ind, n_tpts, n_fts, n_dims_param)]
        return derivatives

    def get_intersept(self, variable_name):
        if self.link_type == 'linear':
            if variable_name == "v0":
                return self.parameters['link_v0'][:, -1]
            if variable_name == "g":
                return self.parameters['link_g'][:, -1]            
            elif variable_name == "tau_mean" or variable_name == 't_mean':
                return self.parameters['link_t_mean'][:, -1]
        elif self.link_type == 'perceptron':
            if variable_name == "v0":
                return self.parameters['link_v0'][-self.dimension:]
            if variable_name == "g":
                return self.parameters['link_g'][-self.dimension:]
            elif variable_name == "tau_mean" or variable_name == 't_mean':
                return self.parameters['link_t_mean'][-1]

    def _get_link_positions(model):
        class LinkPosition(nn.Module):
            def __init__(self):
                super(LinkPosition, self).__init__()

                intermediate_layer = 3
                self.layer_1 = nn.Linear(model.cofactors_dimension, intermediate_layer)
                self.layer_2 = nn.Linear(intermediate_layer, model.dimension)

            def forward(self, x):
                x = self.layer_1(x)
                x = F.relu(x)
                x = self.layer_2(x)
                return x
        return LinkPosition

    def fill_nn(self, network, params):
        params = params.flatten()

        for param in network.parameters():
            param_size = torch.numel(param)
            param.data = torch.nn.Parameter(params[:param_size].reshape(param.shape))
            params = params[param_size:] 

    def compute_individual_positions(self, cofactors, attribute_type=None):
        if attribute_type == 'model':
            link_g = self.parameters['link_g']
        else:
            links = self._get_attributes(attribute_type)
            link_g = links['g']

        if self.link_type == 'linear':
            return torch.exp(link_g @ torch.cat((cofactors, torch.ones(1, cofactors.shape[1], device=self.device))))
        elif self.link_type == 'perceptron':
            model = self._get_link_positions()()
            self.fill_nn(model, link_g)

            return torch.exp(model.forward(cofactors.transpose(0,1)).transpose(0,1))

    def _get_link_speed(model):
        class LinkSpeed(nn.Module):
            def __init__(self):
                super(LinkSpeed, self).__init__()

                intermediate_layer = 3
                self.layer_1 = nn.Linear(model.cofactors_dimension, intermediate_layer)
                self.layer_2 = nn.Linear(intermediate_layer, model.dimension)

            def forward(self, x):
                x = self.layer_1(x)
                x = F.relu(x)
                x = self.layer_2(x)
                return x
        return LinkSpeed

    def compute_individual_speeds(self, cofactors, attribute_type=None):
        """
            A (shape: n_features, n_cofactors + 1)
            cofactors (shape: n_cofactors, n_subjects
        """
        if attribute_type == 'model':
            link_v0 = self.parameters['link_v0']
        else:
            links = self._get_attributes(attribute_type)
            link_v0 = links['v0']

        if self.link_type == 'linear':
            return torch.exp(link_v0 @ torch.cat((cofactors, torch.ones(1, cofactors.shape[1], device=self.device))))
        elif self.link_type == 'perceptron':
            model = self._get_link_speed()()
            self.fill_nn(model, link_v0)

            return torch.exp(model.forward(cofactors.transpose(0,1)).transpose(0,1))

    def _get_link_time(model):
        class LinkTime(nn.Module):
            def __init__(self):
                super(LinkTime, self).__init__()

                intermediate_layer = 3
                self.layer_1 = nn.Linear(model.cofactors_dimension, intermediate_layer)
                self.layer_2 = nn.Linear(intermediate_layer, 1)

            def forward(self, x):
                x = self.layer_1(x)
                x = F.relu(x)
                x = self.layer_2(x)
                return x
        return LinkTime

    def compute_individual_tau_means(self, cofactors, attribute_type=None):
        if attribute_type == 'model':
            link_t_mean = self.parameters['link_t_mean']
        else:
            links = self._get_attributes(attribute_type)
            link_t_mean = links['t_mean']

        if self.link_type == 'linear':
            return (link_t_mean @ torch.cat((cofactors, torch.ones(1, cofactors.shape[1], device=self.device)))).squeeze(-1)
        elif self.link_type == 'perceptron':
            model = self._get_link_time()()
            self.fill_nn(model, link_t_mean)

            return model.forward(cofactors.transpose(0,1)).transpose(0,1)

    ##############################
    ### MCMC-related functions ###
    ##############################

    def initialize_MCMC_toolbox(self, set_v0_prior = False, set_link_prior=True, set_std_prior=True):
        self.MCMC_toolbox = {
            'priors': {'v0_std': 0.01, 'betas_std': 0.01, 'link_v0_std': 0.01, 'link_g_std': 0.01, 'link_t_mean_std': 0.01}, # population parameters
            'attributes': AttributesFactory.attributes(self.name, self.dimension, self.source_dimension, self.device)
        }

        population_dictionary = self._create_dictionary_of_population_realizations()
        self.update_MCMC_toolbox(["all"], population_dictionary)

        # Initialize hyperpriors
        if set_link_prior:
            self.MCMC_toolbox['priors']['link_v0_mean'] = self.parameters['link_v0'].clone().detach()
            self.MCMC_toolbox['priors']['s_link_v0'] = 0.1

            self.MCMC_toolbox['priors']['link_g_mean'] = self.parameters['link_g'].clone().detach()
            self.MCMC_toolbox['priors']['s_link_g'] = 0.1
            self.MCMC_toolbox['priors']['link_t_mean_mean'] = self.parameters['link_t_mean'].clone().detach()
            self.MCMC_toolbox['priors']['s_link_t_mean'] = 0.1

        if set_std_prior:
            pass
            #self.MCMC_toolbox['priors']['sigma_tau_zero'] = 5.
            #self.MCMC_toolbox['priors']['mass_sigma_tau_zero'] = 500.
        if set_v0_prior:
            self.MCMC_toolbox['priors']['v0_mean'] = self.parameters['v0'].clone().detach()
            self.MCMC_toolbox['priors']['s_v0'] = 0.1
            # same on g?

    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        L = name_of_the_variables_that_have_been_changed
        values = {}
        if any(c in L for c in ('link_g', 'all')):
            values['link_g'] = realizations['link_g'].tensor_realizations
        if any(c in L for c in ('link_v0', 'all')):
            values['link_v0'] = realizations['link_v0'].tensor_realizations
        if any(c in L for c in ('link_t_mean', 'all')):
            values['link_t_mean'] = realizations['link_t_mean'].tensor_realizations 
        if any(c in L for c in ('betas', 'all')) and self.source_dimension != 0:
            values['betas'] = realizations['betas'].tensor_realizations

        self.MCMC_toolbox['attributes'].update(name_of_the_variables_that_have_been_changed, values)

    def _center_xi_realizations(self, realizations):
        mean_xi = torch.mean(realizations['xi'].tensor_realizations)
        realizations['xi'].tensor_realizations = realizations['xi'].tensor_realizations - mean_xi

        if self.link_type == 'linear':
            realizations['link_v0'].tensor_realizations[:,-1] = realizations['link_v0'].tensor_realizations[:,-1] + mean_xi
        elif self.link_type == 'perceptron':
            realizations['link_v0'].tensor_realizations[-self.dimension:] = realizations['link_v0'].tensor_realizations[-self.dimension:] + mean_xi

        self.update_MCMC_toolbox(['link_v0'], realizations)
        return realizations

    def _center_tau_realizations(self, realizations):
        mean_tau = torch.mean(realizations['tau'].tensor_realizations)
        #print(f"Mean tau : {mean_tau}")
        realizations['tau'].tensor_realizations = realizations['tau'].tensor_realizations - mean_tau
        if self.link_type == 'linear':
            realizations['link_t_mean'].tensor_realizations[:,-1] = realizations['link_t_mean'].tensor_realizations[:,-1] + mean_tau
        elif self.link_type == 'perceptron':
            realizations['link_t_mean'].tensor_realizations[-1] = realizations['link_t_mean'].tensor_realizations[-1] + mean_tau

        self.update_MCMC_toolbox(['link_t_mean'], realizations)
        return realizations

    def compute_sufficient_statistics(self, data, realizations):

        # <!> by doing this here, we change v0 and thus orthonormal basis and mixing matrix,
        #     the betas / sources are not related to the previous orthonormal basis...
        realizations = self._center_tau_realizations(realizations)
        realizations = self._center_xi_realizations(realizations)

        sufficient_statistics = {
            #'g': realizations['g'].tensor_realizations,
            'link_v0': realizations['link_v0'].tensor_realizations,
            'link_g': realizations['link_g'].tensor_realizations,
            'link_t_mean': realizations['link_t_mean'].tensor_realizations,
            'v0': realizations['v0'].tensor_realizations,
            'tau': realizations['tau'].tensor_realizations,
            'tau_sqrd': torch.pow(realizations['tau'].tensor_realizations, 2),
            'xi': realizations['xi'].tensor_realizations,
            'xi_sqrd': torch.pow(realizations['xi'].tensor_realizations, 2)
        }
        if self.source_dimension != 0:
            sufficient_statistics['betas'] = realizations['betas'].tensor_realizations

        ind_parameters = self.get_param_from_real(realizations)

        data_reconstruction = self.compute_individual_tensorized(data.timepoints,
                                                                 ind_parameters,
                                                                 attribute_type='MCMC')

        data_reconstruction *= data.mask.float()  # speed-up computations

        norm_1 = data.values * data_reconstruction
        norm_2 = data_reconstruction * data_reconstruction

        sufficient_statistics['obs_x_reconstruction'] = norm_1  # .sum(dim=2) # no sum on features...
        sufficient_statistics['reconstruction_x_reconstruction'] = norm_2  # .sum(dim=2) # no sum on features...

        if self.loss == 'crossentropy':
            sufficient_statistics['crossentropy'] = self.compute_individual_attachment_tensorized(data, ind_parameters,
                                                                                                  attribute_type="MCMC")

        return sufficient_statistics

    def update_model_parameters_burn_in(self, data, realizations):

        # <!> by doing this here, we change v0 and thus orthonormal basis and mixing matrix,
        #     the betas / sources are not related to the previous orthonormal basis...
        realizations = self._center_tau_realizations(realizations)
        realizations = self._center_xi_realizations(realizations)

        # Memoryless part of the algorithm
        #self.parameters['g'] = realizations['g'].tensor_realizations.detach()

        link_v0_emp = realizations['link_v0'].tensor_realizations.detach()
        link_g_emp = realizations['link_g'].tensor_realizations.detach()
        link_t_mean_emp = realizations['link_t_mean'].tensor_realizations.detach()

        if self.MCMC_toolbox['priors'].get('link_v0_mean', None) is not None:
            link_v0_mean = self.MCMC_toolbox['priors']['link_v0_mean']
            s_link_v0 = self.MCMC_toolbox['priors']['s_link_v0']
            sigma_link_v0 = self.MCMC_toolbox['priors']['link_v0_std']
            self.parameters['link_v0'] = (1 / (1 / (s_link_v0 ** 2) + 1 / (sigma_link_v0 ** 2))) * (
            link_v0_emp / (sigma_link_v0 ** 2) + link_v0_mean / (s_link_v0 ** 2))
        else:
            self.parameters['link_v0'] = link_v0_emp


        if self.MCMC_toolbox['priors'].get('link_g_mean', None) is not None:
            link_g_mean = self.MCMC_toolbox['priors']['link_g_mean']
            s_link_g = self.MCMC_toolbox['priors']['s_link_g']
            sigma_link_g = self.MCMC_toolbox['priors']['link_g_std']
            self.parameters['link_g'] = (1 / (1 / (s_link_g ** 2) + 1 / (sigma_link_g ** 2))) * (
            link_g_emp / (sigma_link_g ** 2) + link_g_mean / (s_link_g ** 2))
        else:
            self.parameters['link_g'] = link_g_emp

        link_t_mean_emp = realizations['link_t_mean'].tensor_realizations.detach()
        if self.MCMC_toolbox['priors'].get('link_t_mean_mean', None) is not None:
             link_t_mean_mean = self.MCMC_toolbox['priors']['link_t_mean_mean']
             s_link_t_mean = self.MCMC_toolbox['priors']['s_link_t_mean']
             sigma_link_t_mean = self.MCMC_toolbox['priors']['link_t_mean_std']
             self.parameters['link_t_mean'] = (1 / (1 / (s_link_t_mean ** 2) + 1 / (sigma_link_t_mean ** 2))) * (
             link_t_mean_emp / (sigma_link_t_mean ** 2) + link_t_mean_mean / (s_link_t_mean ** 2))
        else:
            self.parameters['link_t_mean'] = link_t_mean_emp

        if self.source_dimension != 0:
            self.parameters['betas'] = realizations['betas'].tensor_realizations.detach()
        xi = realizations['xi'].tensor_realizations.detach()
        # self.parameters['xi_mean'] = torch.mean(xi)
        self.parameters['xi_std'] = torch.std(xi)
        tau = realizations['tau'].tensor_realizations.detach()
        #self.parameters['tau_mean'] = torch.mean(tau)

        if self.MCMC_toolbox['priors'].get('sigma_tau_zero', None) is not None:
            self.parameters['tau_std'] = torch.sqrt((1. / (data.n_individuals + self.MCMC_toolbox['priors'].get('mass_sigma_tau_zero'))) * (torch.sum((tau - self.parameters['tau_mean'])**2) + self.MCMC_toolbox['priors'].get('mass_sigma_tau_zero') * self.MCMC_toolbox['priors'].get('sigma_tau_zero')))
            pass
        else:
            self.parameters['tau_std'] = torch.std(tau)

        param_ind = self.get_param_from_real(realizations)
        # TODO : Why is it MCMC-SAEM? SHouldn't it be computed with the parameters?
        if 'diag_noise' in self.loss:
            squared_diff_per_ft = self.compute_sum_squared_per_ft_tensorized(data, param_ind, attribute_type='MCMC').sum(dim=0)  # sum on individuals
            self.parameters['noise_std'] = torch.sqrt(squared_diff_per_ft / data.n_observations_per_ft.float())
        else:
            squared_diff = self.compute_sum_squared_tensorized(data, param_ind, attribute_type='MCMC').sum()  # sum on individuals
            self.parameters['noise_std'] = torch.sqrt(squared_diff / data.n_observations)

        if self.loss == 'crossentropy':
            self.parameters['crossentropy'] = self.compute_individual_attachment_tensorized(data, param_ind,
                                                                                            attribute_type="MCMC").sum()

        # TODO : This is just for debugging of linear
        # data_reconstruction = self.compute_individual_tensorized(data.timepoints,
        #                                                         self.get_param_from_real(realizations),
        #                                                         attribute_type='MCMC')
        # norm_0 = data.values * data.values * data.mask.float()
        # norm_1 = data.values * data_reconstruction * data.mask.float()
        # norm_2 = data_reconstruction * data_reconstruction * data.mask.float()
        # S1 = torch.sum(torch.sum(norm_0, dim=2))
        # S2 = torch.sum(torch.sum(norm_1, dim=2))
        # S3 = torch.sum(torch.sum(norm_2, dim=2))

        # print("During burn-in : ", torch.sqrt((S1 - 2. * S2 + S3) / (data.dimension * data.n_visits)),
        #       torch.sqrt(squared_diff / (data.n_visits * data.dimension)))

        # Stochastic sufficient statistics used to update the parameters of the model

    def update_model_parameters_normal(self, data, suff_stats):
        # TODO with Raphael : check the SS, especially the issue with mean(xi) and v_k
        # TODO : 1. Learn the mean of xi and v_k
        # TODO : 2. Set the mean of xi to 0 and add it to the mean of V_k
        #self.parameters['g'] = suff_stats['g']
        self.parameters['link_v0'] = suff_stats['link_v0']
        self.parameters['link_g'] = suff_stats['link_g']
        self.parameters['link_t_mean'] = suff_stats['link_t_mean']

        if self.source_dimension != 0:
            self.parameters['betas'] = suff_stats['betas']

        #tau_mean = self.parameters['tau_mean'].clone()
        #tau_mean = self.get_intersept("tau_mean").squeeze(-1)
        tau_mean = torch.tensor(0., dtype=torch.float32)
        tau_std_updt = torch.mean(suff_stats['tau_sqrd']) - 2 * tau_mean * torch.mean(suff_stats['tau'])
        self.parameters['tau_std'] = torch.sqrt(tau_std_updt + tau_mean ** 2)
        #self.parameters['tau_mean'] = torch.mean(suff_stats['tau'])

        xi_mean = self.parameters['xi_mean']
        xi_std_updt = torch.mean(suff_stats['xi_sqrd']) - 2 * xi_mean * torch.mean(suff_stats['xi'])
        self.parameters['xi_std'] = torch.sqrt(xi_std_updt + self.parameters['xi_mean'] ** 2)
        # self.parameters['xi_mean'] = torch.mean(suff_stats['xi'])

        if 'diag_noise' in self.loss:
            # keep feature dependence on feature to update diagonal noise (1 free param per feature)
            S1 = data.L2_norm_per_ft
            S2 = suff_stats['obs_x_reconstruction'].sum(dim=(0, 1))
            S3 = suff_stats['reconstruction_x_reconstruction'].sum(dim=(0, 1))

            self.parameters['noise_std'] = torch.sqrt((S1 - 2. * S2 + S3) / data.n_observations_per_ft.float())
            # tensor 1D, shape (dimension,)
        else: # scalar noise (same for all features)
            S1 = data.L2_norm
            S2 = suff_stats['obs_x_reconstruction'].sum()
            S3 = suff_stats['reconstruction_x_reconstruction'].sum()

            self.parameters['noise_std'] = torch.sqrt((S1 - 2. * S2 + S3) / data.n_observations)

        if self.loss == 'crossentropy':
            self.parameters['crossentropy'] = suff_stats['crossentropy'].sum()

        # print("After burn-in : ", torch.sqrt((S1 - 2. * S2 + S3) / (data.dimension * data.n_visits)))

    ###################################
    ### Random Variable Information ###
    ###################################

    def random_variable_informations(self):

        # --- Population variables
        link_v0_infos = {
            "name": "link_v0",
            "shape": self.link_v0_shape,
            "type": "population",
            "rv_type": "multigaussian",
        }
        link_g_infos = {
            "name": "link_g",
            "shape": self.link_g_shape,
            "type": "population",
            "rv_type": "multigaussian",
        }
        link_t_mean_infos = {
            "name": "link_t_mean",
            "shape": self.link_t_mean_shape,
            "type": "population",
            "rv_type": "multigaussian",
        }

        betas_infos = {
            "name": "betas",
            "shape": torch.Size([self.dimension - 1, self.source_dimension]),
            "type": "population",
            "rv_type": "multigaussian"
        }

        # --- Individual variables

        v0_infos = {
            "name": "v0",
            "shape": torch.Size([self.dimension]),
            "type": "individual",
            "rv_type": "linked"
        }
        g_infos = {
            "name": "g",
            "shape": torch.Size([self.dimension]),
            "type": "individual",
            "rv_type": "linked"
        }

        tau_infos = {
            "name": "tau",
            "shape": torch.Size([1]),
            "type": "individual",
            "rv_type": "gaussian"
        }

        tau_mean_infos = {
             "name": "tau_mean",
             "shape": torch.Size([1]),
             "type": "individual",
             "rv_type": "linked"
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
            "link_v0": link_v0_infos,
            "link_g": link_g_infos,
            "link_t_mean": link_t_mean_infos,
            "v0": v0_infos,
            "tau": tau_infos,
            "tau_mean": tau_mean_infos,
            "xi": xi_infos,
        }        

        if self.source_dimension != 0:
            variables_infos['sources'] = sources_infos
            variables_infos['betas'] = betas_infos

        return variables_infos
