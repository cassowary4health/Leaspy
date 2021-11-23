import torch

from .abstract_multivariate_link_model import AbstractMultivariateLinkModel
from .utils.attributes import AttributesFactory

from leaspy.utils.docs import doc_with_super, doc_with_
from leaspy.utils.subtypes import suffixed_method
from leaspy.exceptions import LeaspyModelInputError

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

    def compute_individual_tensorized_logistic_link(self, timepoints, ind_parameters, attribute_type=None):

        # Population parameters
        g, _, = self._get_attributes(attribute_type)
        g_plus_1 = 1. + g
        b = g_plus_1 * g_plus_1 / g

        # Individual parameters
        xi, tau, v0 = ind_parameters['xi'], ind_parameters['tau'], ind_parameters['v0']

        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)
        # Log likelihood computation
        reparametrized_time = reparametrized_time.unsqueeze(-1) # (n_individuals, n_timepoints, n_features)
        #v0 = v0.reshape(1, 1, -1) # not needed, automatic broadcast on last dim (n_features)
        #v0 = self.compute_individual_speeds(self.cofactors.transpose(0,1))

        LL = v0[:,None,:] * reparametrized_time

        # compute orthonormal basis and mixing matrix for every subject
        a_matrix = torch.zeros((self.dimension, self.source_dimension))
        if self.source_dimension != 0:
            sources = ind_parameters['sources'].reshape(-1, self.source_dimension)
            betas = self.get_beta(attribute_type)

            ortho_basis = torch.stack(self.compute_ortho_basis_indiv(v0, attribute_type))
            a_matrix = ortho_basis @ betas

            wi = (a_matrix @ sources[:,:,None]).squeeze()
            #wi = sources.matmul(a_matrix.t())
            #print(f"Shape of v0 {v0.shape}, sources {sources.shape}, betas {betas.shape}, ortho_basis {ortho_basis.shape}, a_matrix {a_matrix.shape}, wi {wi.shape}, LL {LL.shape}", flush=True) 
            LL += wi.unsqueeze(-2) # unsqueeze for (n_timepoints)
        LL = 1. + g * torch.exp(-LL * b)
        model = 1. / LL

        return model # (n_individuals, n_timepoints, n_features)


    def get_beta(self, attribute_type=None):
        if attribute_type is None:
            return self.attributes.betas
        elif attribute_type == 'MCMC':
            return self.MCMC_toolbox['attributes'].betas

    def compute_ortho_basis_indiv(self, v0, attribute_type=None):
        if attribute_type is None:
            return self.attributes._compute_orthonormal_basis_indiv(v0)
        elif attribute_type == 'MCMC':
            return self.MCMC_toolbox['attributes']._compute_orthonormal_basis_indiv(v0)

    @suffixed_method
    def compute_individual_ages_from_biomarker_values_tensorized(self, value: torch.Tensor,
                                                                 individual_parameters: dict, feature: str):
        pass

    @suffixed_method
    def compute_jacobian_tensorized(self, timepoints, ind_parameters, attribute_type=None):
        pass

    def get_intersept(self, variable_name):
        if variable_name == "v0":
            return self.parameters['link'][:-1, -1]
        elif variable_name == "tau_mean":
            return self.parameters['link'][-1, -1]

    def compute_individual_speeds(self, cofactors, attribute_type=None):
        """
            A (shape: n_features, n_cofactors + 1)
            cofactors (shape: n_cofactors, n_subjects
        """
        if attribute_type == 'model':
            link = self.parameters['link']
        else:
            _, link = self._get_attributes(attribute_type)
        link = link[:-1,:]
        return torch.exp(link @ torch.cat((cofactors, torch.ones(1, cofactors.shape[1], device=self.device))))

    def compute_individual_tau_means(self, cofactors, attribute_type=None):
        if attribute_type == 'model':
            link = self.parameters['link']
        else:
            _, link = self._get_attributes(attribute_type)
        link = link[-1, :]
        return link @ torch.cat((cofactors, torch.ones(1, cofactors.shape[1], device=self.device)))

    ##############################
    ### MCMC-related functions ###
    ##############################

    def initialize_MCMC_toolbox(self, set_v0_prior = False):
        self.MCMC_toolbox = {
            'priors': {'g_std': 0.01, 'v0_std': 0.01, 'betas_std': 0.01, 'link_std': 0.01}, # population parameters
            'attributes': AttributesFactory.attributes(self.name, self.dimension, self.source_dimension, self.device)
        }

        population_dictionary = self._create_dictionary_of_population_realizations()
        self.update_MCMC_toolbox(["all"], population_dictionary)

        # Initialize hyperpriors
        if set_v0_prior:
            self.MCMC_toolbox['priors']['v0_mean'] = self.parameters['v0'].clone().detach()
            self.MCMC_toolbox['priors']['s_v0'] = 0.1
            # same on g?

    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        L = name_of_the_variables_that_have_been_changed
        values = {}
        if any(c in L for c in ('g', 'all')):
            values['g'] = realizations['g'].tensor_realizations
        if any(c in L for c in ('link', 'all')):
            values['link'] = realizations['link'].tensor_realizations            
        if any(c in L for c in ('betas', 'all')) and self.source_dimension != 0:
            values['betas'] = realizations['betas'].tensor_realizations

        self.MCMC_toolbox['attributes'].update(name_of_the_variables_that_have_been_changed, values)

    def _center_xi_realizations(self, realizations):
        mean_xi = torch.mean(realizations['xi'].tensor_realizations)
        realizations['xi'].tensor_realizations = realizations['xi'].tensor_realizations - mean_xi
        realizations['v0'].tensor_realizations = realizations['v0'].tensor_realizations + mean_xi

        self.update_MCMC_toolbox(['v0'], realizations)
        return realizations

    def compute_sufficient_statistics(self, data, realizations):

        # <!> by doing this here, we change v0 and thus orthonormal basis and mixing matrix,
        #     the betas / sources are not related to the previous orthonormal basis...
        # realizations = self._center_xi_realizations(realizations)

        sufficient_statistics = {
            'g': realizations['g'].tensor_realizations,
            'link': realizations['link'].tensor_realizations,
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
        # realizations = self._center_xi_realizations(realizations)

        # Memoryless part of the algorithm
        self.parameters['g'] = realizations['g'].tensor_realizations.detach()
        self.parameters['link'] = realizations['link'].tensor_realizations.detach()

        if self.source_dimension != 0:
            self.parameters['betas'] = realizations['betas'].tensor_realizations.detach()
        xi = realizations['xi'].tensor_realizations.detach()
        # self.parameters['xi_mean'] = torch.mean(xi)
        self.parameters['xi_std'] = torch.std(xi)
        tau = realizations['tau'].tensor_realizations.detach()
        self.parameters['tau_mean'] = torch.mean(tau)
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
        self.parameters['g'] = suff_stats['g']
        self.parameters['link'] = suff_stats['link']

        if self.source_dimension != 0:
            self.parameters['betas'] = suff_stats['betas']

        tau_mean = self.parameters['tau_mean'].clone()
        tau_std_updt = torch.mean(suff_stats['tau_sqrd']) - 2 * tau_mean * torch.mean(suff_stats['tau'])
        self.parameters['tau_std'] = torch.sqrt(tau_std_updt + self.parameters['tau_mean'] ** 2)
        self.parameters['tau_mean'] = torch.mean(suff_stats['tau'])

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
        g_infos = {
            "name": "g",
            "shape": torch.Size([self.dimension]),
            "type": "population",
            "rv_type": "multigaussian"
        }

        link_infos = {
            "name": "link",
            "shape": self.link_shape,
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
            "link": link_infos,
            "v0": v0_infos,
            "tau": tau_infos,
            "tau_mean": tau_mean_infos,
            "xi": xi_infos,
        }        

        if self.source_dimension != 0:
            variables_infos['sources'] = sources_infos
            variables_infos['betas'] = betas_infos

        return variables_infos
