import torch

from leaspy.models.abstract_multivariate_model import AbstractMultivariateModel
from leaspy.models.utils.attributes.logistic_parallel_attributes import LogisticParallelAttributes
from leaspy.models.utils.noise_model import NoiseModel

from leaspy.utils.docs import doc_with_super


@doc_with_super()
class MultivariateParallelModel(AbstractMultivariateModel):
    """
    Logistic model for multiple variables of interest, imposing same average evolution pace for all variables (logistic curves are only time-shifted).

    Parameters
    ----------
    name : str
        Name of the model
    **kwargs
        Hyperparameters of the model
    """
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.parameters["deltas"] = None
        self.MCMC_toolbox['priors']['deltas_std'] = None

    def load_parameters(self, parameters):
        # TODO? Move this method in higher level class AbstractMultivariateModel? (<!> Attributes class)
        self.parameters = {}
        for k in parameters.keys():
            if k in ['mixing_matrix']:
                continue
            self.parameters[k] = torch.tensor(parameters[k])

        # derive the model attributes from model parameters upon reloading of model
        self.attributes = LogisticParallelAttributes(self.name, self.dimension, self.source_dimension)
        self.attributes.update(['all'], self.parameters)

    def compute_individual_tensorized(self, timepoints, individual_parameters, *, attribute_type=None):

        # Population parameters
        g, deltas, mixing_matrix = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1) # (n_individuals, n_timepoints, -> n_features)

        # Model expected value
        t = reparametrized_time + deltas
        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(mixing_matrix.t())  # (n_individuals, n_features)
            g_deltas_exp = g * torch.exp(-deltas)
            b = (1. + g_deltas_exp) ** 2 / g_deltas_exp
            t += (b * wi).unsqueeze(-2)  # to get n_timepoints dimension
        model = 1. / (1. + g * torch.exp(-t))

        return model

    def compute_jacobian_tensorized(self, timepoints, individual_parameters, *, attribute_type=None):
        # TODO: refact highly inefficient (many duplicated code from `compute_individual_tensorized`)

        # Population parameters
        g, deltas, mixing_matrix = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1) # (n_individuals, n_timepoints, -> n_features)
        alpha = torch.exp(xi).reshape(-1, 1, 1)

        # Model expected value
        t = reparametrized_time + deltas
        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(mixing_matrix.t())  # (n_individuals, n_features)
            g_deltas_exp = g * torch.exp(-deltas)
            b = (1. + g_deltas_exp) ** 2 / g_deltas_exp  # (n_features, )
            t += (b * wi).unsqueeze(-2)  # to get n_timepoints dimension
        model = 1. / (1. + g * torch.exp(-t))

        # Jacobian of model expected value w.r.t. individual parameters
        c = model * (1. - model)

        derivatives = {
            'xi': (c * reparametrized_time).unsqueeze(-1),
            'tau': (c * -alpha).unsqueeze(-1),
        }
        if self.source_dimension > 0:
            b = b.reshape((1, 1, -1, 1)) # n_features is third
            derivatives['sources'] = c.unsqueeze(-1) * b * mixing_matrix.expand((1, 1, -1, -1))

        return derivatives

    def compute_individual_ages_from_biomarker_values_tensorized(self, value, individual_parameters, feature):
        raise NotImplementedError("Open an issue on Gitlab if needed.")  # pragma: no cover

    ##############################
    ### MCMC-related functions ###
    ##############################

    def initialize_MCMC_toolbox(self):
        self.MCMC_toolbox = {
            'priors': {'g_std': 0.01, 'deltas_std': 0.01, 'betas_std': 0.01}, # population parameters
            'attributes': LogisticParallelAttributes(self.name, self.dimension, self.source_dimension)
        }

        population_dictionary = self._create_dictionary_of_population_realizations()
        self.update_MCMC_toolbox(["all"], population_dictionary)

    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        L = name_of_the_variables_that_have_been_changed
        values = {}
        if any(c in L for c in ('g', 'all')):
            values['g'] = realizations['g'].tensor_realizations
        if any(c in L for c in ('deltas', 'all')):
            values['deltas'] = realizations['deltas'].tensor_realizations
        if any(c in L for c in ('betas', 'all')) and self.source_dimension != 0:
            values['betas'] = realizations['betas'].tensor_realizations

        self.MCMC_toolbox['attributes'].update(L, values)

    def compute_sufficient_statistics(self, data, realizations):

        # unlink all sufficient statistics from updates in realizations!
        realizations = realizations.clone_realizations()

        sufficient_statistics = {}
        sufficient_statistics['g'] = realizations['g'].tensor_realizations
        sufficient_statistics['deltas'] = realizations['deltas'].tensor_realizations
        if self.source_dimension != 0:
            sufficient_statistics['betas'] = realizations['betas'].tensor_realizations
        sufficient_statistics['tau'] = realizations['tau'].tensor_realizations
        sufficient_statistics['tau_sqrd'] = torch.pow(realizations['tau'].tensor_realizations, 2)
        sufficient_statistics['xi'] = realizations['xi'].tensor_realizations
        sufficient_statistics['xi_sqrd'] = torch.pow(realizations['xi'].tensor_realizations, 2)

        individual_parameters = self.get_param_from_real(realizations)

        data_reconstruction = self.compute_individual_tensorized(data.timepoints,
                                                                 individual_parameters,
                                                                 attribute_type='MCMC')
        data_reconstruction *= data.mask.float() # speed-up computations

        norm_1 = data.values * data_reconstruction #* data.mask.float()
        norm_2 = data_reconstruction * data_reconstruction #* data.mask.float()

        sufficient_statistics['obs_x_reconstruction'] = norm_1 #.sum(dim=2) # no sum on features...
        sufficient_statistics['reconstruction_x_reconstruction'] = norm_2 #.sum(dim=2) # no sum on features...

        if self.noise_model == 'bernoulli':
            sufficient_statistics['crossentropy'] = self.compute_individual_attachment_tensorized(data, individual_parameters,
                                                                                                  attribute_type='MCMC')

        return sufficient_statistics

    def update_model_parameters_burn_in(self, data, realizations):

        # unlink model parameters from updates in realizations!
        realizations = realizations.clone_realizations()

        self.parameters['g'] = realizations['g'].tensor_realizations
        self.parameters['deltas'] = realizations['deltas'].tensor_realizations
        if self.source_dimension != 0:
            self.parameters['betas'] = realizations['betas'].tensor_realizations
        xi = realizations['xi'].tensor_realizations
        self.parameters['xi_mean'] = torch.mean(xi)
        self.parameters['xi_std'] = torch.std(xi)
        tau = realizations['tau'].tensor_realizations
        self.parameters['tau_mean'] = torch.mean(tau)
        self.parameters['tau_std'] = torch.std(tau)

        param_ind = self.get_param_from_real(realizations)
        self.parameters['noise_std'] = NoiseModel.rmse_model(self, data, param_ind, attribute_type='MCMC')

        if self.noise_model == 'bernoulli':
            self.parameters['crossentropy'] = self.compute_individual_attachment_tensorized(data, param_ind,
                                                                                            attribute_type='MCMC').sum()

    def update_model_parameters_normal(self, data, suff_stats):

        self.parameters['g'] = suff_stats['g']
        self.parameters['deltas'] = suff_stats['deltas']
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
        self.parameters['xi_mean'] = torch.mean(suff_stats['xi'])

        # TODO: same as MultivariateModel, should we factorize code?
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

        ## Population variables
        g_infos = {
            "name": "g",
            "shape": torch.Size([1]),
            "type": "population",
            "rv_type": "multigaussian"
        }
        deltas_infos = {
            "name": "deltas",
            "shape": torch.Size([self.dimension - 1]),
            "type": "population",
            "rv_type": "multigaussian",
            "scale": 1.  # cf. GibbsSampler
        }
        betas_infos = {
            "name": "betas",
            "shape": torch.Size([self.dimension - 1, self.source_dimension]),
            "type": "population",
            "rv_type": "multigaussian",
            "scale": .5  # cf. GibbsSampler
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
