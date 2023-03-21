import torch

from leaspy.models.abstract_multivariate_model import AbstractMultivariateModel
from leaspy.models.utils.attributes.logistic_parallel_attributes import LogisticParallelAttributes
from leaspy.utils.typing import DictParamsTorch
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
        self.update_MCMC_toolbox({"all"}, population_dictionary)

    def update_MCMC_toolbox(self, vars_to_update: set, realizations):
        values = {}
        update_all = 'all' in vars_to_update
        if update_all or 'g' in vars_to_update:
            values['g'] = realizations['g'].tensor_realizations
        if update_all or 'deltas' in vars_to_update:
            values['deltas'] = realizations['deltas'].tensor_realizations
        if self.source_dimension != 0 and (update_all or 'betas' in vars_to_update):
            values['betas'] = realizations['betas'].tensor_realizations

        self.MCMC_toolbox['attributes'].update(vars_to_update, values)

    def compute_model_sufficient_statistics(self, data, realizations):

        # unlink all sufficient statistics from updates in realizations!
        realizations = realizations.clone_realizations()

        sufficient_statistics = {
            param: realizations[param].tensor_realizations for param in ("g", "deltas", "tau", "xi")
        }
        if self.source_dimension != 0:
            sufficient_statistics['betas'] = realizations['betas'].tensor_realizations
        for param in ("tau", "xi"):
            sufficient_statistics[f"{param}_sqrd"] = torch.pow(realizations[param].tensor_realizations, 2)

        return sufficient_statistics

    def update_model_parameters_burn_in(self, data, sufficient_statistics):

        for param in ("g", "deltas"):
            self.parameters[param] = sufficient_statistics[param]
        if self.source_dimension != 0:
            self.parameters['betas'] = sufficient_statistics['betas']
        for param in ("xi", "tau"):
            param_realizations = sufficient_statistics[param]
            self.parameters[f"{param}_mean"] = torch.mean(param_realizations)
            self.parameters[f"{param}_std"] = torch.std(param_realizations)

    def update_model_parameters_normal(self, data, sufficient_statistics: DictParamsTorch) -> None:
        # TODO? factorize `update_model_parameters_***` methods?
        from .utilities import compute_std_from_variance

        for param in ("g", "deltas"):
            self.parameters[param] = sufficient_statistics[param]
        if self.source_dimension != 0:
            self.parameters['betas'] = sufficient_statistics['betas']

        for param in ("tau", "xi"):
            param_old_mean = self.parameters[f"{param}_mean"]
            param_cur_mean = torch.mean(sufficient_statistics[param])
            param_variance_update = (
                torch.mean(sufficient_statistics[f"{param}_sqrd"]) -
                2. * param_old_mean * param_cur_mean
            )
            param_variance = param_variance_update + param_old_mean ** 2
            self.parameters[f"{param}_std"] = compute_std_from_variance(param_variance, varname=f"{param}_std")
            self.parameters[f"{param}_mean"] = param_cur_mean

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
