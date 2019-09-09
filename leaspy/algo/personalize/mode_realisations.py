from .abstract_personalize_algo import AbstractPersonalizeAlgo
from ..samplers.hmc_sampler import HMCSampler
from ..samplers.gibbs_sampler import GibbsSampler
from leaspy.utils.realizations.realization import Realization
import torch
import time

class ModeReal(AbstractPersonalizeAlgo):

    def __init__(self, settings):

        # Algorithm parameters
        super().__init__(settings)

    # TODO cloned --> factorize in a utils ???
    def _initialize_samplers(self, model, data):
        infos_variables = model.random_variable_informations()
        self.samplers = dict.fromkeys(infos_variables.keys())
        for variable, info in infos_variables.items():
            if info["type"] == "individual":
                if self.algo_parameters['sampler_ind'] == 'Gibbs':
                    self.samplers[variable] = GibbsSampler(info, data.n_individuals)
                else:
                    self.samplers[variable] = HMCSampler(info, data.n_individuals, self.algo_parameters['eps'])
            else:
                if self.algo_parameters['sampler_pop'] == 'Gibbs':
                    self.samplers[variable] = GibbsSampler(info, data.n_individuals)
                else:
                    self.samplers[variable] = HMCSampler(info, data.n_individuals, self.algo_parameters['eps'])

    def _get_individual_parameters(self, model, data):

        # Initialize realizations storage object
        realizations_history = []

        # Initialize samplers
        self._initialize_samplers(model, data)

        # initialize realizations
        realizations = model.get_realization_object(data.n_individuals)
        realizations.initialize_from_values(data.n_individuals, model)

        # Gibbs sample n_iter times
        for i in range(self.algo_parameters['n_iter']):
            for key in realizations.reals_ind_variable_names:
                self.samplers[key].sample(data, model, realizations, 1.0)

            # Append current realizations if burn in is finished
            if i > self.algo_parameters['n_burn_in_iter']:
                realizations_history.append(realizations.copy())

        # Get for each patient the realization that best fit
        attachments = torch.stack([model.compute_individual_attachment_tensorized(data, model.get_param_from_real(realizations), True) for realizations in realizations_history])

        # Indices min
        indices_min = torch.min(attachments, dim=0)

        # Compute mode of n_iter realizations for each individual variable
        mode_output = {}

        ind_var_names = model.get_individual_variable_name()
        infos = model.random_variable_informations()

        for ind_var_name in ind_var_names:
            mode_output[ind_var_name] = Realization.from_tensor(
                ind_var_name,
                infos[ind_var_name]["shape"],
                "individual",
                torch.stack([realizations_history[indices_min[1][i]][ind_var_name].tensor_realizations[i].clone() for i, idx in enumerate(data.indices)]))


        ind_parameters = model.get_param_from_real(mode_output) # TODO ordering between the ind variables, should not be the case

        return ind_parameters



        """


        for name_variable, info_variable in model.random_variable_informations().items():
            if info_variable['type'] == 'individual':
                mean_variable = torch.stack(
                    [realizations[name_variable].tensor_realizations for realizations in realizations_history]).mean(
                    dim=0).clone().detach()
                mean_output[name_variable] = mean_variable

        # Compute the attachment
        realizations = model.get_realization_object(data.n_individuals)
        for key, value in mean_output.items():
            realizations[key].tensor_realizations = value

        # Get individual realizations from realizations object
        param_ind = model.get_param_from_real(realizations)

        return param_ind
"""


