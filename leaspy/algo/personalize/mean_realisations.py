import warnings

import torch

from leaspy.algo.personalize.abstract_personalize_algo import AbstractPersonalizeAlgo
from leaspy.algo.utils.samplers import AlgoWithSamplersMixin
from leaspy.io.outputs.individual_parameters import IndividualParameters


class MeanReal(AlgoWithSamplersMixin, AbstractPersonalizeAlgo):
    """
    Sampler based algorithm, individual parameters are derived as the mean realization for `n_samples` samplings.

    TODO many stuff is duplicated between this class & mean_real (& other mcmc stuff) --> refactorize???
    TODO BUGFIX? temperature is never updated here unlike in fit (so only algo_parameters['annealing']['initial_temperature'] will be used)

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        Settings of the algorithm.
    """

    name = 'mean_real'

    def _initialize_annealing(self):
        if self.algo_parameters['annealing']['do_annealing']:

            warnings.warn(f'Annealing is currently not implemented for `{self.name}` algorithm, '
                          'explicitely disable it in the algorithm settings to remove this warning.')

            if self.algo_parameters['annealing']['n_iter'] is None:
                self.algo_parameters['annealing']['n_iter'] = int(self.algo_parameters['n_iter'] / 2)

        # Etienne: This is misleading because it will be executed even if no annealing
        self.temperature = self.algo_parameters['annealing']['initial_temperature']
        self.temperature_inv = 1 / self.temperature

    def _get_individual_parameters(self, model, data):

        # Initialize realizations storage object
        realizations_history = []

        # Where are not in a calibration any more so attribute_type=None (not using MCMC toolbox, since undefined!)
        computation_kws = dict(attribute_type=None)

        # Initialize samplers
        self._initialize_samplers(model, data)

        # Initialize Annealing
        self._initialize_annealing()

        # Initialize realizations, but really around their actual values for individual parameters!
        # TODO? is it really needed to restrict to that point individual parameters std-dev?
        realizations = model.initialize_realizations_for_model(data.n_individuals, scale_individual=0.01)
        ind_vars_names = realizations.reals_ind_variable_names

        # Gibbs sample n_iter times (only on individual parameters)
        for i in range(self.algo_parameters['n_iter']):
            for ind_var_name in ind_vars_names:
                self.samplers[ind_var_name].sample(data, model, realizations, self.temperature_inv, **computation_kws)

            # Append current realizations if burn in is finished
            if i > self.algo_parameters['n_burn_in_iter']:
                realizations_history.append(realizations.copy())

        # Create a new realizations and assign each individual parameters variable to its mean realization
        mean_realizations = model.initialize_realizations_for_model(data.n_individuals)

        for ind_var_name in ind_vars_names:
            stacked_reals_var = torch.stack([realizations[ind_var_name].tensor_realizations
                                             for realizations in realizations_history])
            mean_realizations[ind_var_name].tensor_realizations = stacked_reals_var.mean(dim=0).clone().detach()

        # Get individual realizations from realizations object
        param_ind = model.get_param_from_real(mean_realizations)

        ### TODO : The following was added for the conversion from Results to IndividualParameters. Everything should be changed

        individual_parameters = IndividualParameters()
        p_names = list(param_ind.keys())
        n_sub = len(param_ind[p_names[0]])

        for i in range(n_sub):
            p_dict = {k: param_ind[k][i].numpy() for k in p_names}
            p_dict = {k: v[0] if v.shape[0] == 1 else v.tolist() for k, v in p_dict.items()}
            individual_parameters.add_individual_parameters(str(i), p_dict)

        return individual_parameters
