import torch

from leaspy.algo.personalize.abstract_personalize_algo import AbstractPersonalizeAlgo
from leaspy.algo.utils.samplers import AlgoWithSamplersMixin
from leaspy.io.outputs.individual_parameters import IndividualParameters


class ModeReal(AlgoWithSamplersMixin, AbstractPersonalizeAlgo):
    """
    Sampler based algorithm, individual parameters are derivated as the most frequent realization for `n_samples` samplings.

    TODO many stuff is duplicated between this class & mean_real (& other mcmc stuff) --> refactorize???
    TODO BUGFIX? temperature is never updated here unlike in fit (so only algo_parameters['annealing']['initial_temperature'] will be used)

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        Settings of the algorithm.
    """

    name = 'mode_real'

    def _initialize_annealing(self):
        if self.algo_parameters['annealing']['do_annealing']:
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

        # Initialize annealing
        self._initialize_annealing()

        # Initialize realizations, but really around their actual values for individual parameters!
        # TODO? is it really needed to restrict to that point individual parameters std-dev?
        realizations = model.initialize_realizations_for_model(data.n_individuals, scale_individual=0.01)
        ind_vars_names = realizations.reals_ind_variable_names

        # Gibbs sample n_iter times
        for i in range(self.algo_parameters['n_iter']):
            for ind_var_name in ind_vars_names:
                self.samplers[ind_var_name].sample(data, model, realizations, self.temperature_inv, **computation_kws)

            # Append current realizations if burn in is finished
            if i > self.algo_parameters['n_burn_in_iter']:
                realizations_history.append(realizations.copy())

        # Get for each patient the realization that best fits (attachement + regularity)

        # We do not sample the population parameters ("fixed effects") so we don't need attribute_type='MCMC' here
        attachments = torch.stack(
            [model.compute_individual_attachment_tensorized(data, model.get_param_from_real(realizations), **computation_kws)
             for realizations in realizations_history]
        )

        # Regularity for each individual (sum on all individual parameters and on all dimensions of those params)
        regularities = []
        for realizations in realizations_history:
            regularity_ind = 0.
            for ind_var_name in ind_vars_names:
                ind_var_real = realizations[ind_var_name]
                # cf. GibbsSampler._sample_individual_realizations -> TODO mutualize code btw those two methods?
                ind_param_dims_but_individual = self.samplers[ind_var_name].ind_param_dims_but_individual
                regularity_ind += model.compute_regularity_realization(ind_var_real).sum(dim=ind_param_dims_but_individual)
            regularities.append(regularity_ind)
        regularities = torch.stack(regularities)

        # Indices of iterations where loss was minimal (per individual, but tradeoff on ALL individual parameters)
        # TODO? shouldn't the regularity term be balanced with self.temperature_inv here?
        indices_min = torch.argmin(attachments + regularities, dim=0)

        # Create a new realizations and assign each individual parameters variable to its best realization
        mode_realizations = model.initialize_realizations_for_model(data.n_individuals)

        for ind_var_name in ind_vars_names:
            ind_var_best_real = torch.stack(
                [realizations_history[indices_min[individual_i]][ind_var_name].tensor_realizations[individual_i]
                 for individual_i, _ in enumerate(data.indices)]
            )
            mode_realizations[ind_var_name].tensor_realizations = ind_var_best_real.clone().detach()

        # Get individual realizations from realizations object
        ind_parameters = model.get_param_from_real(mode_realizations)

        ### TODO : The following was adding for the conversion from Results to IndividualParameters. Everything should be changed
        # TODO: mutualize between mode & mean real algo!

        individual_parameters = IndividualParameters()
        p_names = list(ind_parameters.keys())
        n_sub = len(ind_parameters[p_names[0]])

        for i in range(n_sub):
            p_dict = {k: ind_parameters[k][i].numpy() for k in p_names}
            p_dict = {k: v[0] if v.shape[0] == 1 else v.tolist() for k, v in p_dict.items()}
            individual_parameters.add_individual_parameters(str(i), p_dict)

        return individual_parameters
