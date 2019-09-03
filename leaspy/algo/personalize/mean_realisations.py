from .abstract_personalize_algo import AbstractPersonalizeAlgo
from ..samplers.hmc_sampler import HMCSampler
from ..samplers.gibbs_sampler import GibbsSampler

import torch
import time

class MeanReal(AbstractPersonalizeAlgo):

    def __init__(self, settings):
        # Algorithm parameters
        super().__init__(settings)



    # TODO cloned --> factorize
    def _initialize_samplers(self, model, data):
        infos_variables = model.random_variable_informations()
        self.samplers = dict.fromkeys(infos_variables.keys())
        for variable, info in infos_variables.items():
            if info["type"] == "individual":
                if self.algo_parameters['sampler_ind']=='Gibbs':
                    self.samplers[variable] = GibbsSampler(info, data.n_individuals)
                else:
                    self.samplers[variable] = HMCSampler(info, data.n_individuals, self.algo_parameters['eps'])
            else:
                if self.algo_parameters['sampler_pop']=='Gibbs':
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

        for i in range(self.algo_parameters['n_iter']):
            for key in realizations.reals_ind_variable_names:
                self.samplers[key].sample(data, model, realizations, 1.0)

            realizations_history.append(realizations)

        # Adapt realizations output
        mean_output = dict.fromkeys(['xi', 'tau', 'sources'])

        for name_variable, info_variable in model.random_variable_informations().items():

            if info_variable['type'] == 'individual':
                mean_variable = torch.stack(
                    [realizations[name_variable].tensor_realizations for realizations in realizations_history]).mean(
                    dim=0).clone().detach()

                mean_output[name_variable] = mean_variable

        # Change data type
        mean_output_patients = dict.fromkeys(data.indices)

        for j, idx in enumerate(data.indices):
            mean_output_patients[idx] = {}
            mean_output_patients[idx]['xi'] = mean_output['xi'][j].numpy()
            mean_output_patients[idx]['tau'] = mean_output['tau'][j].numpy()
            mean_output_patients[idx]['sources'] = mean_output['sources'][j].numpy()

        # Compute the attachment
        realizations = model.get_realization_object(data.n_individuals)
        for key, value in mean_output.items():
            realizations[key].tensor_realizations = value

        param_ind = model.get_param_from_real(realizations)

        return param_ind

    """


    def run(self, model, data):
        time_beginning = time.time()

        # Initialize realizations storage object
        realizations_history = []

        # Initialize samplers
        self._initialize_samplers(model, data)

        # initialize realizations
        realizations = model.get_realization_object(data.n_individuals)

        for i in range(self.algo_parameters['n_iter']):
            for key in realizations.reals_ind_variable_names:
                self.samplers[key].sample(data, model, realizations, 1.0)

            realizations_history.append(realizations)

        # Adapt realizations output
        mean_output = dict.fromkeys(['xi','tau','sources'])


        for name_variable, info_variable in model.random_variable_informations().items():

            if info_variable['type'] == 'individual':
                mean_variable = torch.stack([realizations[name_variable].tensor_realizations for realizations in realizations_history]).mean(dim=0).clone().detach()

                mean_output[name_variable] = mean_variable

        # Change data type
        mean_output_patients = dict.fromkeys(data.indices)

        for j, idx in enumerate(data.indices):
            mean_output_patients[idx] = {}
            mean_output_patients[idx]['xi'] = mean_output['xi'][j].numpy()
            mean_output_patients[idx]['tau'] = mean_output['tau'][j].numpy()
            mean_output_patients[idx]['sources'] = mean_output['sources'][j].numpy()


        # Compute the attachment
        realizations = model.get_realization_object(data.n_individuals)
        for key, value in mean_output.items():
            realizations[key].tensor_realizations = value

        param_ind = model.get_param_from_real(realizations)
        squared_diff = model.compute_sum_squared_tensorized(data, param_ind, MCMC=True).sum()
        err_std = float(squared_diff.detach().numpy())/data.n_visits

        time_end = time.time()
        diff_time = (time_end - time_beginning) / 1000
        print("The standard deviation of the noise at the end of the personalization is of {:.4f}".format(err_std))
        print("Personalization {1} took : {0}s".format(diff_time, self.name))

        return mean_output_patients, err_std
"""





