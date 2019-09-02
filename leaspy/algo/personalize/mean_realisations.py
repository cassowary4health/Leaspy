from .abstract_personalize_algo import AbstractPersonalizeAlgo
from ..samplers.hmc_sampler import HMCSampler
from ..samplers.gibbs_sampler import GibbsSampler

import torch

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


    def run(self, model, data):

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
            print(name_variable, info_variable['type'])
            if info_variable['type'] == 'individual':
                print(name_variable, "ok")
                mean_variable = torch.stack([realizations[name_variable].tensor_realizations for realizations in realizations_history]).mean(dim=0).clone().detach().numpy()

                mean_output[name_variable] = mean_variable

        # Change data type
        mean_output_patients = dict.fromkeys(data.indices)

        for j, idx in enumerate(data.indices):
            mean_output_patients[idx] = {}
            mean_output_patients[idx]['xi'] =  mean_output['xi'][j]
            mean_output_patients[idx]['tau'] = mean_output['tau'][j]
            mean_output_patients[idx]['sources'] = mean_output['sources'][j]

        return mean_output_patients, 0.14






