import torch
from .abstract_model import AbstractModel
import json

class AbstractMultivariateModel(AbstractModel):
    def __init__(self, name):
        super(AbstractMultivariateModel, self).__init__(name)
        self.source_dimension = None
        self.dimension = None
        self.parameters = {
            "g": None, "betas": None,
            "tau_mean": None, "tau_std": None,
            "xi_mean": None,  "xi_std": None,
            "sources_mean": None, "sources_std": None,
            "noise_std": None
        }
        self.bayesian_priors = None
        self.attributes = None

        # MCMC related "parameters"
        self.MCMC_toolbox = {
            'attributes': None,
            'priors': {
                'g_std': None, # tq p0 = 1 / (1+exp(g)) i.e. g = 1/p0 - 1
                'betas_std': None
            }
        }

    def smart_initialization_realizations(self, data, realizations):
        #means_time = torch.Tensor([torch.mean(data.get_times_patient(i)) for i in range(data.n_individuals)]).reshape(realizations['tau'].tensor_realizations.shape)
        #realizations['tau'].tensor_realizations = means_time
        return realizations

    def load_hyperparameters(self, hyperparameters):
        if 'dimension' in hyperparameters.keys():
            self.dimension = hyperparameters['dimension']
        if 'source_dimension' in hyperparameters.keys():
            self.source_dimension = hyperparameters['source_dimension']

    def save(self, path):
        model_parameters_save = self.parameters.copy()
        model_parameters_save['mixing_matrix'] = self.attributes.mixing_matrix
        for key, value in model_parameters_save.items():
            if type(value) in [torch.Tensor]:
                model_parameters_save[key] = value.tolist()
        model_settings = {
            'name': self.name,
            'dimension': self.dimension,
            'source_dimension': self.source_dimension,
            'parameters': model_parameters_save
        }
        with open(path, 'w') as fp:
            json.dump(model_settings, fp)

    def compute_mean_traj(self,timepoints):
        individual_parameters = {
            'xi': torch.Tensor([self.parameters['xi_mean']]),
            'tau': torch.Tensor([self.parameters['tau_mean']]),
            'sources' : torch.zeros(self.source_dimension)
        }

        return self.compute_individual_tensorized(timepoints, individual_parameters)

    def param_ind_from_dict(self,individual_parameters):
        xi, tau, sources = [], [], []
        for key, item in individual_parameters.items():
            xi.append(item['xi'])
            tau.append(item['tau'])
            sources.append(item['sources'])
        xi = torch.tensor(xi,dtype=torch.float32).unsqueeze(1)
        tau = torch.tensor(tau,dtype=torch.float32).unsqueeze(1)
        sources = torch.tensor(sources,dtype=torch.float32)
        return (xi,tau,sources)

    def get_param_from_real(self, realizations):
        #xi = realizations['xi'].tensor_realizations
        #tau = realizations['tau'].tensor_realizations
        #if self.source_dimension == 0:
        #    sources = None
        #else:
        #    sources = realizations['sources'].tensor_realizations

        individual_parameters = dict.fromkeys(self.get_individual_variable_name())

        for variable_ind in self.get_individual_variable_name():
            if variable_ind == "sources" and self.source_dimension == 0:
                individual_parameters[variable_ind] = None
            else:
                individual_parameters[variable_ind] = realizations[variable_ind].tensor_realizations

        return individual_parameters

    def get_xi_tau(self,param_ind):
        xi,tau,sources = param_ind
        return xi,tau