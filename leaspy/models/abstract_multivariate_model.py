from abc import abstractmethod
import json
import math

import torch

from leaspy.models.utils.attributes.attributes_factory import AttributesFactory
from leaspy.models.utils.initialization.model_initialization import initialize_parameters
from .abstract_model import AbstractModel


class AbstractMultivariateModel(AbstractModel):
    """
    Contains the common attributes & methods of the multivariate models.
    """
    def __init__(self, name, **kwargs):
        super().__init__(name)
        self.source_dimension = None
        self.dimension = None
        self.parameters = {
            "g": None,
            "betas": None,
            "tau_mean": None, "tau_std": None,
            "xi_mean": None, "xi_std": None,
            "sources_mean": None, "sources_std": None,
            "noise_std": None
        }
        self.bayesian_priors = None
        self.attributes = None

        # MCMC related "parameters"
        self.MCMC_toolbox = {
            'attributes': None,
            'priors': {
                # for logistic: "p0" = 1 / (1+exp(g)) i.e. exp(g) = 1/p0 - 1
                # for linear: "p0" = g
                'g_std': None,
                'betas_std': None
            }
        }

        # load hyperparameters
        self.load_hyperparameters(kwargs)

    def smart_initialization_realizations(self, data, realizations):
        # TODO : Qui a fait ça? A quoi ça sert?
        # means_time = torch.tensor([torch.mean(data.get_times_patient(i)) for
        # i in range(data.n_individuals)]).reshape(realizations['tau'].tensor_realizations.shape)
        # realizations['tau'].tensor_realizations = means_time
        return realizations

    def initialize(self, dataset, method="default"):
        self.dimension = dataset.dimension
        self.features = dataset.headers

        if self.source_dimension is None:
            self.source_dimension = int(math.sqrt(dataset.dimension))

        self.parameters = initialize_parameters(self, dataset, method)

        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension)
        self.attributes.update(['all'], self.parameters)
        self.is_initialized = True

    @abstractmethod
    def initialize_MCMC_toolbox(self):
        raise NotImplementedError

    def load_hyperparameters(self, hyperparameters):
        if 'dimension' in hyperparameters.keys():
            self.dimension = hyperparameters['dimension']
        if 'source_dimension' in hyperparameters.keys():
            self.source_dimension = hyperparameters['source_dimension']
        if 'features' in hyperparameters.keys():
            self.features = hyperparameters['features']
        if 'loss' in hyperparameters.keys():
            self.loss = hyperparameters['loss']

        if any([key not in ('features', 'loss', 'dimension', 'source_dimension') for key in hyperparameters.keys()]):
            raise ValueError("Only <features>, <loss>, <dimension> and <source_dimension> are valid hyperparameters "
                             f"for an AbstractMultivariateModel! You gave {hyperparameters}.")

    def save(self, path, **kwargs):
        """
        Save Leaspy object as json model parameter file.

        Parameters
        ----------
        path: str
            Path to store the model's parameters.
        **kwargs
            Keyword arguments for json.dump method.
        """
        model_parameters_save = self.parameters.copy()
        model_parameters_save['mixing_matrix'] = self.attributes.mixing_matrix
        for key, value in model_parameters_save.items():
            if type(value) in [torch.Tensor]:
                model_parameters_save[key] = value.tolist()

        model_settings = {
            'name': self.name,
            'features': self.features,
            'dimension': self.dimension,
            'source_dimension': self.source_dimension,
            'loss': self.loss,
            'parameters': model_parameters_save
        }
        with open(path, 'w') as fp:
            json.dump(model_settings, fp, **kwargs)

    @abstractmethod
    def compute_individual_tensorized(self, timepoints, individual_parameters, attribute_type=None):
        raise NotImplementedError

    def compute_mean_traj(self, timepoints):
        individual_parameters = {
            'xi': torch.tensor([self.parameters['xi_mean']], dtype=torch.float32),
            'tau': torch.tensor([self.parameters['tau_mean']], dtype=torch.float32),
            'sources': torch.zeros(self.source_dimension, dtype=torch.float32)
        }

        return self.compute_individual_tensorized(timepoints, individual_parameters)

    def _get_attributes(self, attribute_type):
        if attribute_type is None:
            return self.attributes.get_attributes()
        elif attribute_type == 'MCMC':
            return self.MCMC_toolbox['attributes'].get_attributes()
        else:
            raise ValueError("The specified attribute type does not exist : {}".format(attribute_type))
