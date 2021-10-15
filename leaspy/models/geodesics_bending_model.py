from abc import abstractmethod
import json
import math

import torch

from leaspy import __version__

from leaspy.models.utils.attributes import AttributesFactory
from leaspy.models.utils.initialization.model_initialization import initialize_parameters
from leaspy.utils.docs import doc_with_super
from leaspy.models.abstract_multivariate_model import AbstractMultivariateModel
from leaspy.io.settings.model_settings import ModelSettings

import leaspy.models.utils.OptimB as OptimB

init_mapping = {"identity": torch.nn.Identity(),
                "negidentity": lambda x: -x,
                "logistic": lambda x: 1. / (1. + torch.exp(-x))}


@doc_with_super()
class GeodesicsBending(AbstractMultivariateModel):
    """
    Contains the common attributes & methods of the multivariate models.
    """

    def __init__(self, name, base_model = None, **kwargs):
        super().__init__(name, **kwargs)
        self.base_model = base_model
        self.base_model_path = None
        assert isinstance(self.base_model, AbstractMultivariateModel), "Base model for GeodesicsBending must be a " \
                                                                       "AbstractMultivariateModel"
        self.mapping = None # Diffeomorphism to apply to previous model
        self.kernel_settings = None
        self.parameters = {"weights" : None,
                           "control_points" : None}
        self.initial_mapping = "identity"

        # load hyperparameters
        self.load_hyperparameters(kwargs)

    def init_mapping_function(self):
        self.mapping = init_mapping[self.initial_mapping]

    def mapping_reconstruction(self):
        self.init_mapping_function()

        weights, control_points = self.parameters["weights"], self.parameters["control_points"]
        W = torch.tensor(weights, dtype=torch.float32).clone().detach()
        X = torch.tensor(control_points, dtype=torch.float32).clone().detach()
        mapping = OptimB.transformation_B_compose(X, W, self.kernel_settings, self.mapping)
        self.mapping = mapping

    def initialize(self, dataset, method="default"):
        self.base_model.initialize(dataset, method=method)
        self.dimension = dataset.dimension
        self.features = dataset.headers

        self.source_dimension = self.base_model.source_dimension

        self.is_initialized = True

    @abstractmethod
    def initialize_MCMC_toolbox(self):
        """
        Initialize Monte-Carlo Markov-Chain toolbox for calibration of model
        """
        self.base_model.initialize_MCMC_toolbox()

    @abstractmethod
    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        """
        Update the MCMC toolbox with a collection of realizations of model population parameters.

        Parameters
        ----------
        name_of_the_variables_that_have_been_changed: container[str] (list, tuple, ...)
            Names of the population parameters to update in MCMC toolbox
        realizations : :class:`.CollectionRealization`
            All the realizations to update MCMC toolbox with
        """
        self.base_model.update_MCMC_toolbox(name_of_the_variables_that_have_been_changed, realizations)

    def load_hyperparameters(self, hyperparameters):
        if 'dimension' in hyperparameters.keys():
            self.dimension = hyperparameters['dimension']
        if 'source_dimension' in hyperparameters.keys():
            self.source_dimension = hyperparameters['source_dimension']
        if 'features' in hyperparameters.keys():
            self.features = hyperparameters['features']
        if 'loss' in hyperparameters.keys():
            self.loss = hyperparameters['loss']

        if 'kernel_settings' in hyperparameters.keys():
            self.kernelsettings = hyperparameters['kernelsettings']
        if 'initial_mapping' in hyperparameters.keys():
            self.initial_mapping = hyperparameters['initial_mapping']
            self.init_mapping_function()

        if 'base_model_path' in hyperparameters.keys():
            if self.base_model_path is None:
                self.base_model_path = hyperparameters["save_base_model"]

        expected_hyperparameters = (
            'features', 'loss', 'dimension', 'source_dimension',
            'base_model_path', 'kernel_settings', 'initial_mapping')
        unexpected_hyperparameters = set(hyperparameters.keys()).difference(expected_hyperparameters)
        if len(unexpected_hyperparameters) > 0:
            raise ValueError(
                f"Only {', '.join([f'<{p}>' for p in expected_hyperparameters])} are valid hyperparameters "
                f"for an AbstractMultivariateModel! Unknown hyperparameters: {unexpected_hyperparameters}.")

    def load_parameters(self, parameters):
        self.parameters = {}
        for k in parameters.keys():
            if k in ['mixing_matrix']:
                continue
            self.parameters[k] = torch.tensor(parameters[k], dtype=torch.float32)
        self.mapping_reconstruction()

        # Load base model
        reader = ModelSettings(self.base_model_path)
        self.base_model.load_hyperparameters(reader.hyperparameters)
        self.base_model.load_parameters(reader.parameters)

    def save(self, path, with_mixing_matrix=True, comp=0, **kwargs):
        """
        Save Leaspy object as json model parameter file.

        Parameters
        ----------
        path: str
            Path to store the model's parameters.
        with_mixing_matrix: bool (default True)
            Save the mixing matrix in the exported file in its 'parameters' section.
            <!> It is not a real parameter and its value will be overwritten at model loading
                (orthonormal basis is recomputed from other "true" parameters and mixing matrix is then deduced from this orthonormal basis and the betas)!
            It was integrated historically because it is used for convenience in browser webtool and only there...
        **kwargs
            Keyword arguments for json.dump method.
        """
        model_parameters_save = self.parameters.copy()

        if with_mixing_matrix:
            model_parameters_save['mixing_matrix'] = self.attributes.mixing_matrix

        for key, value in model_parameters_save.items():
            if type(value) in [torch.Tensor]:
                model_parameters_save[key] = value.tolist()

        # Save base model as well
        split = path.split(".")
        if comp == 0:
            model_path = path
        else:
            model_path = ".".join(split[:-1]) + "_base_{}.".format(comp)+split[-1]
        new_path = ".".join(split[:-1]) + "_base_{}.".format(comp+1)+split[-1]
        self.base_model_path = new_path
        if isinstance(self.base_model, GeodesicsBending):
            self.base_model.save(path, with_mixing_matrix=with_mixing_matrix, comp=comp+1, **kwargs)
        else:
            self.base_model.save(self.base_model_path, with_mixing_matrix=with_mixing_matrix, **kwargs)

        model_settings = {
            'leaspy_version': __version__,
            'name': self.name,
            'features': self.features,
            'dimension': self.dimension,
            'source_dimension': self.source_dimension,
            'loss': self.loss,
            'parameters': model_parameters_save,
            'initial_mapping': self.initial_mapping,
            'kernel_settings': self.kernel_settings,
            'base_model_path': self.base_model_path,
        }

        with open(model_path, 'w') as fp:
            json.dump(model_settings, fp, **kwargs)

    def compute_individual_tensorized(self, timepoints, individual_parameters, attribute_type=None):
        trajectories = self.base_model.compute_individual_tensorized(timepoints, individual_parameters, attribute_type=attribute_type)
        mapped_trajectories = self.mapping(trajectories)
        return mapped_trajectories

    def compute_mean_traj(self, timepoints):
        """
        Compute trajectory of the model with individual parameters being the group-average ones.

        TODO check dimensions of io?

        Parameters
        ----------
        timepoints : :class:`torch.Tensor` [1, n_timepoints]

        Returns
        -------
        :class:`torch.Tensor` [1, n_timepoints, dimension]
            The group-average values at given timepoints
        """
        individual_parameters = {
            'xi': torch.tensor([self.parameters['xi_mean']], dtype=torch.float32),
            'tau': torch.tensor([self.parameters['tau_mean']], dtype=torch.float32),
            'sources': torch.zeros(self.source_dimension, dtype=torch.float32)
        }

        return self.compute_individual_tensorized(timepoints, individual_parameters)
