from abc import abstractmethod
import json
import math

import torch

from leaspy import __version__

from leaspy.models.abstract_model import AbstractModel
from leaspy.models.utils.attributes import AttributesFactory
from leaspy.models.utils.attributes.abstract_manifold_model_attributes import AbstractManifoldModelAttributes
from leaspy.models.utils.initialization.model_initialization import initialize_parameters

from leaspy.utils.docs import doc_with_super
from leaspy.exceptions import LeaspyModelInputError


@doc_with_super()
class AbstractMultivariateLinkModel(AbstractModel):
    """
    Contains the common attributes & methods of the multivariate models.

    Parameters
    ----------
    name : str
        Name of the model
    **kwargs
        Hyperparameters for the model

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        if inconsistent hyperparameters
    """
    def __init__(self, name: str, **kwargs):
        super().__init__(name)
        self.source_dimension: int = None
        
        self.dimension: int = None
        self.cofactors_dimension: int = None
        self.cofactors: torch.Tensor = None
        self.link_v0_shape = None
        self.link_t_mean_shape = None

        self.parameters = {
            "g": None,
            "betas": None,
            "tau_std": None,
            "link_v0": None,
            "link_t_mean": None,
            "xi_mean": None, "xi_std": None,
            "sources_mean": None, "sources_std": None,
            "noise_std": None
        }
        self.bayesian_priors = None
        self.attributes: AbstractManifoldModelAttributes = None

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

    """
    def smart_initialization_realizations(self, data, realizations):
        # TODO : Qui a fait ça? A quoi ça sert?
        # means_time = torch.tensor([torch.mean(data.get_times_patient(i)) for
        # i in range(data.n_individuals)]).reshape(realizations['tau'].tensor_realizations.shape)
        # realizations['tau'].tensor_realizations = means_time
        return realizations
    """

    def initialize(self, dataset, method="default", precomputed=None):
        self.dimension = dataset.dimension
        self.features = dataset.headers
        self.cofactors_dimension = dataset.cofactors_dimension
        self.cofactors = dataset.cofactors
        #self.link_shape = torch.Size([self.dimension+1, self.cofactors_dimension+1])
        self.link_v0_shape = torch.Size([self.dimension, self.cofactors_dimension+1])
        self.link_t_mean_shape = torch.Size([1, self.cofactors_dimension+1])


        if self.source_dimension is None:
            self.source_dimension = int(math.sqrt(dataset.dimension))

        self.parameters = initialize_parameters(self, dataset, method, precomputed=precomputed)

        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension, self.device)
        self.attributes.update(['all'], self.parameters)
        self.is_initialized = True

    @abstractmethod
    def initialize_MCMC_toolbox(self):
        """
        Initialize Monte-Carlo Markov-Chain toolbox for calibration of model

        TODO to move in a "MCMC-model interface"
        """
        pass

    @abstractmethod
    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        """
        Update the MCMC toolbox with a collection of realizations of model population parameters.

        TODO to move in a "MCMC-model interface"

        Parameters
        ----------
        name_of_the_variables_that_have_been_changed : container[str] (list, tuple, ...)
            Names of the population parameters to update in MCMC toolbox
        realizations : :class:`.CollectionRealization`
            All the realizations to update MCMC toolbox with
        """
        pass

    def load_hyperparameters(self, hyperparameters):
        if 'dimension' in hyperparameters.keys():
            self.dimension = hyperparameters['dimension']
        if 'source_dimension' in hyperparameters.keys():
            self.source_dimension = hyperparameters['source_dimension']
        if 'features' in hyperparameters.keys():
            self.features = hyperparameters['features']
        if 'loss' in hyperparameters.keys():
            self.loss = hyperparameters['loss']

        expected_hyperparameters = ('features', 'loss', 'dimension', 'source_dimension')
        unexpected_hyperparameters = set(hyperparameters.keys()).difference(expected_hyperparameters)
        if len(unexpected_hyperparameters) > 0:
            raise LeaspyModelInputError(
                    f"Only {expected_hyperparameters} are valid hyperparameters for an AbstractMultivariateModel! "
                    f"Unknown hyperparameters: {unexpected_hyperparameters}.")

    def save(self, path, with_mixing_matrix=True, **kwargs):
        """
        Save Leaspy object as json model parameter file.

        Parameters
        ----------
        path : str
            Path to store the model's parameters.
        with_mixing_matrix : bool (default True)
            Save the mixing matrix in the exported file in its 'parameters' section.
            <!> It is not a real parameter and its value will be overwritten at model loading
            (orthonormal basis is recomputed from other "true" parameters and mixing matrix
            is then deduced from this orthonormal basis and the betas)!
            It was integrated historically because it is used for convenience in browser webtool and only there...
        **kwargs
            Keyword arguments for json.dump method.
            Default to: dict(indent=2)
        """
        model_parameters_save = self.parameters.copy()

        if with_mixing_matrix:
            model_parameters_save['mixing_matrix'] = self.attributes.mixing_matrix

        for key, value in model_parameters_save.items():
            if isinstance(value, torch.Tensor):
                model_parameters_save[key] = value.tolist()

        model_settings = {
            'leaspy_version': __version__,
            'name': self.name,
            'features': self.features,
            'dimension': self.dimension,
            'source_dimension': self.source_dimension,
            'loss': self.loss,
            'parameters': model_parameters_save
        }

        # Default json.dump kwargs:
        kwargs = {'indent': 2, **kwargs}

        with open(path, 'w') as fp:
            json.dump(model_settings, fp, **kwargs)

    @abstractmethod
    def compute_individual_tensorized(self, timepoints, individual_parameters, attribute_type=None):
        pass

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
            'xi': torch.tensor([self.parameters['xi_mean']], dtype=torch.float32, device=self.device),
            'tau': torch.tensor(0.0),#torch.tensor([self.get_intersept('tau_mean')], dtype=torch.float32, device=self.device),
            'sources': torch.zeros(self.source_dimension, dtype=torch.float32, device=self.device),
            'v0': torch.exp(self.get_intersept('v0')[None,:]),
            'tau_mean': self.get_intersept('tau_mean')[None,:],
        }

        return self.compute_individual_tensorized(timepoints, individual_parameters)

    def _get_attributes(self, attribute_type):
        if attribute_type is None:
            return self.attributes.get_attributes()
        elif attribute_type == 'MCMC':
            return self.MCMC_toolbox['attributes'].get_attributes()
        else:
            raise LeaspyModelInputError(f"The specified attribute type does not exist: {attribute_type}. "
                                        "Should be None or 'MCMC'.")
