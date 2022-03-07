from abc import abstractmethod
import json
import math
import warnings

import torch

from leaspy import __version__

from leaspy.models.abstract_model import AbstractModel
from leaspy.models.utils.attributes import AttributesFactory
from leaspy.models.utils.attributes.abstract_manifold_model_attributes import AbstractManifoldModelAttributes
from leaspy.models.utils.initialization.model_initialization import initialize_parameters
from leaspy.models.utils.noise_model import NoiseModel

from leaspy.utils.typing import KwargsType, List, Optional
from leaspy.utils.docs import doc_with_super
from leaspy.exceptions import LeaspyModelInputError


@doc_with_super()
class AbstractMultivariateModel(AbstractModel):
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
        self.noise_model = 'gaussian_diagonal'

        self.parameters = {
            "g": None,
            "betas": None,
            "tau_mean": None, "tau_std": None,
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

        # Load hyperparameters at end to overwrite default for new hyperparameters
        self.load_hyperparameters(kwargs)

    """
    def smart_initialization_realizations(self, data, realizations):
        # TODO : Qui a fait ça? A quoi ça sert?
        # means_time = torch.tensor([torch.mean(data.get_times_patient(i)) for
        # i in range(data.n_individuals)]).reshape(realizations['tau'].tensor_realizations.shape)
        # realizations['tau'].tensor_realizations = means_time
        return realizations
    """

    def initialize(self, dataset, method: str = 'default'):

        if dataset.dimension < 2:
            raise LeaspyModelInputError("A multivariate model should have at least 2 features but your dataset "
                                        f"only contains {dataset.dimension} features ({dataset.headers}).")

        self.dimension = dataset.dimension
        self.features = dataset.headers

        if self.source_dimension is None:
            self.source_dimension = int(math.sqrt(dataset.dimension))
            warnings.warn('You did not provide `source_dimension` hyperparameter for multivariate model, '
                          f'setting it to ⌊√dimension⌋ = {self.source_dimension}.')

        elif not (isinstance(self.source_dimension, int) and 0 <= self.source_dimension < self.dimension):
            raise LeaspyModelInputError(f"Sources dimension should be an integer in [0, dimension - 1[ "
                                        f"but you provided `source_dimension` = {self.source_dimension} whereas `dimension` = {self.dimension}")

        self.parameters = initialize_parameters(self, dataset, method)

        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension)

        # Postpone the computation of attributes when really needed!
        #self.attributes.update(['all'], self.parameters)

        self.is_initialized = True

    @abstractmethod
    def initialize_MCMC_toolbox(self) -> None:
        """
        Initialize Monte-Carlo Markov-Chain toolbox for calibration of model
        """
        # TODO to move in a "MCMC-model interface"

    @abstractmethod
    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed: List[str], realizations) -> None:
        """
        Update the MCMC toolbox with a collection of realizations of model population parameters.

        Parameters
        ----------
        name_of_the_variables_that_have_been_changed : container[str] (list, tuple, ...)
            Names of the population parameters to update in MCMC toolbox
        realizations : :class:`.CollectionRealization`
            All the realizations to update MCMC toolbox with
        """
        # TODO to move in a "MCMC-model interface"

    def load_hyperparameters(self, hyperparameters: KwargsType):

        expected_hyperparameters = ('features', 'dimension', 'source_dimension')

        if 'features' in hyperparameters.keys():
            self.features = hyperparameters['features']

        if 'dimension' in hyperparameters.keys():
            if self.features and hyperparameters['dimension'] != len(self.features):
                raise LeaspyModelInputError(f"Dimension provided ({hyperparameters['dimension']}) does not match features ({len(self.features)})")
            self.dimension = hyperparameters['dimension']

        if 'source_dimension' in hyperparameters.keys():
            if not (
                isinstance(hyperparameters['source_dimension'], int)
                and (hyperparameters['source_dimension'] >= 0)
                and (not self.dimension or hyperparameters['source_dimension'] <= self.dimension - 1)
            ):
                raise LeaspyModelInputError(f"Source dimension should be an integer in [0, dimension - 1], not {hyperparameters['source_dimension']}")
            self.source_dimension = hyperparameters['source_dimension']

        # load new `noise_model` directly in-place & add the recognized hyperparameters to known tuple
        expected_hyperparameters += NoiseModel.set_noise_model_from_hyperparameters(self, hyperparameters)

        self._raise_if_unknown_hyperparameters(expected_hyperparameters, hyperparameters)

    def save(self, path: str, with_mixing_matrix: bool = True, **kwargs):
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
            'noise_model': self.noise_model,
            'parameters': model_parameters_save
        }

        # Default json.dump kwargs:
        kwargs = {'indent': 2, **kwargs}

        with open(path, 'w') as fp:
            json.dump(model_settings, fp, **kwargs)

    @abstractmethod
    def compute_individual_tensorized(self, timepoints, individual_parameters, *, attribute_type=None) -> torch.FloatTensor:
        pass

    def compute_mean_traj(self, timepoints, *, attribute_type: Optional[str] = None):
        """
        Compute trajectory of the model with individual parameters being the group-average ones.

        TODO check dimensions of io?

        Parameters
        ----------
        timepoints : :class:`torch.Tensor` [1, n_timepoints]
        attribute_type : 'MCMC' or None

        Returns
        -------
        :class:`torch.Tensor` [1, n_timepoints, dimension]
            The group-average values at given timepoints
        """
        individual_parameters = {
            'xi': torch.tensor([self.parameters['xi_mean']]),
            'tau': torch.tensor([self.parameters['tau_mean']]),
            'sources': torch.zeros(self.source_dimension)
        }

        return self.compute_individual_tensorized(timepoints, individual_parameters, attribute_type=attribute_type)

    def _get_attributes(self, attribute_type: Optional[str]):
        if attribute_type is None:
            return self.attributes.get_attributes()
        elif attribute_type == 'MCMC':
            return self.MCMC_toolbox['attributes'].get_attributes()
        else:
            raise LeaspyModelInputError(f"The specified attribute type does not exist: {attribute_type}. "
                                        "Should be None or 'MCMC'.")
