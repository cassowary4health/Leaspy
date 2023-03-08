from abc import ABC, abstractmethod
from leaspy.utils.typing import FeatureType, List, Optional
from leaspy.exceptions import LeaspyModelInputError


class BaseModel(ABC):
    """
    Base model class from which all Leaspy models should inherit.

    It defines the interface that a model should implement to be
    compatible with Leaspy.

    Parameters
    ----------
    name : str
        Name of the model

    **kwargs
        Hyperparameters of the model

    Attributes
    ----------
    name : str
        Name of the model

    is_initialized : bool
        Is the model initialized?

    features : list[str]
        List of model features (None if not initialization)

    dimension : int
        Number of features
    """

    def __init__(self, name: str, **kwargs):
        self.is_initialized: bool = False
        self.name = name
        self._features: Optional[List[FeatureType]] = None
        self._dimension: Optional[int] = None

    @property
    def features(self) -> Optional[List[FeatureType]]:
        return self._features

    @features.setter
    def features(self, features: List[FeatureType]):
        """
        Features setter.
        Ensure coherence between dimension and features attributes.
        """
        if self.dimension is not None and len(features) != self.dimension:
            raise ValueError(
                f"Cannot set the model's features to {features}, because "
                f"the model has been configured with a dimension of {self.dimension}."
            )
        self._features = features

    @property
    def dimension(self) -> Optional[int]:
        """
        The dimension of the model.
        If the private attribute is defined, then it takes precedence over the feature length.
        The associated setters are responsible for their coherence.
        """
        if self._dimension is not None:
            return self._dimension
        if self.features is not None:
            return len(self.features)
        return None

    @dimension.setter
    def dimension(self, dimension: int):
        """
        Dimension setter.
        Ensures coherence between dimension and feature attributes.
        """
        if self.features is None:
            self._dimension = dimension
        else:
            if len(self.features) != dimension:
                raise ValueError(
                    f"Model has {len(self.features)} features. Cannot set the dimension to {dimension}."
                )

    def validate_compatibility_of_dataset(self, dataset) -> None:
        """
        Raise if the given dataset is not compatible with the current model.

        Parameters
        ----------
        dataset : :class:`.Dataset`
            The dataset we want to model.

        Raises
        ------
        LeaspyModelInputError :
            If the dataset has a number of dimensions smaller than 2.
            If the dataset does not have the same dimensionality as the model.
            If the dataset's headers do not match the model's.
        """
        if self.dimension is not None and dataset.dimension != self.dimension:
            raise LeaspyModelInputError(
                f"Unmatched dimensions: {self.dimension} (model) ≠ {dataset.dimension} (data)."
            )
        if self.features is not None and dataset.headers != self.features:
            raise LeaspyModelInputError(
                f"Unmatched features: {self.features} (model) ≠ {dataset.headers} (data)."
            )

    @abstractmethod
    def initialize(self, dataset, method: str = 'default') -> None:
        """
        Initialize the model given a dataset and an initialization method.

        After calling this method :attr:`is_initialized` should be True and model should be ready for use.

        Parameters
        ----------
        dataset : :class:`.Dataset`
            The dataset we want to initialize from.
        method : str
            A custom method to initialize the model
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str, **kwargs) -> None:
        """
        Save Leaspy object as json model parameter file.

        Parameters
        ----------
        path : str
            Path to store the model's parameters.

        **kwargs
            Additional parameters for writing.
        """
        raise NotImplementedError