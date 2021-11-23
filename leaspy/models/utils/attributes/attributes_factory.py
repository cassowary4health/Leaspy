import torch

from leaspy.models.utils.attributes.abstract_attributes import AbstractAttributes
from leaspy.models.utils.attributes import LogisticParallelAttributes, LogisticAttributes, LogisticLinkAttributes, LinearAttributes

from leaspy.exceptions import LeaspyModelInputError


class AttributesFactory:
    """
    Return an `Attributes` class object based on the given parameters.
    """

    _attributes = {
        'logistic': LogisticAttributes,
        'logistic_link': LogisticLinkAttributes,
        'univariate_logistic': LogisticAttributes,

        'logistic_parallel': LogisticParallelAttributes,

        'linear': LinearAttributes,
        'univariate_linear': LinearAttributes,

        #'mixed_linear-logistic': ... # TODO
    }

    @classmethod
    def attributes(cls, name: str, dimension: int, source_dimension: int = None, device: torch.device = None) -> AbstractAttributes:
        """
        Class method to build correct model attributes depending on model `name`.

        Parameters
        ----------
        name : str
        dimension : int
        source_dimension : int, optional (default None)
        device : torch.device (optional, will default to torch.device("cpu") through input None)

        Returns
        -------
        :class:`.AbstractAttributes`

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            if any inconsistent parameter.
        """
        if isinstance(name, str):
            name = name.lower()
        else:
            raise LeaspyModelInputError("The `name` argument must be a string!")

        if name not in cls._attributes:
            raise LeaspyModelInputError(f"The name '{name}' you provided for the attributes is not supported."
                                        f"Valid choices are: {list(cls._attributes.keys())}")

        if 'univariate' in name and dimension != 1:
            raise LeaspyModelInputError(f"{name}: `dimension` should be 1 when 'univariate' is part of model `name`, not {dimension}!")

        return cls._attributes[name](name, dimension, source_dimension, device)
