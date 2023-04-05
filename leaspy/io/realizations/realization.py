from __future__ import annotations

import abc
from typing import Tuple, Optional, TYPE_CHECKING

import torch

from leaspy.utils.typing import ParamType

if TYPE_CHECKING:
    from leaspy.models.abstract_model import AbstractModel


class AbstractRealization:
    """
    Abstract class for Realization.

    Attributes
    ----------
    name : ParamType
        The name of the variable associated with the realization.
    shape : Tuple[int, ...]
        The shape of the tensor realization.
    tensor_realizations : torch.Tensor
        The tensor realization.
    """
    def __init__(self, name: ParamType, shape: Tuple[int, ...], **kwargs):
        self.name = name
        self.shape = shape
        self._tensor_realizations: Optional[torch.Tensor] = None

    @property
    def tensor_realizations(self) -> torch.Tensor:
        return self._tensor_realizations

    @tensor_realizations.setter
    def tensor_realizations(self, tensor_realizations: torch.Tensor):
        if not isinstance(tensor_realizations, torch.Tensor):
            raise TypeError(
                f"Expected a torch tensor object but received a {type(tensor_realizations)} instead."
            )
        self._tensor_realizations = tensor_realizations

    def set_tensor_realizations_element(self, element: torch.Tensor, dim: tuple[int, ...]) -> None:
        """
        Manually change the value (in-place) of `tensor_realizations` at dimension `dim`.

        Parameters
        ----------
        element : torch.Tensor
            The element to put in the tensor realization.
        dim : Tuple[int, ...]
            The dimension where to put the element.
        """
        if not isinstance(element, torch.Tensor):
            raise TypeError(
                f"Expected a torch tensor object but received a {type(element)} instead."
            )
        self._tensor_realizations[dim] = element

    @classmethod
    def from_tensor(
        cls,
        name: str,
        shape: Tuple[int, ...],
        tensor_realization: torch.Tensor,
    ):
        """
        Create realization from variable info and torch tensor object.

        Parameters
        ----------
        name : str
            Variable name.
        shape : tuple of int
            Shape of variable (multiple dimensions allowed).
        tensor_realization : :class:`torch.Tensor`
            Actual realizations, whose shape is given by `shape`.

        Returns
        -------
        AbstractRealization
        """
        # TODO : a check of shapes
        realization = cls(name, shape)
        realization.tensor_realizations = tensor_realization.clone().detach()
        return realization

    @abc.abstractmethod
    def initialize(
        self,
        model: AbstractModel,
        *,
        init_at_mean: bool = False
    ):
        """
        Initialize realization from a given model.

        Parameters
        ----------
        model : :class:`.AbstractModel`
            The model you want realizations for.
        init_at_mean : bool (default False)
            If True: individual variable will be initialized at its mean (from model parameters)
            Otherwise: individual variable will be a random draw from a Gaussian distribution
            with loc and scale parameter from model parameters.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            if unknown variable type
        """

    def set_autograd(self) -> None:
        """
        Set autograd for tensor of realizations.

        TODO remove? only in legacy code

        Raises
        ------
        :class:`ValueError`
            if inconsistent internal request

        See Also
        --------
        torch.Tensor.requires_grad_
        """
        if self._tensor_realizations.requires_grad:
            raise ValueError("Realizations are already using autograd")
        self._tensor_realizations.requires_grad_(True)

    def unset_autograd(self) -> None:
        """
        Unset autograd for tensor of realizations

        TODO remove? only in legacy code

        Raises
        ------
        :class:`ValueError`
            if inconsistent internal request

        See Also
        --------
        torch.Tensor.requires_grad_
        """
        if not self._tensor_realizations.requires_grad:
            raise ValueError("Realizations are already detached")
        self._tensor_realizations.requires_grad_(False)

    def __str__(self):
        s = f"Realization of {self.name}\n"
        s += f"Shape : {self.shape}\n"
        return s


class IndividualRealization(AbstractRealization):
    """
    Class for realizations of individual variables.
    """
    def __init__(self, name: ParamType, shape: Tuple[int, ...], n_individuals: int):
        super().__init__(name, shape)
        self.n_individuals = n_individuals

    def initialize(
        self,
        model: AbstractModel,
        *,
        init_at_mean: bool = False
    ):
        if init_at_mean:
            self.initialize_at_mean(model.parameters[f"{self.name}_mean"])
        else:
            self.initialize_around_mean(
                model.parameters[f"{self.name}_mean"],
                model.parameters[f"{self.name}_std"],
            )

    def initialize_at_mean(self, mean: torch.Tensor) -> None:
        self._tensor_realizations = mean * torch.ones((self.n_individuals, *self.shape))

    def initialize_around_mean(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        distribution = torch.distributions.normal.Normal(loc=mean, scale=std)
        self._tensor_realizations = distribution.sample(
            sample_shape=(self.n_individuals, *self.shape)
        )

    def __str__(self):
        s = super().__str__()
        s += f"Variable type : individual"
        return s

    def __deepcopy__(self, memo) -> IndividualRealization:
        """
        Deep-copy the Realization object (magic method invoked with using copy.deepcopy)

        It clones the underlying tensor and detach it from the computational graph

        Returns
        -------
        `IndividualRealization`
        """
        new = IndividualRealization(self.name, self.shape, self.n_individuals)
        new.tensor_realizations = self.tensor_realizations.clone().detach()
        return new


class PopulationRealization(AbstractRealization):
    """
    Class for realizations of population variables.
    """

    def initialize(
        self,
        model: AbstractModel,
        *,
        init_at_mean: bool = False
    ) -> None:
        self._tensor_realizations = model.parameters[self.name].reshape(self.shape)

    def __str__(self):
        s = super().__str__()
        s += f"Variable type : population"
        return s

    def __deepcopy__(self, memo) -> PopulationRealization:
        """
        Deep-copy the Realization object (magic method invoked with using copy.deepcopy)

        It clones the underlying tensor and detach it from the computational graph

        Returns
        -------
        `PopulationRealization`
        """
        new = PopulationRealization(self.name, self.shape)
        new.tensor_realizations = self.tensor_realizations.clone().detach()
        return new
