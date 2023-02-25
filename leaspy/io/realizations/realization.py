from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from leaspy.exceptions import LeaspyModelInputError
from leaspy.utils.typing import ParamType, Tuple
from leaspy.models.utils.distributions import get_distribution

if TYPE_CHECKING:
    from leaspy.models.abstract_model import AbstractModel


class Realization:
    """
    Contains the realization of a given parameter.

    Parameters
    ----------
    name : str
        Variable name
    shape : tuple of int
        Shape of variable (multiple dimensions allowed)
    dist_name : str
        Name of distribution
    variable_type : str
        ``'individual'`` or ``'population'`` variable?

    Attributes
    ----------
    name : str
        Variable name
    shape : tuple of int
        Shape of variable (multiple dimensions allowed)
    dist_name : str
        Name of distribution
    variable_type : str
        ``'individual'`` or ``'population'`` variable?
    tensor_realizations : :class:`torch.Tensor`
        Actual realizations, whose shape is given by `shape`
    """
    def __init__(self, name: ParamType, shape: Tuple[int, ...], dist_name: str, variable_type: str):
        self.name = name
        self.shape = shape
        self.dist_name = dist_name
        self.variable_type = variable_type
        self._tensor_realizations: torch.FloatTensor = None

    @classmethod
    def from_tensor(cls, *, tensor_realizations: torch.FloatTensor, **params):
        """
        Create realization from variable infos and torch tensor object

        Parameters
        ----------
        tensor_realizations : :class:`torch.Tensor`
            Actual realizations, whose shape is given by `shape`
        **params
            Base keyword parameters to be sent to constructor of `Realization`.

        Returns
        -------
        :class:`.Realization`
        """
        # TODO : a check of shapes
        realization = cls(**params)
        realization.tensor_realizations = tensor_realizations.clone().detach()
        return realization

    def initialize(self, n_individuals: int, model: AbstractModel, *, individual_variable_init_at_mean: bool = False):
        """
        Initialize realization from a given model.

        Parameters
        ----------
        n_individuals : int > 0
            Number of individuals
        model : :class:`.AbstractModel`
            The model you want realizations for.
        individual_variable_init_at_mean : bool (default False)
            If True: individual variable will be initialized at its mean (from model parameters)
            Otherwise: individual variable will be a random draw from a Gaussian distribution
            with loc and scale parameter from model parameters.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            if unknown variable type
        """

        if self.variable_type == "population":
            self._tensor_realizations = model.parameters[self.name].reshape(self.shape) # avoid 0D / 1D tensors mix
        elif self.variable_type == 'individual':
            if individual_variable_init_at_mean:
                # TODO: rather use "mode" or "loc" terminology instead of "mean" (since incorrect for ALD)
                self._tensor_realizations = model.parameters[f"{self.name}_mean"] * torch.ones((n_individuals, *self.shape))
            else:
                sampler = get_distribution(self.dist_name, self.name, model.parameters).sample
                self._tensor_realizations = sampler(sample_shape=(n_individuals, *self.shape))
        else:
            raise LeaspyModelInputError(f"Unknown variable type '{self.variable_type}'.")

    @property
    def tensor_realizations(self) -> torch.FloatTensor:
        return self._tensor_realizations

    @tensor_realizations.setter
    def tensor_realizations(self, tensor_realizations: torch.FloatTensor):
        # TODO, check that it is a torch tensor (not variable for example)
        self._tensor_realizations = tensor_realizations

    def set_tensor_realizations_element(self, element: torch.FloatTensor, dim: tuple[int, ...]):
        """
        Manually change the value (in-place) of `tensor_realizations` at dimension `dim`.
        """
        # TODO, check that it is a torch tensor (not variable for example) when assigning
        self._tensor_realizations[dim] = element

    def __str__(self):
        s = f"Realization of {self.name}\n"
        s += f"Shape : {self.shape}\n"
        s += f"Variable type : {self.variable_type} ({self.dist_name})"
        return s

    def set_autograd(self):
        """
        Set autograd for tensor of realizations

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
            self._tensor_realizations.requires_grad_(True) # in-place
        else:
            raise ValueError("Realizations are already using autograd")

    def unset_autograd(self):
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
        if self._tensor_realizations.requires_grad_:
            #self._tensor_realizations = self._tensor_realizations.detach()
            self._tensor_realizations.requires_grad_(False) # in-place (or `detach_()` )
        else:
            raise ValueError("Realizations are already detached")

    def __deepcopy__(self, memo) -> Realization:
        """
        Deep-copy the Realization object (magic method invoked with using copy.deepcopy)

        It clones the underlying tensor and detach it from the computational graph

        Returns
        -------
        `Realization`
        """
        return Realization.from_tensor(
            name=self.name,
            shape=self.shape,
            dist_name=self.dist_name,
            variable_type=self.variable_type,
            tensor_realizations=self.tensor_realizations
        )
