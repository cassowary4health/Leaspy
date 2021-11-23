from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from leaspy.exceptions import LeaspyModelInputError
from leaspy.utils.typing import ParamType, Tuple

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
    variable_type : str
        ``'individual'`` or ``'population'`` variable?

    Attributes
    ----------
    name : str
        Variable name
    shape : tuple of int
        Shape of variable (multiple dimensions allowed)
    variable_type : str
        ``'individual'`` or ``'population'`` variable?
    tensor_realizations : :class:`torch.Tensor`
        Actual realizations, whose shape is given by `shape`
    """
    def __init__(self, name: ParamType, shape: Tuple[int, ...], variable_type: str, rv_type: str = None):
        self.name = name
        self.shape = shape
        self.variable_type = variable_type
        self.rv_type = rv_type
        self._tensor_realizations: torch.FloatTensor = None

    @classmethod
    def from_tensor(cls, name: str, shape: Tuple[int, ...], variable_type: str, tensor_realization: torch.FloatTensor):
        """
        Create realization from variable infos and torch tensor object

        Parameters
        ----------
        name : str
            Variable name
        shape : tuple of int
            Shape of variable (multiple dimensions allowed)
        variable_type : str
            ``'individual'`` or ``'population'`` variable?
        tensor_realization : :class:`torch.Tensor`
            Actual realizations, whose shape is given by `shape`

        Returns
        -------
        :class:`.Realization`
        """
        # TODO : a check of shapes
        realization = cls(name, shape, variable_type)
        realization._tensor_realizations = tensor_realization.clone().detach()
        return realization

    def initialize(self, n_individuals: int, model: AbstractModel, scale_individual: float = 1.0):
        """
        Initialize realization from a given model.

        Parameters
        ----------
        n_individuals : int > 0
            Number of individuals
        model : :class:`.AbstractModel`
            The model you want realizations for.
        scale_individual : float > 0
            Multiplicative factor to scale the std-dev as given by model parameters

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            if unknown variable type
        """

        if self.variable_type == "population":
            self._tensor_realizations = model.parameters[self.name].reshape(self.shape) # avoid 0D / 1D tensors mix
        elif self.variable_type == 'individual':
            if self.rv_type == 'linked':
                if self.name == 'v0':
                    self._tensor_realizations: torch.Tensor = torch.zeros((n_individuals, *self.shape), device=model.device)
                elif self.name == 'tau_mean':
                    self._tensor_realizations: torch.Tensor = torch.zeros((n_individuals, *self.shape), device=model.device)
            else:
                if f"{self.name}_mean" in model.parameters:
                    distribution = torch.distributions.normal.Normal(loc=model.parameters[f"{self.name}_mean"],
                                                                 scale=scale_individual * model.parameters[f"{self.name}_std"])  # TODO change later, to have low variance when initialized
                else:
                    mean = model.get_intersept(f"{self.name}_mean")
                    distribution = torch.distributions.normal.Normal(loc=mean,
                                                                 scale=scale_individual * model.parameters[f"{self.name}_std"])  # TODO change later, to have low variance when initialized
                self._tensor_realizations = distribution.sample(sample_shape=(n_individuals, *self.shape))
        else:
            raise LeaspyModelInputError(f"Unknown variable type '{self.variable_type}'.")

    @property
    def tensor_realizations(self) -> torch.FloatTensor:
        return self._tensor_realizations

    @tensor_realizations.setter
    def tensor_realizations(self, tensor_realizations: torch.FloatTensor):
        # TODO, check that it is a torch tensor (not variable for example)
        self._tensor_realizations = tensor_realizations

    def set_tensor_realizations_element(self, element, dim: int):
        """
        Manually change the value (in-place) of `tensor_realizations` at dimension `dim`.
        """
        # TODO, check that it is a torch tensor (not variable for example) when assigning
        self._tensor_realizations[dim] = element

    def __str__(self):
        str = f"Realization of {self.name}\n"
        str += f"Shape : {self.shape}\n"
        str += f"Variable type : {self.variable_type}\n"
        return str

    def set_autograd(self):
        """
        Set autograd for tensor of realizations

        See Also
        --------
        torch.Tensor.requires_grad_

        Raises
        ------
        :class:`ValueError`
            if inconsistent internal request
        """
        if not self._tensor_realizations.requires_grad:
            self._tensor_realizations.requires_grad_(True) # in-place
        else:
            raise ValueError("Realizations are already using autograd")

    def unset_autograd(self):
        """
        Unset autograd for tensor of realizations

        See Also
        --------
        torch.Tensor.requires_grad_

        Raises
        ------
        :class:`ValueError`
            if inconsistent internal request
        """
        if self._tensor_realizations.requires_grad_:
            #self._tensor_realizations = self._tensor_realizations.detach()
            self._tensor_realizations.requires_grad_(False) # in-place (or `detach_()` )
        else:
            raise ValueError("Realizations are already detached")

    def copy(self):
        """
        Copy the Realization object

        Notes
        -----
        From PyTorch :meth:`torch.Tensor.clone` doc:
            Unlike copy_(), this function is recorded in the computation graph.
            Gradients propagating to the cloned tensor will propagate to the original tensor.
        """
        new_realization = Realization(self.name, self.shape, self.variable_type)
        new_realization.tensor_realizations = self.tensor_realizations.clone()
        return new_realization
