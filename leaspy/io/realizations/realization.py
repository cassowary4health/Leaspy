from __future__ import annotations
from typing import TYPE_CHECKING
from scipy import stats
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
    def __init__(self, name: ParamType, shape: Tuple[int, ...], variable_type: str):
        self.name = name
        self.shape = shape
        self.variable_type = variable_type
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
        realization.tensor_realizations = tensor_realization.clone().detach()
        return realization

    def initialize(self,dataset, model: AbstractModel, *, individual_variable_init_at_mean: bool = False):
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
        n_individuals = dataset.n_individuals
        if self.variable_type == "population":
            self._tensor_realizations = model.parameters[self.name].reshape(self.shape) # avoid 0D / 1D tensors mix
        elif self.variable_type == 'individual':
            if True:  # individual_variable_init_at_mean:
                if self.name == 'tau':
                    time_w_nan = torch.where(dataset.timepoints == 0, torch.tensor(float('nan')), dataset.timepoints)
                    self._tensor_realizations = time_w_nan.nanmean(1).reshape((n_individuals, *self.shape))
                    # print(time_w_nan.nanmean(1).shape, self.shape)
                elif self.name == 'xi':
                    list_slope = []
                    # print(dataset.timepoints.shape, dataset.values.shape)
                    for x, y in zip(dataset.timepoints, dataset.values[:, :, 0]):
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x[x != 0], y[x != 0])
                        # print(slope)
                        list_slope.append((torch.tensor(slope).clamp(min=0.001) * 4).log())
                    # print(list_slope)
                    torch_slope = torch.tensor(list_slope).reshape((n_individuals, *self.shape)).float()
                    torch_slope = torch.where(torch_slope.isnan(), model.parameters[f"{self.name}_mean"], torch_slope)
                    self._tensor_realizations = torch_slope
                else:
                    raise

            else:
                distribution = torch.distributions.normal.Normal(loc=model.parameters[f"{self.name}_mean"],
                                                                scale=model.parameters[f"{self.name}_std"])
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

    def set_tensor_realizations_element(self, element: torch.FloatTensor, dim: tuple[int, ...]):
        """
        Manually change the value (in-place) of `tensor_realizations` at dimension `dim`.
        """
        # TODO, check that it is a torch tensor (not variable for example) when assigning
        self._tensor_realizations[dim] = element

    def __str__(self):
        s = f"Realization of {self.name}\n"
        s += f"Shape : {self.shape}\n"
        s += f"Variable type : {self.variable_type}"
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
        return Realization.from_tensor(self.name, self.shape, self.variable_type, self.tensor_realizations)
