import torch
from typing import List, Optional
from .base import AbstractRandomVariable, VariableCollection
from leaspy.io.data.dataset import Dataset


class LinkedVariable(AbstractRandomVariable):
    """
    Linked variables class.
    Their evaluate method calls a link function which depends on the values of other variables.
    They require parent variables to work.

    Parameters
    ----------
    name : str
        The name of the variable
    shape : torch.Size
        The shape of the random variable
    dependent_variables : Dict[str, AbstractRandomVariable]
        The list of all variables necessary for the computation of this variable
    link_function : Callable[List[torch.Tensor], torch.Tensor]
        The link function to compute the value of this variable based on the current state of depending variables
    init_value : torch.Tensor
        Initial value to be put in the cache

    Attributes
    ----------
    name : str
        The name of the variable
    _shape : torch.Size
        The shape of the variable
    cache : Union[None, torch.Tensor]
        Cache for the current value of the variable
    sons : List[AbstractRandomVariable]
        The list of variables depending on the value of this variable
    parents : List[AbstractRandomVariable]
        The list of all variables necessary for the computation of this variable
    link_function : Callable[List[torch.Tensor], torch.Tensor]
        The link function to compute the value of this variable based on the current state of depending variables
    """

    def __init__(
            self,
            name: str,
            shape: torch.Size,
            dependent_variables: List[AbstractRandomVariable],
            link_function,
            init_value: Optional[torch.Tensor] = None,
    ):
        super().__init__(name, shape, init_value=init_value)
        self.parents = VariableCollection()
        self.parents.extend(dependent_variables)
        for parent in self.parents:
            parent.sons.add(self)
        self.link_function = link_function

    def _evaluate(self, data: Optional[Dataset] = None) -> torch.Tensor:
        value = self.link_function(self.parents)
        self._cache = value
        return value


class LinkFunction:
    def __init__(self):
        pass

    @abstractmethod
    def function(self, x: VariableCollection) -> torch.Tensor:
        """
        The function to evaluate
        """

    def __call__(self, x: VariableCollection):
        return self.function(x)


class Exp(LinkFunction):
    def function(self, x: VariableCollection):
        if len(x) != 1:
            raise ValueError(
                "The Exponential LinkFunction accepts only a "
                f"single dependant variable. You provided {len(x)}."
            )
        values = x.evalute()
        return torch.exp(values[0])
