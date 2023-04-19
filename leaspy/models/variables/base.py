import torch
from typing import Optional, List, Iterable
from abc import abstractmethod
from leaspy.io.data.dataset import Dataset


class AbstractRandomVariable:
    """
    Variables class. They have a tree-like architecture.
    They depend on leaf variables (SampledVariables) which are
    sampled directly while LinkedVariables are evaluated by
    a bottom-up computation of the tree.

    Parameters
    ----------
    name : str
        The name of the variable
    shape : torch.Size
        The shape of the random variable
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
    """

    def __init__(self, name: str, shape: torch.Size, init_value: Optional[torch.Tensor] = None):
        self.name = name
        self.shape = shape
        self._cache = init_value
        self.sons: VariableCollection = VariableCollection()
        self.parents: VariableCollection = VariableCollection()

    def get_value(self, data: Optional[Dataset] = None) -> torch.Tensor:
        """
        Returns the current value of the variable (evaluates if needed).

        Parameters
        ----------
        data : Dataset, optional
            The dataset on which to sample if evaluating the
            variable requires sampling.

        Returns
        -------
        torch.Tensor : the current value of the variable
        """
        if self._cache is None:
            self._evaluate(data)
        return self._cache

    @abstractmethod
    def _evaluate(self, data: Optional[Dataset] = None) -> torch.Tensor:
        """
        Computes the value of the variable depending on the
        state of depending variables.

        Parameters
        ----------
        data : Dataset, optional
            The dataset on which to sample if evaluating the
            variable requires sampling.

        Returns
        -------
        torch.Tensor: the new value of the variable
        """

    def update(self, data: Optional[Dataset] = None):
        """
        Evaluates the variable and updates all depending variables recursively.

        Parameters
        ----------
        data : Dataset, optional
            The dataset on which to sample if evaluating the
            variable requires sampling.
        """
        old = self._cache
        new = self._evaluate(data)
        if new != old:
            self.reset()
            self._cache = new
            # will not reset because cache is None,
            # so only evaluates and propagates to the grandsons
            self.sons.update()

    def reset(self):
        """
        Erases the cache as well as all sons' cache.
        """
        # check if cache is already deleted to avoid
        # any loops (should not exist in proper design)
        if self._cache is not None:
            self._cache = None
            self.sons.reset()


class VariableCollection:
    def __init__(self):
        self._variables: List[AbstractRandomVariable] = []

    @property
    def names(self):
        return [variable.name for variable in self._variables]

    def get_values(self):
        return [variable.get_value() for variable in self._variables]

    def apply(self, func):
        return [func(variable.get_value() for variable in self._variables)]

    def has_variable(self, other: AbstractRandomVariable) -> bool:
        """Return True if the collection already has a variable with the same name."""
        return other.name in self.names

    def add(self, new: AbstractRandomVariable):
        if self.has_variable(new):
            raise ValueError("Variable already in collection !")
        self._variables.append(new)

    def extend(self, variables: Iterable[AbstractRandomVariable]):
        for variable in variables:
            self.add(variable)

    def remove(self, name: str):
        pass

    def update(self):
        for variable in self._variables:
            variable.update()

    def reset(self):
        for variable in self._variables:
            variable.reset()

    def __len__(self):
        return len(self._variables)

    def __iter__(self):
        return self._variables.__iter__()
