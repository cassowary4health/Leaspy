from __future__ import annotations
from typing import List, Callable, Iterable, TYPE_CHECKING
import torch

from .realization import (
    AbstractRealization,
    IndividualRealization,
    PopulationRealization,
)
from leaspy.utils.typing import DictParamsTorch
if TYPE_CHECKING:
    from leaspy.models.abstract_model import AbstractModel


class CollectionRealization:
    """
    Realizations of population and individual parameters.
    """
    def __init__(self):
        self.realizations: List[AbstractRealization] = []

    @property
    def population(self) -> List[PopulationRealization]:
        """
        Return the list of realizations for population variables.
        """
        return [r for r in self.realizations if isinstance(r, PopulationRealization)]

    @property
    def population_tensors(self) -> List[torch.Tensor]:
        """
        Return the list of tensor realizations for population variables.
        """
        return [r.tensor_realizations for r in self.population]

    @property
    def population_tensors_dict(self) -> DictParamsTorch:
        """
        Return the dictionary 'variable_name:tensor_realization' for population variables.
        """
        return {r.name: r.tensor_realizations for r in self.population}

    @property
    def individual(self) -> List[IndividualRealization]:
        """
        Return the list of realizations for individual variables.
        """
        return [r for r in self.realizations if isinstance(r, IndividualRealization)]

    @property
    def individual_tensors(self) -> List[torch.Tensor]:
        """
        Return the list of tensor realizations for individual variables.
        """
        return [r.tensor_realizations for r in self.individual]

    @property
    def individual_tensors_dict(self) -> DictParamsTorch:
        """
        Return the dictionary 'variable_name:tensor_realization' for individual variables.
        """
        return {r.name: r.tensor_realizations for r in self.individual}

    @property
    def names(self) -> List[str]:
        """
        Return the list of variable names whatever their type.
        """
        return [r.name for r in self.realizations]

    @property
    def population_names(self) -> List[str]:
        """
        Return the list of population variable names.
        """
        return [r.name for r in self.population]

    @property
    def individual_names(self) -> List[str]:
        """
        Return the list of individual variable names.
        """
        return [r.name for r in self.individual]

    def get_tensor_by_name(self, variable_name: str) -> torch.Tensor:
        """
        Return a tensor for the realization having the provided variable name.

        Parameters
        ----------
        variable_name : str
            The name of the variable for which to query the associated realization tensor.

        Returns
        -------
        torch.Tensor :
            The realization tensor associated with the variable name.
        """
        return self.get_by_name(variable_name).tensor_realizations

    def get_tensor(self, variables: Iterable[str]) -> List[torch.Tensor]:
        """
        Get a list of tensor for realizations having their name in the provided list.

        Parameters
        ----------
        variables : list of str
            The names of the variables for which to query the realization tensors.

        Returns
        -------
        List[torch.Tensor] :
            The realization tensors.
        """
        return [self.get_tensor_by_name(v) for v in variables]

    def get_tensor_dict(self, variables: Iterable[str]) -> DictParamsTorch:
        """
        Get a dictionary of 'variable_name:tensor_realizations' for variables
        having their name in the provided list.

        Parameters
        ----------
        variables : list of str
        The names of the variables for which to query the realization tensors.

        Returns
        -------
        DictParamsTorch :
        The dictionary holding variable names as keys and realization tensors as values.
        """
        return {v: self.get_tensor_by_name(v) for v in variables}

    def get_by_name(self, variable_name: str) -> AbstractRealization:
        """
        Get a specific Realization instance by its name.

        Parameters
        ----------
        variable_name : str
            The name of the variable for which to get the corresponding realization.

        Returns
        -------
        AbstractRealization :
            The Realization instance.

        Raises
        ------
        ValueError :
            If the provided variable name does not match any realization's name.
        """
        for r in self.realizations:
            if r.name == variable_name:
                return r
        raise ValueError(
            "The CollectionRealization instance does not have a Realization "
            f"matching the provided name {variable_name}."
        )

    def initialize(
        self,
        n_individuals: int,
        model: AbstractModel,
        skip_variable: Callable[[dict], bool] = None,
        **realization_init_kws,
    ) -> None:
        """
        Initialize the CollectionRealization instance from a Model instance.

        Parameters
        ----------
        n_individuals : int
            The number of individuals in the data.

        model : AbstractModel
            The model from which to initialize the collection of realizations.

        skip_variable : Callable or bool, optional
            Whether some variables should be skipped or not.

        realization_init_kws : dict
            Kwargs for initializing the Realizations.
        """
        self.initialize_population(model, skip_variable, **realization_init_kws)
        self.initialize_individuals(n_individuals, model, skip_variable, **realization_init_kws)

    def initialize_population(
        self,
        model: AbstractModel,
        skip_variable: Callable[[dict], bool] = None,
        **realization_init_kws,
    ) -> None:
        """
        Initialize the population part of the CollectionRealization instance from a Model instance.
        """
        info = model.get_population_random_variable_information()
        for variable, info_variable in info.items():
            if skip_variable is not None and skip_variable(info_variable):
                continue
            realization = PopulationRealization(
                info_variable["name"],
                info_variable["shape"],
            )
            realization.initialize(model, **realization_init_kws)
            self.realizations.append(realization)

    def initialize_individuals(
        self,
        n_individuals: int,
        model: AbstractModel,
        skip_variable: Callable[[dict], bool] = None,
        **realization_init_kws,
    ) -> None:
        """
        Initialize the individual part of the CollectionRealization instance from a Model instance.
        """
        info = model.get_individual_random_variable_information()
        for variable, info_variable in info.items():
            if skip_variable is not None and skip_variable(info_variable):
                continue
            realization = IndividualRealization(
                info_variable["name"],
                info_variable["shape"],
                n_individuals=n_individuals,
            )
            realization.initialize(model, **realization_init_kws)
            self.realizations.append(realization)


def clone_realizations(realizations: CollectionRealization) -> CollectionRealization:
    """
    Deep-copy of provided realizations.

    In particular the underlying realizations are cloned and detached.

    Parameters
    ----------
    realizations : CollectionRealization
        The collection of realizations to be cloned.

    Returns
    -------
    CollectionRealization :
        The cloned collection of realizations.
    """
    import copy

    new = CollectionRealization()
    new.realizations = [copy.deepcopy(r) for r in realizations.realizations]

    return new
