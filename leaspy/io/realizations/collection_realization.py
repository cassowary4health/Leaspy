from __future__ import annotations
from typing import TYPE_CHECKING
import copy

from leaspy.io.realizations.realization import Realization

from leaspy.utils.typing import ParamType, Dict, List

if TYPE_CHECKING:
    from leaspy.models.abstract_model import AbstractModel

# type alias for reuse
DictReals = Dict[ParamType, Realization]


class CollectionRealization:
    """
    Realizations of population and individual parameters.
    """
    def __init__(self):
        self.realizations: DictReals = {}

        self.reals_pop_variable_names: List[ParamType] = []
        self.reals_ind_variable_names: List[ParamType] = []

    def initialize(self, n_individuals: int, model: AbstractModel, *,
                   scale_individual: float = 1.):
        """
        Initialize the Collection Realization with a model.

        Parameters
        ----------
        n_individuals : int
            Number of individuals modelled
        model : :class:`.AbstractModel`
            Model we initialize from
        scale_individual : float > 0
            Multiplicative factor to scale the std-dev of individual parameters.
            cf. :meth:`.Realization.initialize`
        """
        # Indices
        infos = model.random_variable_informations()
        for variable, info_variable in infos.items():
            realization = Realization(info_variable['name'], info_variable['shape'], info_variable['type'])
            realization.initialize(n_individuals, model, scale_individual=scale_individual)
            self.realizations[variable] = realization

        # Name of variables per type
        self.reals_pop_variable_names = [name for name, info_variable in infos.items() if
                                         info_variable['type'] == 'population']
        self.reals_ind_variable_names = [name for name, info_variable in infos.items() if
                                         info_variable['type'] == 'individual']

    def __getitem__(self, variable_name: ParamType):
        return self.realizations[variable_name]

    # TODO: implement __getattr__ to delegate to realizations dictionary almost all methods

    def keys(self):
        """Return all variable names."""
        return self.realizations.keys()

    def values(self):
        """Return all realization objects."""
        return self.realizations.values()

    def items(self):
        """Return all pairs of variable name / realization object."""
        return self.realizations.items()

    def clone_realizations(self) -> CollectionRealization:
        """
        Deep-copy of self instance.

        In particular the underlying realizations are cloned and detached.

        Returns
        -------
        `CollectionRealization`
        """
        new_realizations = CollectionRealization()

        new_realizations.reals_pop_variable_names = copy.copy(self.reals_pop_variable_names)
        new_realizations.reals_ind_variable_names = copy.copy(self.reals_ind_variable_names)
        new_realizations = copy.deepcopy(self.realizations)

        return new_realizations
