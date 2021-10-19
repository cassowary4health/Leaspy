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

        Idem that :meth:`.initialize_from_values` method except it calls :meth:`.Realization.initialize` with ``scale_individual=1`` by default.

        Parameters
        ----------
        n_individuals : int
            Number of individuals modelled
        model : :class:`.AbstractModel`
            Model we initialize from
        scale_individual : float
            Individual scale, cf. :meth:`.Realization.initialize`
        """
        # Indices
        infos = model.random_variable_informations()
        for variable, info_variable in infos.items():
            realization = Realization(info_variable['name'], info_variable['shape'], info_variable['type'])
            realization.initialize(n_individuals, model, scale_individual=scale_individual)  ## TODO Check with Raphael
            self.realizations[variable] = realization

        # Name of variables per type
        self.reals_pop_variable_names = [name for name, info_variable in infos.items() if
                                         info_variable['type'] == 'population']
        self.reals_ind_variable_names = [name for name, info_variable in infos.items() if
                                         info_variable['type'] == 'individual']

    def initialize_from_values(self, n_individuals: int, model: AbstractModel):
        """cf. `initialize`"""
        return self.initialize(n_individuals, model, scale_individual=0.01)

    def __getitem__(self, variable_name: ParamType):
        return self.realizations[variable_name]

    def to_dict(self):
        """
        Returns 2 dictionaries with realizations

        Returns
        -------
        reals_pop : dict[var_name: str, :class:`torch.FloatTensor`]
            Realizations of population variables
        reals_ind : dict[var_name: str, :class:`torch.FloatTensor`]
            Realizations of individual variables
        """
        reals_pop: DictReals = {}
        for pop_var in self.reals_pop_variable_names:
            reals_pop[pop_var] = self.realizations[pop_var].tensor_realizations

        reals_ind: DictReals = {}
        for ind_var in self.reals_ind_variable_names:
            reals_ind[ind_var] = self.realizations[ind_var].tensor_realizations

        return reals_pop, reals_ind

    def keys(self):
        """
        Return all variable names
        """
        return self.realizations.keys()

    def copy(self):
        """
        Copy of self instance

        Returns
        -------
        `CollectionRealization`
        """
        new_realizations = CollectionRealization()

        new_realizations.reals_pop_variable_names = copy.copy(self.reals_pop_variable_names)
        new_realizations.reals_ind_variable_names = copy.copy(self.reals_ind_variable_names)
        new_realizations = copy.deepcopy(self.realizations)

        return new_realizations
