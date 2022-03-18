from abc import ABC, abstractmethod

import torch

from leaspy.exceptions import LeaspyModelInputError
from leaspy.utils.typing import KwargsType, Tuple, Optional

from leaspy.io.data.dataset import Dataset
from leaspy.models.abstract_model import AbstractModel
from leaspy.io.realizations.collection_realization import CollectionRealization


class AbstractSampler(ABC):
    """
    Abstract sampler class.

    Parameters
    ----------
    info : dict[str, Any]
        The dictionary describing the random variable to sample.
        It should contains the following entries:
            * name : str
            * shape : tuple[int, ...]
            * type : 'population' or 'individual'
    n_patients : int > 0
        Number of patients (useful for individual variables)
    acceptation_history_length : int > 0 (default 25)
        Deepness (= number of iterations) of the history kept for computing the mean acceptation rate.
        (It is the same for population or individual variables.)

    Attributes
    ----------
    name : str
        Name of variable
    shape : tuple
        Shape of variable
    acceptation_history_length : int
        Deepness (= number of iterations) of the history kept for computing the mean acceptation rate.
        (It is the same for population or individual variables.)
    ind_param_dims_but_individual : tuple[int, ...], optional (only for individual variable)
        The dimension(s) to aggregate when computing regularity of individual parameters
        For now there's only one extra dimension whether it's tau, xi or sources
        but in the future it could be extended. We do not sum first dimension (=0) which
        will always be the dimension reserved for individuals.
    acceptation_history : :class:`torch.Tensor`
        History of binary acceptations to compute mean acceptation rate for the sampler in MCMC-SAEM algorithm.
        It keeps the history of the last `acceptation_history_length` steps.

    Raises
    ------
    :exc:`.LeaspyModelInputError`
    """

    def __init__(self, info: KwargsType, n_patients: int, *, acceptation_history_length: int = 25):

        self.name: str = info["name"]
        self.shape: Tuple[int, ...] = info["shape"]
        self.acceptation_history_length = acceptation_history_length

        self.ind_param_dims_but_individual: Optional[Tuple[int, ...]] = None

        if info["type"] == "population":
            self.type = 'pop'
            # Initialize the acceptation history
            if len(self.shape) not in {1, 2}:
                # convention: shape of pop variable is 1D or 2D
                raise LeaspyModelInputError("Dimension of population variable should be 1 or 2")
            else:
                full_shape = (self.acceptation_history_length, *self.shape)
                self.acceptation_history = torch.zeros(full_shape)

        elif info["type"] == "individual":
            self.type = 'ind'
            # Initialize the acceptation history
            if len(self.shape) != 1:
                raise LeaspyModelInputError("Dimension of individual variable should be 1")
            # <!> We do not take into account the dimensionality of individual parameter for acceptation rate
            full_shape = (self.acceptation_history_length, n_patients)
            self.acceptation_history = torch.zeros(full_shape)

            # The dimension(s) to sum when computing regularity of individual parameters
            # For now there's only one extra dimension whether it's tau, xi or sources
            # but in the future it could be extended. We never sum dimension 0 which
            # will always be the individual dimension.
            self.ind_param_dims_but_individual = tuple(range(1, 1 + len(self.shape)))  # for now it boils down to (1,)
        else:
            raise LeaspyModelInputError(f"Unknown variable type '{info['type']}': nor 'population' nor 'individual'.")

    def sample(self, dataset: Dataset, model: AbstractModel, realizations: CollectionRealization, temperature_inv: float, **attachment_computation_kws) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Sample new realization (either population or individual) for a given realization state, dataset, model and temperature

        <!> Modifies in-place the realizations object,
        <!> as well as the model through its `update_MCMC_toolbox` for population variables.

        Parameters
        ----------
        dataset : :class:`.Dataset`
            Dataset class object build with leaspy class object Data, model & algo
        model : :class:`.AbstractModel`
            Model for loss computations and updates
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
            Contain the current state & information of all the variables of interest
        temperature_inv : float > 0
            Inverse of the temperature used in tempered MCMC-SAEM
        **attachment_computation_kws
            Optional keyword arguments for attachment computations.
            As of now, we only use it for individual variables, and only `attribute_type`.
            It is used to know whether to compute attachments from the MCMC toolbox (esp. during fit)
            or to compute it from regular model parameters (esp. during personalization in mean/mode realization)

        Returns
        -------
        attachment, regularity_var : `torch.FloatTensor` 0D (population variable) or 1D (individual variable, with length `n_individuals`)
            The attachment and regularity (only for the current variable) at the end of this sampling step
            (globally or per individual, depending on variable type).
        """
        if self.type == 'pop':
            return self._sample_population_realizations(dataset, model, realizations, temperature_inv, **attachment_computation_kws)
        else:
            return self._sample_individual_realizations(dataset, model, realizations, temperature_inv, **attachment_computation_kws)

    @abstractmethod
    def _sample_population_realizations(self, data, model, realizations, temperature_inv, **attachment_computation_kws) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Sample population variables"""

    @abstractmethod
    def _sample_individual_realizations(self, data, model, realizations, temperature_inv, **attachment_computation_kws) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Sample individual variables"""

    def _group_metropolis_step(self, alpha: torch.FloatTensor) -> torch.FloatTensor:
        """
        Compute the acceptance decision (0. for False & 1. for True).

        Parameters
        ----------
        alpha : :class:`torch.FloatTensor` > 0

        Returns
        -------
        accepted : :class:`torch.FloatTensor`, same shape as `alpha`
            Acceptance decision (0. or 1.).
        """
        accepted = torch.rand(alpha.size()) < alpha
        return accepted.float()

    def _metropolis_step(self, alpha: float) -> bool:
        """
        Compute the Metropolis acceptance decision.

        If better (alpha>=1): accept
        If worse (alpha<1): accept with probability alpha

        <!> This function is critical for the reproducibility between machines.
        Different architectures might lead to different rounding errors on alpha
        (e.g: 1. - 1e-6 vs 1. + 1e-6). If we were to draw only for alpha < 1 (and not when alpha >= 1),
        then it would cause the internal seed of pytorch to change or not depending on the case
        which would lead to very different results afterwards (all the random numbers would be affected).

        Parameters
        ----------
        alpha : float > 0

        Returns
        -------
        bool
            acceptance decision (False or True)
        """
        # Sample a realization from uniform law
        # Choose to keep iff realization is < alpha (probability alpha)
        # <!> Always draw a number even if it seems "useless" (cf. docstring warning)
        return torch.rand(1).item() < alpha

    def _update_acceptation_rate(self, accepted: torch.FloatTensor):
        """
        Update acceptation rate from history of boolean accepted values for each dimension of each variable
        (except for multivariate individual parameters)

        Parameters
        ----------
        accepted : :class:`torch.FloatTensor` (0. or 1.)

        Raises
        ------
        :exc:`.LeaspyModelInputError`
        """

        # Concatenate the new acceptation result at end of new one (forgetting the oldest acceptation rate)
        old_acceptation_history = self.acceptation_history[1:]

        if self.type == "pop":
            self.acceptation_history = torch.cat([old_acceptation_history, accepted.unsqueeze(0)])
        elif self.type == "ind":
            self.acceptation_history = torch.cat([old_acceptation_history, accepted.unsqueeze(0)])
        else:
            raise LeaspyModelInputError(f"Unknown variable type '{self.type}': nor 'pop' nor 'ind'.")
