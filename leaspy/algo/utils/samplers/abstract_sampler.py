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

    Attributes
    ----------
    acceptation_temp : :class:`torch.Tensor`
        Acceptation rate for the sampler in MCMC-SAEM algorithm
        Keep the history of the last `temp_length` last steps
    name : str
        Name of variable
    shape : tuple
        Shape of variable
    temp_length : int
        Deepness of the history kept in the acceptation rate `acceptation_temp`
        Length of the `acceptation_temp` torch tensor
    ind_param_dims_but_individual : tuple[int, ...], optional (only for individual variable)
        The dimension(s) to aggregate when computing regularity of individual parameters
        For now there's only one extra dimension whether it's tau, xi or sources
        but in the future it could be extended. We do not sum first dimension (=0) which
        will always be the dimension reserved for individuals.

    Raises
    ------
    :exc:`.LeaspyModelInputError`
    """

    def __init__(self, info: KwargsType, n_patients: int):

        self.name: str = info["name"]
        self.shape: Tuple[int, ...] = info["shape"]
        self.temp_length: int = 25  # For now the same between pop and ind #TODO this is an hyperparameter

        self.ind_param_dims_but_individual: Optional[Tuple[int, ...]] = None

        if info["type"] == "population":
            self.type = 'pop'
            # Initialize the acceptation history
            if len(self.shape) not in {1, 2}:
                # convention: shape of pop variable is 1D or 2D
                raise LeaspyModelInputError("Dimension of population variable should be 1 or 2")
            else:
                full_shape = (self.temp_length, *self.shape)
                self.acceptation_temp = torch.zeros(full_shape)

        elif info["type"] == "individual":
            self.type = 'ind'
            # Initialize the acceptation history
            if len(self.shape) != 1:
                raise LeaspyModelInputError("Dimension of individual variable should be 1")
            # <!> We do not take into account the dimensionality of individual parameter for acceptation rate
            full_shape = (self.temp_length, n_patients)
            self.acceptation_temp = torch.zeros(full_shape)

            # The dimension(s) to sum when computing regularity of individual parameters
            # For now there's only one extra dimension whether it's tau, xi or sources
            # but in the future it could be extended. We never sum dimension 0 which
            # will always be the individual dimension.
            self.ind_param_dims_but_individual = tuple(range(1, 1 + len(self.shape)))  # for now it boils down to (1,)
        else:
            raise LeaspyModelInputError(f"Unknown variable type '{info['type']}': nor 'population' nor 'individual'.")

    def sample(self, data: Dataset, model: AbstractModel, realizations: CollectionRealization, temperature_inv: float, **attachment_computation_kws):
        """
        Sample new realization (either population or individual) for a given realization state, dataset, model and temperature

        <!> Modifies in-place the realizations object,
        <!> as well as the model through its `update_MCMC_toolbox` for population variables.

        Parameters
        ----------
        data : :class:`.Dataset`
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
        """
        if self.type == 'pop':
            self._sample_population_realizations(data, model, realizations, temperature_inv, **attachment_computation_kws)
        else:
            self._sample_individual_realizations(data, model, realizations, temperature_inv, **attachment_computation_kws)

    @abstractmethod
    def _sample_population_realizations(self, data, model, realizations, temperature_inv, **attachment_computation_kws):
        pass

    @abstractmethod
    def _sample_individual_realizations(self, data, model, realizations, temperature_inv, **attachment_computation_kws):
        pass

    def _group_metropolis_step(self, alpha: torch.FloatTensor) -> torch.FloatTensor:
        """
        Compute the acceptance decision (0. for False & 1. for True).

        Parameters
        ----------
        alpha : :class:`torch.FloatTensor`

        Returns
        -------
        accepted : :class:`torch.FloatTensor`, same shape as `alpha`
            Acceptance decision (0. or 1.).
        """
        # TODO: avoid sampling at indices where it is not needed (i.e. those where alpha >= 1)
        accepted = torch.rand(alpha.size()) < alpha
        return accepted.float()

    def _metropolis_step(self, alpha) -> bool:
        """
        Compute the Metropolis acceptance decision
        If better (alpha>=1): accept
        If worse (alpha<1): accept with probability alpha

        Parameters
        ----------
        alpha : :class:`torch.Tensor`

        Returns
        -------
        bool
            acceptance decision (False or True)
        """
        if alpha >= 1:
            # Case 1: we improved the LogL
            return True
        else:
            # Case 2: we decreased the LogL
            # Sample a realization from uniform law
            # Choose to keep iff realization is < alpha (proba alpha)
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
        old_acceptation_temp = self.acceptation_temp[1:]

        if self.type == "pop":
            self.acceptation_temp = torch.cat([old_acceptation_temp, accepted.unsqueeze(0)])
        elif self.type == "ind":
            self.acceptation_temp = torch.cat([old_acceptation_temp, accepted.unsqueeze(0)])
        else:
            raise LeaspyModelInputError(f"Unknown variable type '{self.type}': nor 'pop' nor 'ind'.")
