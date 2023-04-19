import torch
from typing import List, Optional
from .base import AbstractRandomVariable
from leaspy.io.data.dataset import Dataset
from leaspy.algo.utils.samplers.abstract_sampler import AbstractSampler
from leaspy.io.realizations.collection_realization import CollectionRealization


class SampledVariable(AbstractRandomVariable):
    """
    Sampled variables class. Their evaluate method calls a sampler.
    They have no parent.

    Parameters
    ----------
    name : str
        The name of the variable
    shape : torch.Size
        The shape of the random variable
    sampler : AbstractSampler
        Sampler for the RV if already initialized
    model : :class:`~.models.abstract_model.AbstractModel`
        Model to which the RV is attached
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
    sampler : AbstractSampler
        Sampler for the RV
    """

    def __init__(
            self,
            name: str,
            shape: torch.Size,
            sampler: AbstractSampler,
            initial_realization: CollectionRealization,
            init_value: Optional[torch.Tensor] = None,
    ):
        super().__init__(name, shape, init_value=init_value)
        self.sampler = sampler
        self.realization = initial_realization

    def _evaluate(self, data: Optional[Dataset] = None) -> torch.Tensor:
        value = self.sampler.sample(data, self.model, self.realization, 1.0)
        self._cache = value
        return value

    def posterior_distribution(self, n_samples: int) -> torch.Tensor:
        """
        Returns an array of samples.

        Parameters
        ----------
        n_samples : int
            Number of samples from posterior

        Returns
        -------
        torch.Tensor :
            a stacked array of samples
        """
        samples = [self._evaluate() for _ in range(n_samples)]
        return torch.stack(samples, dim=0)
