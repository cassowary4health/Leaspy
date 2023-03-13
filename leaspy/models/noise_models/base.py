"""BaseNoiseModel defines the common interface for noise models in Leaspy."""

import abc
import torch
import warnings

from typing import Callable, TypeVar

from leaspy.io.data.dataset import Dataset


T = TypeVar('T')


def constant_return_factory(x: T) -> Callable[[], T]:
    """Helper function to return a function returning the input value."""
    def constant_return():
        return x
    return constant_return


class BaseNoiseModel(abc.ABC):
    """Base class for noise models.

    Attributes
    ----------
    distribution : torch.distributions.Distribution
        The distribution the noise model samples from.
    """
    _valid_distribution_parameters = ()

    def __init__(self, **kwargs):
        self._distribution = None
        self._distribution_parameters = {}


    @property
    def distribution(self):
        return self._distribution

    @property
    def distribution_parameters(self):
        return self._distribution_parameters

    def set_distribution_to_none(self):
        """Will set the noise distribution to None.
        Use this method to turn a noise model instance
        into a dummy model sampling the provided values.
        """
        self._distribution = None

    def update_distribution_parameters(self, parameters: dict):
        for k, v in parameters.items():
            if k in self._valid_distribution_parameters:
                self._distribution_parameters[k] = v
            else:
                warnings.warn(f"Cannot set parameter {k} for noise model.")

    def configure_from_parameters(self, parameters: dict, **kwargs) -> None:
        """Configure noise model with model parameters."""
        pass

    def sample_around(self, values: torch.FloatTensor) -> torch.FloatTensor:
        """Realization around `values` with respect to noise model."""
        return self.sampler_around(values)()

    def sampler_around(self, loc: torch.FloatTensor) -> Callable[[], torch.FloatTensor]:
        """Return the noise sampling function around input values."""
        if self.distribution is None:
            return constant_return_factory(loc)
        return self.rv_around(loc).sample

    def rv_around(self, loc: torch.FloatTensor) -> torch.distributions.Distribution:
        """Return the torch distribution centred around values (only if noise is not None)."""
        if self.distribution is None:
            raise ValueError('Random variable around values is undefined when there is no noise.')

        return self.distribution(loc, **self._distribution_parameters)

    @abc.abstractmethod
    def compute_log_likelihood_from_dataset(
            self,
            data: Dataset,
            prediction: torch.FloatTensor,
    ) -> torch.FloatTensor:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_attachment(self, data: Dataset, prediction: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_objective(self, values: torch.FloatTensor, predicted: torch.FloatTensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_gradient(
        self,
        values: torch.FloatTensor,
        predicted: torch.FloatTensor,
        grads: torch.FloatTensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_sufficient_statistics(self, data: Dataset, prediction: torch.FloatTensor) -> dict:
        return {"log-likelihood": self.compute_log_likelihood_from_dataset(data, prediction)}

    def get_parameters(self, data: Dataset, prediction: torch.FloatTensor) -> dict:
        return {"log-likelihood": self.compute_attachment(data, prediction).sum()}

    def get_updated_parameters_from_sufficient_statistics(
            self,
            data: Dataset,
            sufficient_statistics: dict,
    ) -> dict:
        if "log-likelihood" not in sufficient_statistics:
            raise ValueError(
                "Could not find the log likelihood in the provided "
                f"sufficient statistics: {sufficient_statistics}."
            )
        return {"log-likelihood": sufficient_statistics["log-likelihood"].sum()}
