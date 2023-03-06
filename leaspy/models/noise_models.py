import abc

import torch
import math
from typing import Callable, TypeVar
from torch.distributions.constraints import unit_interval

from leaspy.io.data.dataset import Dataset


T = TypeVar('T')
TWO_PI = torch.tensor(2 * math.pi)
DEFAULT_NOISE_STD = .1


class MultinomialDistribution(torch.distributions.Distribution):
    """Class for a multinomial distribution with only sample method.

    Parameters
    ----------
    sf : torch.FloatTensor
        Values of the survival function [P(X > l) for l=0..L-1 where L is max_level]
        from which the distribution samples.
        Ordinal levels are assumed to be in the last dimension.
        Those values must be in [0, 1], and decreasing when ordinal level increases (not checked).

    Attributes
    ----------
    cdf : torch.FloatTensor
        The cumulative distribution function [P(X <= l) for l=0..L] from which the distribution samples.
        The shape of latest dimension is L+1 where L is max_level.
        We always have P(X <= L) = 1
    """
    arg_constraints = {}
    validate_args = False

    def __init__(self, sf: torch.Tensor):
        super().__init__()
        assert unit_interval.check(sf).all(), "Bad probabilities in MultinomialDistribution"
        # shape of the sample (we discard the last dimension, used to store the different ordinal levels)
        self._sample_shape = sf.shape[:-1]
        # store the cumulative distribution function with trailing P(X <= L) = 1
        self.cdf = torch.cat((1. - sf, torch.ones((*self._sample_shape, 1))), dim=-1)

    @classmethod
    def from_pdf(cls, pdf: torch.Tensor):
        """Generate a new MultinomialDistribution from its probability density
        function instead of its survival function.
        """
        from .utils import compute_ordinal_sf_from_ordinal_pdf

        return cls(compute_ordinal_sf_from_ordinal_pdf(pdf))

    def sample(self):
        """Multinomial sampling.

        We sample uniformly on [0, 1( but for the latest dimension corresponding
        to ordinal levels this latest dimension will be broadcast when comparing
        with `cdf`.

        Returns
        -------
        out : torch.IntTensor
            Vector of integer values corresponding to the multinomial sampling.
            Result is in [[0, L]]
        """
        r = torch.rand(self._sample_shape).unsqueeze(-1)
        out = (r < self.cdf).int().argmax(dim=-1)
        return out


def constant_return_factory(x: T) -> Callable[[], T]:
    """Helper function to return a function returning the input value."""
    def constant_return():
        return x
    return constant_return


def convert_scalar_to_1d_tensors(value) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        return torch.tensor(value).float().view(-1)
    return value


class BaseNoiseModel(abc.ABC):
    """Base class for noise models.

    Attributes
    ----------
    name : str
        Name of the model.

    is_ordinal : bool
        Whether the noise model is ordinal or not.

    distribution : torch.distributions.Distribution
        The distribution the noise model samples from.
    """
    _valid_distribution_parameters = ()

    def __init__(self, name: str, **kwargs):
        self.name = name
        self._is_ordinal = None
        self._distribution = None
        self._distribution_parameters = {}

    @property
    def is_ordinal(self):
        return self._is_ordinal

    @property
    def distribution(self):
        return self._distribution

    @property
    def distribution_parameters(self):
        return self._distribution_parameters

    def update_distribution_parameters(self, parameters: dict):
        import warnings

        for k, v in parameters.items():
            if k in self._valid_distribution_parameters:
                self._distribution_parameters[k] = v
            else:
                warnings.warn(f"Cannot set parameter {k} for model {self.name}.")

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

    @staticmethod
    @abc.abstractmethod
    def compute_log_likelihood(self, data: Dataset, prediction: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_attachment(self, data: Dataset, prediction: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError


class BernouilliNoiseModel(BaseNoiseModel):
    """Class implementing Bernouilli noise models."""
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._distribution = torch.distributions.bernoulli.Bernoulli
        self._is_ordinal = False

    @staticmethod
    def compute_log_likelihood(self, data: Dataset, prediction: torch.FloatTensor) -> torch.FloatTensor:
        return (
            data.values * torch.log(prediction)
            + (1. - data.values) * torch.log(1. - prediction)
        )

    def compute_attachment(self, data: Dataset, prediction: torch.FloatTensor) -> torch.FloatTensor:
        attachment = -torch.sum(
            data.mask.float() * self.compute_log_likelihood(data, prediction),
            dim=(1, 2),
        )
        return attachment.reshape((data.n_individuals,))


class AbstractGaussianNoiseModel(abc.ABC, BaseNoiseModel):
    """Base class for Gaussian noise models."""
    _valid_distribution_parameters = ("scale",)

    def __init__(self, name: str, noise_std: torch.Tensor, **kwargs):
        super().__init__(name, **kwargs)
        self._is_ordinal = False
        if (noise_std <= 0).any():
            raise ValueError(
                "The noise `scale` parameter should be > 0, "
                f"which is not the case for {noise_std}."
            )
        self._distribution = torch.distributions.normal.Normal
        self.update_distribution_parameters({"scale": noise_std})

    @staticmethod
    def compute_l2_residuals_per_individual_per_feature(
            self, data: Dataset, prediction: torch.FloatTensor
    ) -> torch.FloatTensor:
        r1 = data.mask.float() * (prediction - data.values)
        return (r1 * r1).sum(dim=1)

    def compute_attachment(self, data: Dataset, prediction: torch.FloatTensor) -> torch.FloatTensor:
        noise_var = self.distribution_parameters['scale'] ** 2
        noise_var = noise_var.expand((1, data.dimension))
        attachment = (0.5 / noise_var) @ self.compute_l2_residuals_per_individual_per_feature.t()
        attachment += 0.5 * torch.log(TWO_PI * noise_var) @ data.n_observations_per_ind_per_ft.float().t()
        return attachment.reshape((data.n_individuals,))

    @abc.abstractmethod
    def compute_rmse(self, data: Dataset, predictions: torch.FloatTensor) -> torch.Tensor:
        raise NotImplementedError


class GaussianScalarNoiseModel(AbstractGaussianNoiseModel):
    """Class implementing scalar Gaussian noise models."""
    def __init__(self, name: str, noise_std=DEFAULT_NOISE_STD, **kwargs):
        noise_std = convert_scalar_to_1d_tensors(noise_std)
        if noise_std.numel() != 1:
            raise ValueError(
                f"You have provided a noise `scale` ({noise_std}) of dimension {noise_std.numel()} "
                "whereas the `noise_struct` = 'gaussian_scalar' you requested requires a "
                "univariate scale (e.g. `scale = 0.1`)."
            )
        super().__init__(name, noise_std, **kwargs)

    def compute_rmse(self, data: Dataset, predictions: torch.FloatTensor) -> torch.Tensor:
        return torch.sqrt(
            self.compute_l2_residuals_per_individual_per_feature(
                data, predictions
            ).sum(dim=1).sum(dim=0) / data.n_observations
        )


class GaussianDiagonalNoiseModel(AbstractGaussianNoiseModel):
    """Class implementing diagonal Gaussian noise models."""
    def __init__(self, name: str, noise_std=DEFAULT_NOISE_STD, **kwargs):
        super().__init__(name, convert_scalar_to_1d_tensors(noise_std), **kwargs)

    def compute_rmse(self, data: Dataset, predictions: torch.FloatTensor) -> torch.Tensor:
        return torch.sqrt(
            self.compute_l2_residuals_per_individual_per_feature(data, predictions).sum(dim=0) /
            data.n_observations_per_ft.float()
        )


class AbstractOrdinalNoiseModel(abc.ABC, BaseNoiseModel):
    """Base class for Ordinal noise models."""
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._is_ordinal = True

    def compute_attachment(self, data: Dataset, prediction: torch.FloatTensor) -> torch.FloatTensor:
        attachment = -torch.sum(
            data.mask.float() * self.compute_log_likelihood(data, prediction),
            dim=(1, 2),
        )
        return attachment.reshape((data.n_individuals,))


class OrdinalNoiseModel(AbstractOrdinalNoiseModel):
    """Class implementing ordinal noise models."""
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._distribution = MultinomialDistribution.from_pdf

    @staticmethod
    def compute_log_likelihood(self, data: Dataset, prediction: torch.FloatTensor) -> torch.Tensor:
        pdf = data.get_one_hot_encoding(sf=False, ordinal_infos=self.ordinal_infos)

        return torch.log((prediction * pdf).sum(dim=-1))


class OrdinalRankingNoiseModel(AbstractOrdinalNoiseModel):
    """Class implementing ordinal ranking noise models."""
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._distribution = MultinomialDistribution

    @staticmethod
    def compute_log_likelihood(self, data: Dataset, prediction) -> torch.Tensor:
        sf = data.get_one_hot_encoding(sf=True, ordinal_infos=self.ordinal_infos)
        cdf = (1. - sf) * self.ordinal_infos['mask']

        return (sf * torch.log(prediction) + cdf * torch.log(1. - prediction)).sum(dim=-1)
