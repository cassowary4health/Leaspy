import abc

import torch
import math
from typing import Callable, TypeVar, Optional
from torch.distributions.constraints import unit_interval
from typing import Union

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
        from leaspy.models.utils.ordinal import compute_ordinal_sf_from_ordinal_pdf

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
    is_ordinal : bool
        Whether the noise model is ordinal or not.

    distribution : torch.distributions.Distribution
        The distribution the noise model samples from.
    """
    _valid_distribution_parameters = ()

    def __init__(self, **kwargs):
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

    def set_distribution_to_none(self):
        """Will set the noise distribution to None.
        Use this method to turn a noise model instance
        into a dummy model sampling the provided values.
        """
        self._distribution = None

    def update_distribution_parameters(self, parameters: dict):
        import warnings

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


class BernoulliNoiseModel(BaseNoiseModel):
    """Class implementing Bernouilli noise models."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._distribution = torch.distributions.bernoulli.Bernoulli
        self._is_ordinal = False

    def compute_log_likelihood_from_dataset(
            self,
            data: Dataset,
            prediction: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return self.compute_log_likelihood(data.values, prediction)

    def compute_log_likelihood(self, values: torch.FloatTensor, prediction: torch.FloatTensor) -> torch.FloatTensor:
        prediction = torch.clamp(prediction, 1e-7, 1. - 1e-7)
        return values * torch.log(prediction) + (1. - values) * torch.log(1. - prediction)

    def compute_attachment(self, data: Dataset, prediction: torch.FloatTensor) -> torch.FloatTensor:
        attachment = -torch.sum(
            data.mask.float() * self.compute_log_likelihood(data.values, prediction),
            dim=(1, 2),
        )
        return attachment.reshape((data.n_individuals,))

    def compute_objective(self, values: torch.FloatTensor, predicted: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        nan_mask = torch.isnan(values)
        neg_cross_entropy = self.compute_log_likelihood(values, predicted)
        neg_cross_entropy[nan_mask] = 0.0
        return -torch.sum(neg_cross_entropy)

    def compute_gradient(
        self,
        values: torch.FloatTensor,
        predicted: torch.FloatTensor,
        grads: torch.FloatTensor,
    ) -> torch.Tensor:
        nan_mask = torch.isnan(values)
        predicted = torch.clamp(predicted, 1e-7, 1. - 1e-7)
        diff = predicted - values
        diff[nan_mask] = 0.0
        cross_entropy_fact = diff / (predicted * (1. - predicted))
        return torch.sum(cross_entropy_fact.unsqueeze(-1) * grads, dim=(0, 1))


class AbstractGaussianNoiseModel(BaseNoiseModel, abc.ABC):
    """Base class for Gaussian noise models."""
    _valid_distribution_parameters = ("scale",)

    def __init__(self, noise_std: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self._is_ordinal = False
        self._distribution = torch.distributions.normal.Normal
        self.set_noise_std(noise_std)

    def check_noise_std(self, noise_std: torch.Tensor) -> torch.Tensor:
        if (noise_std <= 0).any():
            raise ValueError(
                "The noise `scale` parameter should be > 0, "
                f"which is not the case for {noise_std}."
            )
        return noise_std

    def set_noise_std(self, noise_std: torch.Tensor) -> None:
        self.update_distribution_parameters({"scale": self.check_noise_std(noise_std)})

    def compute_l2_residuals_per_individual_per_feature(
            self, data: Dataset, prediction: torch.FloatTensor
    ) -> torch.FloatTensor:
        r1 = data.mask.float() * (prediction - data.values)
        return (r1 * r1).sum(dim=1)

    def get_noise_var_in_dimension(self, dimension: int) -> torch.FloatTensor:
        noise_var = self.distribution_parameters['scale'] ** 2
        return noise_var.expand((1, dimension))

    def compute_attachment(self, data: Dataset, prediction: torch.FloatTensor) -> torch.FloatTensor:
        noise_var = self.get_noise_var_in_dimension(data.dimension)
        attachment = (0.5 / noise_var) @ self.compute_l2_residuals_per_individual_per_feature(data, prediction).t()
        attachment += 0.5 * torch.log(TWO_PI * noise_var) @ data.n_observations_per_ind_per_ft.float().t()
        return attachment.reshape((data.n_individuals,))

    def compute_log_likelihood_from_dataset(self, data: Dataset, prediction: torch.FloatTensor) -> torch.FloatTensor:
        return self.compute_rmse(data, prediction)

    @abc.abstractmethod
    def compute_rmse(self, data: Dataset, predictions: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    def get_sufficient_statistics(self, data: Dataset, prediction: torch.FloatTensor) -> dict:
        prediction *= data.mask.float()
        statistics = super().get_sufficient_statistics(data, prediction)
        statistics.update(
            {
                "obs_x_reconstruction": data.values * prediction,
                "reconstruction_x_reconstruction": prediction * prediction,
            }
        )
        return statistics

    def check_sufficient_statistics(self, sufficient_statistics: dict) -> None:
        for stat in ("obs_x_reconstruction", "reconstruction_x_reconstruction"):
            if stat not in sufficient_statistics:
                raise ValueError(
                    f"Could not find the {stat} in the provided "
                    f"sufficient statistics: {sufficient_statistics}."
                )

    def get_updated_parameters_from_sufficient_statistics(
            self,
            data: Dataset,
            sufficient_statistics: dict,
    ) -> dict:
        from .utilities import compute_std_from_variance
        parameters = super().get_updated_parameters_from_sufficient_statistics(data, sufficient_statistics)
        self.check_sufficient_statistics(sufficient_statistics)
        noise_var = self.compute_noise_variance_from_sufficient_statistics(data, sufficient_statistics)
        parameters["noise_std"] = compute_std_from_variance(noise_var, varname='noise_std')
        return parameters

    @abc.abstractmethod
    def compute_noise_variance_from_sufficient_statistics(
            self,
            data: Dataset,
            sufficient_statistics: dict,
    ) -> torch.FloatTensor:
        raise NotImplementedError

    def get_parameters(self, data: Dataset, prediction: torch.FloatTensor) -> dict:
        parameters = super().get_parameters(data, prediction)
        parameters["noise_std"] = self.compute_rmse(data, prediction)
        return parameters

    def compute_objective(self, values: torch.FloatTensor, predicted: torch.FloatTensor, **kwargs) -> torch.Tensor:
        model_dimension = kwargs.pop("model_dimension", 1)
        nan_mask = torch.isnan(values)
        diff = predicted - values
        diff[nan_mask] = 0.0
        noise_var = self.get_noise_var_in_dimension(model_dimension)
        return torch.sum((0.5 / noise_var) @ (diff * diff).t())

    def compute_gradient(
            self,
            values: torch.FloatTensor,
            predicted: torch.FloatTensor,
            grads: torch.FloatTensor,
            **kwargs,
    ) -> torch.Tensor:
        model_dimension = kwargs.pop("model_dimension", 1)
        nan_mask = torch.isnan(values)
        diff = predicted - values
        diff[nan_mask] = 0.0
        noise_var = self.get_noise_var_in_dimension(model_dimension)
        return torch.sum((diff / noise_var).unsqueeze(-1) * grads, dim=(0, 1))


class GaussianScalarNoiseModel(AbstractGaussianNoiseModel):
    """Class implementing scalar Gaussian noise models."""
    def __init__(self, noise_std=DEFAULT_NOISE_STD, **kwargs):
        super().__init__(noise_std, **kwargs)

    def check_noise_std(self, noise_std: torch.Tensor) -> torch.Tensor:
        noise_std = convert_scalar_to_1d_tensors(noise_std)
        if noise_std.numel() != 1:
            raise ValueError(
                f"You have provided a noise `scale` ({noise_std}) of dimension {noise_std.numel()} "
                "whereas the `noise_struct` = 'gaussian_scalar' you requested requires a "
                "univariate scale (e.g. `scale = 0.1`)."
            )
        return super().check_noise_std(noise_std)

    def compute_rmse(self, data: Dataset, predictions: torch.FloatTensor) -> torch.FloatTensor:
        return torch.sqrt(
            self.compute_l2_residuals_per_individual_per_feature(
                data, predictions
            ).sum(dim=1).sum(dim=0) / data.n_observations
        )

    def compute_noise_variance_from_sufficient_statistics(
            self,
            data: Dataset,
            sufficient_statistics: dict,
    ) -> torch.FloatTensor:
        s1 = sufficient_statistics['obs_x_reconstruction'].sum()
        s2 = sufficient_statistics['reconstruction_x_reconstruction'].sum()
        return (data.L2_norm - 2. * s1 + s2) / data.n_observations


class GaussianDiagonalNoiseModel(AbstractGaussianNoiseModel):
    """Class implementing diagonal Gaussian noise models."""
    def __init__(self, noise_std=DEFAULT_NOISE_STD, **kwargs):
        super().__init__(noise_std, **kwargs)

    def check_noise_std(self, noise_std: torch.Tensor) -> torch.Tensor:
        return super().check_noise_std(convert_scalar_to_1d_tensors(noise_std))

    def compute_rmse(self, data: Dataset, predictions: torch.FloatTensor) -> torch.FloatTensor:
        return torch.sqrt(
            self.compute_l2_residuals_per_individual_per_feature(data, predictions).sum(dim=0) /
            data.n_observations_per_ft.float()
        )

    def compute_noise_variance_from_sufficient_statistics(
            self,
            data: Dataset,
            sufficient_statistics: dict,
    ) -> torch.FloatTensor:
        s1 = sufficient_statistics['obs_x_reconstruction'].sum(dim=(0, 1))
        s2 = sufficient_statistics['reconstruction_x_reconstruction'].sum(dim=(0, 1))
        return (data.L2_norm_per_ft - 2. * s1 + s2) / data.n_observations_per_ft.float()


class AbstractOrdinalNoiseModel(BaseNoiseModel, abc.ABC):
    """Base class for Ordinal noise models."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_ordinal: bool = True
        self.batch_deltas: bool = False
        self.max_level: Optional[int] = None
        self.features: list = []
        self.mask: Optional[torch.FloatTensor] = None

    @property
    def ordinal_infos(self) -> dict:
        return {
            "batch_deltas": self.batch_deltas,
            "max_level": self.max_level,
            "features": self.features,
            "mask": self.mask,
        }

    def configure_from_parameters(self, parameters: dict, **kwargs) -> None:
        """Set the ordinal info from model parameters."""
        model_features = kwargs.pop("features", [])
        deltas_p = self._check_delta_parameters(parameters, model_features)
        if self.batch_deltas:
            self._set_features_and_max_level_from_parameters_batch_deltas(parameters, model_features)
        else:
            self._set_features_and_max_level_from_deltas(deltas_p)
        self.build_mask()

    def _check_delta_parameters(self, parameters: dict, model_features: list) -> dict:
        deltas_p = {k: v for k, v in parameters.items() if k.startswith('deltas')}
        expected = {'deltas'} if self.batch_deltas else {f'deltas_{ft}' for ft in model_features}
        if deltas_p.keys() != expected:
            raise ValueError(
                f"Unexpected delta parameters. Expected {expected} but got {deltas_p.keys()}"
            )
        return deltas_p

    def _set_features_and_max_level_from_parameters_batch_deltas(
            self,
            parameters: dict,
            model_features: list,
    ) -> None:
        undef_levels = torch.isinf(parameters['deltas'])
        self.max_level = undef_levels.shape[1] + 1
        for i, feat in enumerate(model_features):
            undef_levels_ft = undef_levels[i, :]
            max_lvl_ft = self.max_level
            if undef_levels_ft.any():
                max_lvl_ft = undef_levels_ft.int().argmax().item() + 1
            self.features.append({"name": feat, "max_level": max_lvl_ft})

    def _set_features_and_max_level_from_deltas(self, deltas: dict) -> None:
        for feat, v in deltas.items():
            self.features.append(
                {"name": feat.lstrip("deltas_"), "max_level": v.shape[0] + 1}
            )
        self.max_level = max([feat["max_level"] for feat in self.features])

    def build_mask(self):
        """
        Build the mask to account for possible difference in levels per feature.

        The shape of the mask is (1, 1, dimension, ordinal_max_level).
        """
        self.mask = torch.cat([
            torch.cat([
                torch.ones((1, 1, 1, feat['max_level'])),
                torch.zeros((1, 1, 1, self.max_level - feat['max_level'])),
            ], dim=-1) for feat in self.features
        ], dim=2)

    def compute_attachment(self, data: Dataset, prediction: torch.FloatTensor) -> torch.FloatTensor:
        attachment = -torch.sum(
            data.mask.float() * self.compute_log_likelihood_from_dataset(data, prediction),
            dim=(1, 2),
        )
        return attachment.reshape((data.n_individuals,))

    @abc.abstractmethod
    def compute_log_likelihood(self, values: torch.FloatTensor, prediction: torch.FloatTensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_objective(self, values: torch.FloatTensor, predicted: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        nan_mask = torch.isnan(values)
        predicted = torch.clamp(predicted, 1e-7, 1. - 1e-7)
        log_likelihood = self.compute_log_likelihood(values, predicted)
        log_likelihood[nan_mask[..., 0]] = 0.0
        return -torch.sum(log_likelihood)


class OrdinalNoiseModel(AbstractOrdinalNoiseModel):
    """Class implementing ordinal noise models."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._distribution = MultinomialDistribution.from_pdf

    def compute_log_likelihood_from_dataset(self, data: Dataset, prediction: torch.FloatTensor) -> torch.Tensor:
        pdf = data.get_one_hot_encoding(sf=False, ordinal_infos=self.ordinal_infos)
        return self.compute_log_likelihood(pdf, prediction)

    def compute_log_likelihood(self, values: torch.FloatTensor, prediction: torch.FloatTensor) -> torch.Tensor:
        return torch.log((values * prediction).sum(dim=-1))

    def compute_gradient(
            self,
            values: torch.FloatTensor,
            predicted: torch.FloatTensor,
            grads: torch.FloatTensor,
    ) -> torch.Tensor:
        nan_mask = torch.isnan(values)
        predicted = torch.clamp(predicted, 1e-7, 1. - 1e-7)
        log_likelihood_grad_fact = values / predicted
        log_likelihood_grad_fact[nan_mask] = 0.0
        grad = torch.sum(log_likelihood_grad_fact.unsqueeze(-1) * grads, dim=2)
        return -grad.sum(dim=(0, 1))


class OrdinalRankingNoiseModel(AbstractOrdinalNoiseModel):
    """Class implementing ordinal ranking noise models."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._distribution = MultinomialDistribution

    def compute_log_likelihood_from_dataset(self, data: Dataset, prediction) -> torch.Tensor:
        sf = data.get_one_hot_encoding(sf=True, ordinal_infos=self.ordinal_infos)
        return self.compute_log_likelihood(sf, prediction)

    def compute_log_likelihood(
            self,
            values: torch.FloatTensor,
            prediction: torch.FloatTensor,
            **kwargs,
    ) -> torch.Tensor:
        squeeze_mask = kwargs.pop("squeeze_mask", True)
        if squeeze_mask:
            cdf = (1. - values) * self.mask.squeeze(0)
        else:
            cdf = (1. - values) * self.mask
        return (values * torch.log(prediction) + cdf * torch.log(1. - prediction)).sum(dim=-1)

    def compute_gradient(
            self,
            values: torch.FloatTensor,
            predicted: torch.FloatTensor,
            grads: torch.FloatTensor,
    ) -> torch.Tensor:
        nan_mask = torch.isnan(values)
        predicted = torch.clamp(predicted, 1e-7, 1. - 1e-7)
        diff = predicted - values
        diff[nan_mask] = 0.0
        log_likelihood_grad_fact = -diff * self.mask.squeeze(0) / (predicted * (1. - predicted))
        grad = torch.sum(log_likelihood_grad_fact.unsqueeze(-1) * grads, dim=2)
        return -grad.sum(dim=(0, 1))


NOISE_MODELS = {
    "bernoulli": BernoulliNoiseModel,
    "gaussian-scalar": GaussianScalarNoiseModel,
    "gaussian-diagonal": GaussianDiagonalNoiseModel,
    "ordinal": OrdinalNoiseModel,
    "ordinal-ranking": OrdinalRankingNoiseModel,
}


def noise_model_factory(noise_model: Union[str, BaseNoiseModel]) -> BaseNoiseModel:
    """
    Factory for noise models.

    Parameters
    ----------
    noise_model : str or BaseNoiseModel
        If an instance of a subclass of BaseNoiseModel, returns the instance.
        If a string, then return the appropriate class.

    Returns
    -------
    BaseNoiseModel :
        The desired noise model.

    Raises
    ------
    ValueError:
        If noise_model is not supported.
    """
    if isinstance(noise_model, BaseNoiseModel):
        return noise_model
    noise_model = noise_model.lower()
    noise_model = noise_model.replace("_", "-")
    try:
        return NOISE_MODELS[noise_model]
    except KeyError:
        raise ValueError(
            f"Noise model {noise_model} is not supported."
            f"Supported noise models are : {NOISE_MODELS.keys()}"
        )
