"""Module defining gaussian noise models."""

import abc
import torch
import math

from .base import BaseNoiseModel
from leaspy.io.data.dataset import Dataset


TWO_PI = torch.tensor(2 * math.pi)
DEFAULT_NOISE_STD = .1


def convert_scalar_to_1d_tensors(value) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        return torch.tensor(value).float().view(-1)
    return value


class AbstractGaussianNoiseModel(BaseNoiseModel, abc.ABC):
    """Base class for Gaussian noise models."""
    _valid_distribution_parameters = ("scale",)

    def __init__(self, noise_std: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
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
        from leaspy.models.utilities import compute_std_from_variance
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
