"""Module defining gaussian noise models."""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, ClassVar
from abc import abstractclassmethod
from dataclasses import dataclass
import math

import torch

from .base import DistributionFamily, BaseNoiseModel
from leaspy.models.utilities import compute_std_from_variance
from leaspy.exceptions import LeaspyInputError

if TYPE_CHECKING:
    from leaspy.io.data.dataset import Dataset


TWO_PI = torch.tensor(2 * math.pi)


class GaussianFamily(DistributionFamily):
    """For Gaussian noise models."""

    free_parameters = frozenset({"scale"})
    factory = torch.distributions.Normal

    def validate_scale(self, scale: torch.Tensor) -> torch.FloatTensor:
        """Scale parameter validation (may be extended in children classes)."""
        scale = scale.float()
        if (scale <= 0).any():
            raise LeaspyInputError(
                "The noise `scale` parameter should be > 0, "
                f"which is not the case for {scale}."
            )
        return scale


@dataclass
class AbstractGaussianNoiseModel(GaussianFamily, BaseNoiseModel):
    """Base class for Gaussian noise models."""

    scale_dimension: Optional[int] = None

    def validate_scale(self, scale: torch.Tensor) -> torch.FloatTensor:
        """Add a size-validation for scale parameter."""
        scale = super().validate_scale(scale)
        if self.scale_dimension is not None and scale.numel() != self.scale_dimension:
            raise LeaspyInputError(
                f"You have provided a noise `scale` ({scale}) of size {scale.numel()} "
                f"whereas a size = {self.scale_dimension} was expected."
            )
        return scale

    @staticmethod
    def get_residuals(
        data: Dataset, predictions: torch.FloatTensor
    ) -> torch.FloatTensor:
        return data.mask.float() * (predictions - data.values)

    @classmethod
    def compute_l2_residuals(
        cls, data: Dataset, predictions: torch.FloatTensor
    ) -> torch.FloatTensor:
        res = cls.get_residuals(data, predictions)
        return res * res

    def _get_noise_var_in_dimension(self, dimension: int) -> torch.FloatTensor:
        self.raise_if_partially_defined()
        noise_var = self.parameters["scale"] ** 2
        # shape: (n_individuals, n_visits, n_features)
        return noise_var.expand((1, 1, dimension))

    def _compute_nll_from_residuals(
        self,
        data: Dataset,
        residuals: torch.FloatTensor,
        *,
        incl_const: bool = True,
        with_gradient: bool = False,
    ) -> torch.FloatTensor:
        """Return negative log-likelihood (without summation), and optionally its jacobian w.r.t prediction."""
        noise_var = self._get_noise_var_in_dimension(data.dimension)
        nll = 0.5 / noise_var * residuals * residuals
        if incl_const:
            nll += (
                0.5
                * torch.log(TWO_PI * noise_var)
                * data.mask.float()
            )
        if not with_gradient:
            return nll
        return nll, residuals / noise_var

    def compute_nll(
        self,
        data: Dataset,
        predictions: torch.FloatTensor,
        *,
        with_gradient: bool = False,
    ) -> torch.FloatTensor:
        """Negative log-likelihood without summation (and its gradient w.r.t. predictions if requested)."""
        residuals = self.get_residuals(data, predictions)
        return self._compute_nll_from_residuals(
            data, residuals, incl_const=True, with_gradient=with_gradient
        )

    def compute_sufficient_statistics(
        self, data: Dataset, predictions: torch.FloatTensor
    ):
        """Compute the specific sufficient statistics and metrics for this noise-model."""
        predictions = data.mask.float() * predictions
        return {
            "obs_x_reconstruction": data.values * predictions,
            "reconstruction_x_reconstruction": predictions * predictions,
        }

    def update_parameters_from_sufficient_statistics(
        self,
        data: Dataset,
        sufficient_statistics: dict,
    ) -> None:
        """In-place update of free parameters from provided sufficient statistics."""
        noise_var = self._compute_noise_variance_from_sufficient_statistics(
            data, sufficient_statistics
        )
        self.update_parameters(
            scale=compute_std_from_variance(noise_var, varname="noise_std")
        )

    @abstractclassmethod
    def _compute_noise_variance_from_sufficient_statistics(
        cls,
        data: Dataset,
        sufficient_statistics: dict,
    ) -> torch.FloatTensor:
        ...

    def update_parameters_from_predictions(
        self, data: Dataset, predictions: torch.FloatTensor
    ) -> None:
        """In-place update of free parameters from provided predictions."""
        self.update_parameters(scale=self.compute_rmse(data, predictions))

    @classmethod
    def compute_rmse(
        cls, data: Dataset, predictions: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Computes root mean squared error of provided data vs. predictions."""
        l2_res = cls.compute_l2_residuals(data, predictions)
        mse = cls._compute_mse_from_l2_residuals(data, l2_res)
        return torch.sqrt(mse)

    @abstractclassmethod
    def _compute_mse_from_l2_residuals(
        cls,
        data: Dataset,
        l2_residuals: torch.FloatTensor,
    ) -> torch.FloatTensor:
        ...

    canonical_loss_properties: ClassVar = ("standard-deviation of the noise", ".2%")

    @classmethod
    def compute_canonical_loss(
        cls,
        data: Dataset,
        predictions: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute a human-friendly overall loss (RMSE)."""
        return cls.compute_rmse(data, predictions)


class GaussianScalarNoiseModel(AbstractGaussianNoiseModel):
    """Class implementing scalar Gaussian noise models."""

    scale_dimension: int = 1

    def validate_scale(self, scale: torch.Tensor) -> torch.Tensor:
        return super().validate_scale(scale).view(())

    @classmethod
    def _compute_mse_from_l2_residuals(
        cls, data: Dataset, l2_res: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Also sum on features."""
        return l2_res.sum() / data.n_observations

    @classmethod
    def _compute_noise_variance_from_sufficient_statistics(
        cls,
        data: Dataset,
        sufficient_statistics: dict,
    ) -> torch.FloatTensor:
        """Sum on features."""
        s1 = sufficient_statistics["obs_x_reconstruction"].sum()
        s2 = sufficient_statistics["reconstruction_x_reconstruction"].sum()
        return (data.L2_norm - 2.0 * s1 + s2) / data.n_observations


class GaussianDiagonalNoiseModel(AbstractGaussianNoiseModel):
    """Class implementing diagonal Gaussian noise models."""

    def validate_scale(self, scale: torch.Tensor) -> torch.Tensor:
        return super().validate_scale(scale).view(-1)

    @classmethod
    def _compute_mse_from_l2_residuals(
        cls, data: Dataset, l2_res: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Do not sum on features."""
        return l2_res.sum(dim=(0, 1)) / data.n_observations_per_ft.float()

    @classmethod
    def _compute_noise_variance_from_sufficient_statistics(
        cls,
        data: Dataset,
        sufficient_statistics: dict,
    ) -> torch.FloatTensor:
        """Do not sum on features."""
        s1 = sufficient_statistics["obs_x_reconstruction"].sum(dim=(0, 1))
        s2 = sufficient_statistics["reconstruction_x_reconstruction"].sum(dim=(0, 1))
        return (
            data.L2_norm_per_ft - 2.0 * s1 + s2
        ) / data.n_observations_per_ft.float()
