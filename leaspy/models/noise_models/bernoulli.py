"""Module defining the Bernoulli noise model."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from .base import DistributionFamily, BaseNoiseModel

if TYPE_CHECKING:
    from leaspy.io.data.dataset import Dataset


class BernoulliFamily(DistributionFamily):
    """For Bernoulli noise model."""

    factory = torch.distributions.Bernoulli
    free_parameters = frozenset()


class BernoulliNoiseModel(BernoulliFamily, BaseNoiseModel):
    """Class implementing Bernoulli noise models."""

    def compute_nll(
        self,
        data: Dataset,
        predictions: torch.FloatTensor,
        *,
        with_gradient: bool = False,
    ) -> torch.FloatTensor:
        """Negative log-likelihood and its gradient wrt predictions."""
        predictions = torch.clamp(predictions, 1e-7, 1.0 - 1e-7)
        ll = data.values * torch.log(predictions) + (1.0 - data.values) * torch.log(
            1.0 - predictions
        )
        nll = -data.mask.float() * ll
        if not with_gradient:
            return nll
        nll_grad = (
            data.mask.float()
            * (predictions - data.values)
            / (predictions * (1.0 - predictions))
        )
        return nll, nll_grad
