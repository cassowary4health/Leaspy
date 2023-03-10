"""Module defining the Bernoulli noise model."""

import torch

from leaspy.io.data.dataset import Dataset
from .base import BaseNoiseModel


class BernoulliNoiseModel(BaseNoiseModel):
    """Class implementing Bernoulli noise models."""
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
