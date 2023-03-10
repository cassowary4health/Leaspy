"""Module defining ordinal noise models."""

import abc
import torch

from typing import Optional

from leaspy.io.data.dataset import Dataset
from .base import BaseNoiseModel
from .distributions import MultinomialDistribution


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
