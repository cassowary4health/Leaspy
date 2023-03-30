"""Module defining ordinal noise models."""

from __future__ import annotations
from typing import Optional, Dict, Tuple, Union
from dataclasses import dataclass

import torch

from .base import BaseNoiseModel
from leaspy.utils.distributions import MultinomialDistribution
from leaspy.utils.typing import FeatureType, KwargsType
from leaspy.io.data.dataset import Dataset


@dataclass
class AbstractOrdinalNoiseModel(BaseNoiseModel):
    """
    Base class for Ordinal noise models.

    Attributes
    ----------
    max_levels : dict, optional
        Maximum levels for ordinal noise.
    """

    max_levels: Optional[Dict[FeatureType, int]] = None

    def to_dict(self) -> KwargsType:
        """
        Serialize instance as dictionary.
        Do NOT export hyper-parameters that are derived
        (error-prone and boring checks when re-creating).
        """
        return {"max_levels": self.max_levels}

    def _update_cached_hyperparameters(self) -> None:
        """
        Update hyperparameters in cache.
        """
        if self.max_levels is None:
            self._max_level: Optional[int] = None
            self._mask: Optional[torch.Tensor] = None
            return

        assert isinstance(self.max_levels, dict)
        self._max_level = max(self.max_levels.values())
        self._mask = torch.stack(
            [
                torch.cat(
                    [
                        torch.ones(ft_max_level),
                        torch.zeros(self.max_level - ft_max_level),
                    ],
                    dim=-1,
                )
                for ft_max_level in self.max_levels.values()
            ],
        )

    def __setattr__(self, name: str, val) -> None:
        super().__setattr__(name, val)
        if name == "max_levels":
            # nota: we do not use property setter logic so not to loss benefits
            # from dataclass (including repr with 'public' attribute `max_levels`)
            # source: https://stackoverflow.com/a/66412774
            self._update_cached_hyperparameters()

    @property
    def max_level(self):
        return self._max_level

    @property
    def mask(self):
        return self._mask

    @property
    def ordinal_infos(self) -> KwargsType:
        return {
            "max_levels": self.max_levels,
            "max_level": self.max_level,
            "mask": self.mask,
        }


class OrdinalNoiseModel(AbstractOrdinalNoiseModel):
    """
    Class implementing ordinal noise models (likelihood is based on PDF).

    Attributes
    ----------
    free_parameters: frozenset(str)
        Name of all the free parameters (but `loc`) needed to characterize the distribution.
        Nota: for each parameter, if a method named "validate_xxx" exists (torch.Tensor -> torch.Tensor),
        then it will be used for user-input validation of parameter "xxx".
    factory : None or function(free parameters values) -> torch.distributions.Distribution
        The factory for the distribution family.
    """

    factory = MultinomialDistribution.from_pdf
    free_parameters = frozenset()

    def compute_nll(
        self,
        data: Dataset,
        predictions: torch.Tensor,
        *,
        with_gradient: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute the negative log-likelihood and its gradient wrt predictions.

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the log likelihood.
        predictions: :class:`torch.Tensor`
            The model's predictions from which to compute the log likelihood.
        with_gradient: bool, optional
            If True, returns also the gradient of the negative log likelihood
            wrt the predictions.
            If False, only returns the negative log likelihood.
            Default=False.

        Returns
        -------
        :class:`torch.Tensor` or tuple of :class:`torch.Tensor`
            The negative log likelihood (and its jacobian if requested).
        """
        predictions = torch.clamp(predictions, 1e-7, 1.0 - 1e-7)
        pdf = data.get_one_hot_encoding(sf=False, ordinal_infos=self.ordinal_infos)
        nll = -data.mask.float() * torch.log((pdf * predictions).sum(dim=-1))
        if not with_gradient:
            return nll
        nll_grad = -data.mask[..., None].float() * pdf / predictions
        return nll, nll_grad


class OrdinalRankingNoiseModel(AbstractOrdinalNoiseModel):
    """
    Class implementing ordinal ranking noise models (likelihood is based on SF).

    Attributes
    ----------
    free_parameters: frozenset(str)
        Name of all the free parameters (but `loc`) needed to characterize the distribution.
        Nota: for each parameter, if a method named "validate_xxx" exists (torch.Tensor -> torch.Tensor),
        then it will be used for user-input validation of parameter "xxx".
    factory : None or function(free parameters values) -> torch.distributions.Distribution
        The factory for the distribution family.
    """

    factory = MultinomialDistribution
    free_parameters = frozenset()

    def compute_nll(
        self,
        data: Dataset,
        predictions: torch.Tensor,
        *,
        with_gradient: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute the negative log-likelihood and its gradient wrt predictions.

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the log likelihood.
        predictions: :class:`torch.Tensor`
            The model's predictions from which to compute the log likelihood.
        with_gradient: bool, optional
            If True, returns also the gradient of the negative log likelihood
            wrt the predictions.
            If False, only returns the negative log likelihood.
            Default=False.

        Returns
        -------
        :class:`torch.Tensor` or tuple of :class:`torch.Tensor`
            The negative log likelihood (and its jacobian if requested).
        """
        predictions = torch.clamp(predictions, 1e-7, 1.0 - 1e-7)
        sf = data.get_one_hot_encoding(sf=True, ordinal_infos=self.ordinal_infos)
        cdf = (1.0 - sf) * self.mask[None, None, ...]
        ll = (sf * torch.log(predictions) + cdf * torch.log(1.0 - predictions)).sum(
            dim=-1
        )
        nll = -data.mask.float() * ll
        if not with_gradient:
            return nll
        nll_grad = (
            data.mask[..., None].float()
            * (predictions - sf)
            / (predictions * (1.0 - predictions))
        )
        return nll, nll_grad
