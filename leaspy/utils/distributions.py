"""Module defining useful distribution for ordinal noise model."""

from typing import Union

import numpy as np
import torch
from torch.distributions.constraints import unit_interval


def discrete_sf_from_pdf(pdf: Union[torch.Tensor, np.ndarray]):
    """
    Compute the discrete survival function values [P(X > l), i.e.
    P(X >= l+1), l=0..L-1] (l=0..L-1) from the discrete probability density
    [P(X = l), l=0..L] (assuming discrete levels are in last dimension).
    """
    return (1 - pdf.cumsum(-1))[..., :-1]


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
        assert unit_interval.check(
            sf
        ).all(), "Bad probabilities in MultinomialDistribution"
        # shape of the sample (we discard the last dimension, used to store the different ordinal levels)
        self._sample_shape = sf.shape[:-1]
        # store the cumulative distribution function with trailing P(X <= L) = 1
        self.cdf = torch.cat((1.0 - sf, torch.ones((*self._sample_shape, 1))), dim=-1)

    @classmethod
    def from_pdf(cls, pdf: torch.Tensor):
        """Generate a new MultinomialDistribution from its probability density
        function instead of its survival function.
        """
        return cls(discrete_sf_from_pdf(pdf))

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