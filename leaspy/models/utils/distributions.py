from dataclasses import dataclass

import torch
from scipy import stats


@dataclass
class AsymmetricLaplace:
    """
    Asymmetric Laplace distribution (not implemented yet in PyTorch).

    Parameters
    ----------
    kappa, loc, scale : :class:`torch.FloatTensor`
        Parameters of the distribution in canonical form (lambda = 1/scale).

    Attributes
    ----------
    kappa, loc, scale : :class:`torch.FloatTensor`
        Parameters of the distribution in canonical form (lambda = 1/scale).
    base_dist : scipy distribution
        Scipy distribution to generate samples.
    """
    kappa: torch.FloatTensor
    loc: torch.FloatTensor
    scale: torch.FloatTensor

    @staticmethod
    def _to_numpy(t):
        if isinstance(t, torch.Tensor):
            return t.cpu().numpy()
        return t

    def __post_init__(self):
        assert (self.kappa > 0).all()
        assert (self.scale > 0).all()
        self.base_dist = stats.laplace_asymmetric(
            kappa=self._to_numpy(self.kappa),
            loc=self._to_numpy(self.loc),
            scale=self._to_numpy(self.scale),
        )

    @property
    def variance(self):
        """Variance of distribution (same interface as PyTorch distributions)."""
        return torch.tensor(self.base_dist.var(), dtype=torch.float32)

    def icdf(self, q):
        """Inverse cumultative distribution function (same interface as PyTorch distributions)."""
        return torch.tensor(self.base_dist.ppf(self._to_numpy(q)), dtype=torch.float32)

    @staticmethod
    def get_canonical_parameters_of_exp_centered_dist(asym) -> dict:
        """
        Get canonical ALD parameters for the custom sub-family of distributions exponentially-centered (i.e. E[exp(X)] = 1) & with mode = 0.

        For that sub-family we only have 1 degree of fredom, through parameter that we named `asym` (= "lambda * kappa - 1/2").

        Parameters
        ----------
        asym : :class:`torch.FloatTensor` (or other array-like type supporting operations)
            The custom `asymmetry` parameter, that completely parametrizes the sub-family of ALD distributions that verifies:
            - mode[X] = 0
            - E[exp(X)] = 1

        Returns
        -------
        dict[str, *]
            The canonical parameters of an ALD distribution
        """
        assert (asym > 0.5).all()
        params = dict(
            loc=0.,
            scale=(asym**2 - 1/4)**(-.5),  # inverse of "lambda"
        )
        params['kappa'] = (asym + 0.5) * params['scale']
        return params

    @classmethod
    def get_exp_centered_dist(cls, asym):
        """Get the distribution for custom sub-family of exponentially-centered."""
        return cls(**cls.get_canonical_parameters_of_exp_centered_dist(asym))

    def sample(self, sample_shape: tuple) -> torch.FloatTensor:
        """Get torch samples from this distribution."""
        return torch.tensor(self.base_dist.rvs(size=sample_shape), dtype=torch.float32)


def get_distribution(dist_name: str, var_name: str, model_params: dict):
    """Get a sampler (return torch samples) for the given variable."""
    if 'gaussian' in dist_name:
        return torch.distributions.normal.Normal(
            loc=model_params[f"{var_name}_mean"], scale=model_params[f"{var_name}_std"]
        )
    elif dist_name == 'ALD_exp_centered':
        return AsymmetricLaplace.get_exp_centered_dist(model_params[f"{var_name}_asym"])
    else:
        raise NotImplementedError(dist_name)

