from __future__ import annotations
from typing import TYPE_CHECKING
import warnings

import torch

from leaspy.exceptions import LeaspyAlgoInputError, LeaspyInputError, LeaspyModelInputError
from leaspy.utils.typing import TypeVar, KwargsType, Tuple, Callable, Optional, DictParamsTorch

if TYPE_CHECKING:
    from leaspy.io.data.dataset import Dataset
    from leaspy.models.abstract_model import AbstractModel


T = TypeVar('T')
def constant_return_factory(x: T) -> Callable[[], T]:
    """Helper function to return a function returning the input value."""
    def constant_return():
        return x
    return constant_return


class NoiseModel:
    """
    Helper class to define and work with different noise structures in models.

    TODO? It may be of interest to define an abstract noise structure class with
    all methods needed to be transparently integrated in models and algorithms
    and then to create children classes of it (scalar gaussian, diagonal gaussian,
    more complex gaussian noise structures, Bernoulli realization, ...)

    Parameters
    ----------
    noise_struct : str, optional
        Noise structure requested. Multiple options:
            * None: no noise at all (default)
            * 'bernoulli': Bernoulli realization
            * 'gaussian_scalar': Gaussian noise with scalar std-dev, to give as `scale` parameter
            * 'gaussian_diagonal': Gaussian noise with 1 std-dev per feature (<!> order), to give as `scale` parameter
    **noise_kws : :class:`torch.FloatTensor`, optional
        Only needed and expected for Gaussian noise: the std-dev requested for noise.

    Attributes
    ----------
    name : str
        Correspond to the parameter `noise_struct`
    distribution_factory : function [torch.Tensor, **kws] -> torch.distributions.Distribution
        A function taking a :class:`torch.Tensor` of values first, possible keyword arguments
        and returning a noise generator (instance of class :class:`torch.distributions.Distribution`),
        which can sample around these values with respect to noise structure.
    distributions_kws : dict[str, Any]
        Extra keyword parameters to be passed to `distribution_factory` apart the centering values.

    Raises
    ------
    :exc:`.LeaspyInputError`
        If `noise_sruct` is not supported.
    """

    """Mapping from naming of noise parameter in Leaspy model to the related torch distribution parameters."""
    model_params_to_distributions_kws = {
        'noise_std': 'scale',  # useful for Gaussian distribution
    }

    """Valid structures for noise."""
    VALID_NOISE_STRUCT = {'bernoulli', 'gaussian_scalar', 'gaussian_diagonal'}

    """For backward-compatibility only."""
    OLD_MAPPING_FROM_LOSS = {
        'MSE': 'gaussian_scalar',
        'MSE_diag_noise': 'gaussian_diagonal',
        'crossentropy': 'bernoulli'
    }


    @classmethod
    @property
    def distribution_kws_to_model_params(cls):
        """Mapping from torch distribution parameters to the related noise parameter naming in Leaspy model."""
        return {v: k for k, v in cls.model_params_to_distributions_kws.items()}

    def __init__(self, noise_struct: str = None, **noise_kws):

        self.name = noise_struct
        self.distribution_factory: Optional[Callable[..., torch.distributions.Distribution]] = None
        self.distributions_kws: KwargsType = {}

        noise_struct_supported = False
        noise_kws_keys = set(noise_kws.keys())

        # Various possibilities for noise structure
        if noise_struct is None:
            noise_struct_supported = True
            if noise_kws_keys:
                raise LeaspyAlgoInputError(f"`noise_struct` = None should not have {noise_kws_keys} parameters.")

        elif noise_struct == 'bernoulli':
            noise_struct_supported = True
            self.distribution_factory = torch.distributions.bernoulli.Bernoulli
            if noise_kws_keys:
                raise LeaspyAlgoInputError(f"`noise_struct` = 'bernoulli' should not have {noise_kws_keys} parameters.")

        elif 'gaussian' in noise_struct:
            # 'gaussian_scalar' or 'gaussian_diagonal' for now
            self.distribution_factory = torch.distributions.normal.Normal

            if noise_kws_keys != {'scale'}:
                raise LeaspyAlgoInputError("Only `scale` (= noise std-dev) is expected for Gaussian noise.")

            if not isinstance(noise_kws['scale'], torch.Tensor):
                noise_kws = torch.tensor(noise_kws['scale'])
            noise_scale = noise_kws['scale'].view(-1)
            self.distributions_kws = {'scale': noise_scale}

            # 1 noise per feature (manually specified)
            if noise_struct == 'gaussian_scalar':
                noise_struct_supported = True

                if len(noise_scale) != 1:
                    raise LeaspyInputError(f"You have provided a noise `scale` ({noise_scale}) of dimension {len(noise_scale)} whereas the "
                                           "`noise_struct` = 'gaussian_scalar' you requested requires a univariate scale (e.g. `scale = 0.1`).")

            elif noise_struct == 'gaussian_diagonal':
                # allow univariate scale with 'gaussian_diagonal'
                noise_struct_supported = True


        if not noise_struct_supported:
            raise LeaspyInputError(f"`noise_struct` = '{noise_struct}' is not supported. "
                                   f"Please use one noise structure among {self.VALID_NOISE_STRUCT} or None.")

    @property
    def scale(self) -> Optional[torch.FloatTensor]:
        """A quick short-cut for scale of Gaussian noises."""
        return self.distributions_kws.get('scale', None)

    @classmethod
    def from_model(cls, model: AbstractModel, noise_struct: str = 'model', **noise_kws):
        """
        Initialize a noise model as in the regular initialization but with special keywords to derive it from model own noise.

        It also automatically performs some consistency checks between noise provided and model.

        Parameters
        ----------
        model : :class:`~.AbstractModel`, optional
            The model you want to generate noise for.
            Only used when inheriting noise structure or to perform checks on Gaussian diagonal noise.
        noise_struct : str, optional (default 'model')
            Noise structure requested. Multiple options:
                * 'model': use the noise structure from model, as well as the noise parameters from model (if any)
                * 'inherit_struct' (or deprecated 'default'): use the noise structure from model provided
                (but not the actual parameters of noise from model, if any)
                * All other regular noise structures supported (cf. class docstring)
        noise_kws :
            Extra parameters for noise (cf. class docstring)
            Not to be used when `noise_struct` = 'model' (default)

        Returns
        -------
        :class:`.NoiseModel`
        """

        if noise_struct in ['inherit_struct', 'model', 'default']:

            if noise_struct == 'default':
                warnings.warn("`noise_struct` = 'default' is deprecated and will soon be dropped due to ambiguity, "
                              "use 'inherit_struct' instead for same behavior.", FutureWarning)

            if noise_struct == 'model':
                if noise_kws:
                    raise LeaspyAlgoInputError("Extra keyword arguments to specify noise should NOT be provided "
                                               "when `noise_struct` = 'model' in NoiseModel.from_model.")

                # use all noise parameters directly from model parameters (if any)
                # for now there is only 'noise_std' available and for gaussian noise only
                # TODO: we could also only use them as default values, possibly overwritten by noise_kws
                noise_kws = {
                    cls.model_params_to_distributions_kws[model_param]: model_val
                    for model_param, model_val in model.parameters.items()
                    if model_param in cls.model_params_to_distributions_kws
                    and not (model_val is None or model.noise_model == 'bernoulli')
                    # previous line needed because `noise_std` is included in 'bernoulli' models even if not part of true model parameters!
                }

            # substitute 'noise_struct' str with model one ("structure" only, not parameters unless if it was 'model')
            noise_struct = model.noise_model

        # Instantiate noise model normally (with the special keywords having been substituted)
        noise_gen = cls(noise_struct, **noise_kws)

        # Check the compatibility with model
        noise_gen.check_compat_with_model(model)

        return noise_gen

    def check_compat_with_model(self, model: AbstractModel):
        """Raise if `noise_model` is not compatible with `model` (consistency checks)."""

        # only check 'gaussian_diagonal' for now
        if self.name == 'gaussian_diagonal':
            noise_scale_numel = self.scale.numel()
            if noise_scale_numel != model.dimension:
                raise LeaspyInputError(
                        "You requested a 'gaussian_diagonal' noise. However, the attribute `scale` you gave has "
                        f"{noise_scale_numel} elements, which mismatches with model dimension of {model.dimension}. "
                        f"Please give a list of std-dev for every features {model.features}, in order.")

    def rv_around(self, loc: torch.FloatTensor) -> torch.distributions.Distribution:
        """Return the torch distribution centred around values (only if noise is not None)."""
        if self.distribution_factory is not None:
            return self.distribution_factory(loc, **self.distributions_kws)

    def sampler_around(self, loc: torch.FloatTensor) -> Callable[[], torch.FloatTensor]:
        """Return the noise sampling function around input values."""
        if self.distribution_factory is None:
            # No noise: return raw values (no copy)
            return constant_return_factory(loc)
        else:
            return self.rv_around(loc).sample

    def sample_around(self, model_loc_values: torch.FloatTensor) -> torch.FloatTensor:
        """Realization around `model_loc_values` with respect to noise model."""
        return self.sampler_around(model_loc_values)()  # Better to store sampler if multiple calls needed

    @staticmethod
    def rmse_model(model: AbstractModel, dataset: Dataset, individual_params: DictParamsTorch, *,
                   scalar: bool = None, **computation_kwargs) -> torch.FloatTensor:
        """
        Helper function to compute the root mean square error (RMSE) from model reconstructions.

        Parameters
        ----------
        model : :class:`~.AbstractModel`
            Subclass object of `AbstractModel`.
        dataset : :class:`~.Dataset`
            Dataset to compute reconstruction errors from.
        individual_params : DictParamsTorch
            Object containing the computed individual parameters (torch format).
        scalar : bool or None (default)
            Should we compute a scalar RMSE (averaged on all features) or one RMSE per feature (same order)?
            If None, it will fetch `noise_model` from model and choose scalar mode iif 'scalar' in `noise_model` string.
        **computation_kwargs
            Additional kwargs for `model.compute_sum_squared_***_tensorized` method

        Returns
        -------
        :class:`torch.FloatTensor`
            The RMSE tensor (1D) of length 1 if `scalar` else `model.dimension`.
        """

        if scalar is None:
            scalar = 'scalar' in model.noise_model

        if scalar:
            sum_squared = model.compute_sum_squared_tensorized(dataset, individual_params,
                                                               **computation_kwargs).sum(dim=0)  # sum on individuals
            return torch.sqrt(sum_squared / dataset.n_observations)
        else:
            # 1 noise per feature
            sum_squared_per_ft = model.compute_sum_squared_per_ft_tensorized(dataset, individual_params,
                                                                             **computation_kwargs).sum(dim=0)   # sum on individuals
            return torch.sqrt(sum_squared_per_ft / dataset.n_observations_per_ft.float())

    @classmethod
    def _extract_noise_model_from_old_loss_for_backward_compatibility(cls, loss: Optional[str]) -> Optional[str]:
        """Only for backward-compatibility with old loss."""

        if loss is None:
            return

        noise_struct = cls.OLD_MAPPING_FROM_LOSS.get(loss, None)
        if noise_struct is None:
            raise LeaspyModelInputError(f'Old loss "{loss}" is not known and not supported anymore. '
                                        f'It should have been in {set(cls.OLD_MAPPING_FROM_LOSS.keys())}.')

        warnings.warn('`loss` hyperparameter of model is deprecated, it should be named `noise_model` from now on. '
                      f'Please replace parameter with: `noise_model` = "{noise_struct}".', FutureWarning)

        return noise_struct

    @classmethod
    def set_noise_model_from_hyperparameters(cls, model, hyperparams: KwargsType) -> Tuple[str, ...]:
        """
        Set `noise_model` of a model from hyperparameters.

        Parameters
        ----------
        model :
            Where to set noise model (in-place)
        hyperparams : dict[str, Any]
            where to look for noise model

        Returns
        -------
        tuple[str]
            Additional recognized hyperparameters for models.
        """

        # BACKWARD-COMPAT
        if 'loss' in hyperparams.keys():
            model.noise_model = cls._extract_noise_model_from_old_loss_for_backward_compatibility(hyperparams['loss'])
        # END

        if 'noise_model' in hyperparams.keys():
            if hyperparams['noise_model'] not in cls.VALID_NOISE_STRUCT:
                raise LeaspyModelInputError(f'`noise_model` should be in {cls.VALID_NOISE_STRUCT}, not "{hyperparams["noise_model"]}".')
            model.noise_model = hyperparams['noise_model']

        return ('loss', 'noise_model')
