from __future__ import annotations
from typing import TYPE_CHECKING, Union
import warnings

import torch

from leaspy.exceptions import LeaspyInputError, LeaspyModelInputError
from leaspy.utils.typing import TypeVar, KwargsType, Tuple, Callable, Optional, DictParamsTorch
from leaspy.models.utils.noise_struct import NoiseStruct, NOISE_STRUCTS

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
    noise_struct : NoiseStruct, or str in `VALID_NOISE_STRUCTS`, or None
        The noise structure to build noise model upon.
        If this is a string, noise structure will be searched among pre-defined noise structures:
            * `'bernoulli'`: Bernoulli realization
            * `'gaussian_scalar'`: Gaussian noise with scalar std-dev, to give as `scale` parameter
            * `'gaussian_diagonal'`: Gaussian noise with 1 std-dev per feature (<!> order), to give as `scale` parameter
        If None: no noise at all (in particular, ultimately `sample(r)_around(values)` will just return `values`)
    **noise_kws
        Keyword arguments to fully characterize the noise structure.
        For now, only one parameter is supported, for Gaussian noise structures:
            * `scale` (:class:`torch.FloatTensor`): the std-dev requested for noise.

    Attributes
    ----------
    struct : NoiseStruct
        The noise structure, which contains its metadata / characteristics.
    distributions_kws : dict[str, Any]
        Extra keyword parameters to be passed to `struct.distribution_factory` apart the centering values.

    Raises
    ------
    :exc:`.LeaspyInputError`
        If `noise_struct` is not supported.

    See Also
    --------
    :meth:`NoiseModel.from_model`
    """

    """Valid structures for noise (except None)."""
    VALID_NOISE_STRUCTS = {k for k in NOISE_STRUCTS.keys() if k is not None}

    """For backward-compatibility only."""
    OLD_MAPPING_FROM_LOSS = {
        'MSE': 'gaussian_scalar',
        'MSE_diag_noise': 'gaussian_diagonal',
        'crossentropy': 'bernoulli'
    }

    def __init__(self, noise_struct: Union[NoiseStruct, str, None], **noise_kws):

        if isinstance(noise_struct, NoiseStruct):
            self.struct = noise_struct
        else:
            self.struct = self.get_named_noise_struct(noise_struct)

        noise_kws_keys = set(noise_kws.keys())
        expected_noise_kws = set(self.struct.model_kws_to_dist_kws.values())

        # Validate the noise keyword arguments that were sent
        pbs = []
        missing_parameters = expected_noise_kws.difference(noise_kws_keys)
        unexpected_parameters = noise_kws_keys.difference(expected_noise_kws)
        if missing_parameters:
            pbs.append(f'should have {missing_parameters} parameters')
        if unexpected_parameters:
            pbs.append(f'should NOT have {unexpected_parameters} parameters')
        if pbs:
            raise LeaspyInputError(f"`noise_struct` = '{noise_struct}' {' but '.join(pbs)}.")

        # validate & clean the noise parameters
        self.distributions_kws = self.struct.validate_dist_kws(noise_kws)

    @classmethod
    def get_named_noise_struct(cls, name: Optional[str]) -> NoiseStruct:
        """Helper to get a default noise structure from its name."""
        noise_struct = NOISE_STRUCTS.get(name, None)

        if noise_struct is None:
            raise LeaspyInputError(f"`noise_struct` = '{name}' is not supported. "
                                   f"Please use one among {cls.VALID_NOISE_STRUCTS} or None.")

        return noise_struct

    def check_compat_with_model(self, model: AbstractModel):
        """Raise if `noise_model` is not compatible with `model` (consistency checks)."""
        self.struct.with_contextual_validators(model=model).validate_dist_kws(self.distributions_kws)

    @classmethod
    def from_model(cls, model: AbstractModel, noise_struct: str = 'model', **noise_kws):
        """
        Initialize a noise model as in the `from_name` initialization but with special keywords
        so to easily inherit noise model from the one of an existing model.

        It also automatically performs some consistency checks between model and noise parameters provided.
        As of now, it is mainly useful for simulation algorithm.

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
        **noise_kws : any
            Extra parameters for noise (cf. class docstring)
            Not to be used when `noise_struct` = 'model' (default)

        Returns
        -------
        :class:`.NoiseModel`
        """
        get_noise_parameters_from_model = False

        if noise_struct in ['inherit_struct', 'model', 'default']:

            if noise_struct == 'default':
                warnings.warn("`noise_struct` = 'default' is deprecated and will soon be dropped due to ambiguity, "
                              "use 'inherit_struct' instead for same behavior.", FutureWarning)

            if noise_struct == 'model':
                get_noise_parameters_from_model = True

            # define the 'noise_struct' from the model one (_structure_ only, not parameters unless if it was 'model')
            noise_struct = model.noise_model

        # get default, named, noise structure
        noise_struct_obj = cls.get_named_noise_struct(noise_struct)

        # complete validators of noise structure with dynamic ones, that depend on model information
        noise_struct_obj = noise_struct_obj.with_contextual_validators(model=model)

        if get_noise_parameters_from_model:
            # TODO? we could also only use model noise parameters as default values, possibly overwritten by noise_kws
            if noise_kws:
                raise LeaspyInputError("Extra keyword arguments to specify noise should NOT be provided "
                                       "when `noise_struct` = 'model' in NoiseModel.from_model.")

            # use all noise parameters directly from model parameters (if any)
            # for now there is only 'noise_std' available and for gaussian noise only
            # the mapping of parameters is specific to a noise structure to be general
            model_kws_to_dist_kws = noise_struct_obj.model_kws_to_dist_kws
            noise_kws = {
                model_kws_to_dist_kws[model_param]: model_val
                for model_param, model_val in model.parameters.items()
                if model_param in model_kws_to_dist_kws
            }

        # Instantiate noise model normally (with the special keywords having been substituted)
        # In particular compat with model will be checked here thanks to previous dynamic validators
        return cls(noise_struct_obj, **noise_kws)

    @property
    def scale(self) -> Optional[torch.FloatTensor]:
        """A quick short-cut for scale of Gaussian noises (really useful??)."""
        return self.distributions_kws.get('scale', None)

    def rv_around(self, loc: torch.FloatTensor) -> torch.distributions.Distribution:
        """Return the torch distribution centred around values (only if noise is not None)."""
        if self.struct.distribution_factory is None:
            raise LeaspyInputError('Random variable around values is undefined when there is no noise.')

        return self.struct.distribution_factory(loc, **self.distributions_kws)

    def sampler_around(self, loc: torch.FloatTensor) -> Callable[[], torch.FloatTensor]:
        """Return the noise sampling function around input values."""
        if self.struct.distribution_factory is None:
            # No noise: return raw values (no copy)
            return constant_return_factory(loc)
        else:
            return self.rv_around(loc).sample

    def sample_around(self, model_loc_values: torch.FloatTensor) -> torch.FloatTensor:
        """Realization around `model_loc_values` with respect to noise model."""
        # <!> Better to store sampler if multiple calls needed
        return self.sampler_around(model_loc_values)()

    ## HELPER METHODS ##
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
            return None

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
        model : AbstractModel
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
            if hyperparams['noise_model'] not in cls.VALID_NOISE_STRUCTS:
                raise LeaspyModelInputError(f'`noise_model` should be in {cls.VALID_NOISE_STRUCTS}, '
                                            f'not "{hyperparams["noise_model"]}".')
            model.noise_model = hyperparams['noise_model']

        return ('loss', 'noise_model')
