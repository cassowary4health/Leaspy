from __future__ import annotations

import re
import math
from abc import abstractmethod
import copy
import json

import torch
from torch._tensor_str import PRINT_OPTS as torch_print_opts

from leaspy import __version__
from leaspy.models.base import BaseModel
from leaspy.models.noise_models import (
    BaseNoiseModel,
    NoiseModelFactoryInput,
    noise_model_factory,
    noise_model_export,
)
from leaspy.models.utilities import tensor_to_list
from leaspy.io.realizations.realization import Realization
from leaspy.io.realizations.collection_realization import CollectionRealization
from leaspy.io.data.dataset import Dataset

from leaspy.exceptions import LeaspyIndividualParamsInputError, LeaspyModelInputError
from leaspy.utils.typing import (
    FeatureType,
    KwargsType,
    DictParams,
    DictParamsTorch,
    Union,
    List,
    Dict,
    Tuple,
    Iterable,
    Optional,
)


TWO_PI = torch.tensor(2 * math.pi)


#  TODO? refact so to only contain methods needed for the Leaspy api + add another
#  abstract class (interface) on top of it for MCMC fittable models + one for "manifold models"

class AbstractModel(BaseModel):
    """
    Contains the common attributes & methods of the different models.

    Parameters
    ----------
    name : str
        The name of the model
    noise_model : str or BaseNoiseModel
        The noise model for observations (keyword-only parameter).
    **kwargs
        Hyperparameters for the model

    Attributes
    ----------
    is_initialized : bool
        Indicates if the model is initialized
    name : str
        The model's name
    features : list[str]
        Names of the model features
    parameters : dict
        Contains the model's parameters
    noise_model : BaseNoiseModel
        The noise model associated to the model.
    regularization_distribution_factory : function dist params -> :class:`torch.distributions.Distribution`
        Factory of torch distribution to compute log-likelihoods for regularization (gaussian by default)
        (Not used anymore)
    fit_metrics : dict
        Contains the metrics that are measured during the fit of the model and reported to the user.
    """

    def __init__(
        self,
        name: str,
        *,
        noise_model: NoiseModelFactoryInput,
        fit_metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.parameters: Optional[KwargsType] = None
        self._noise_model: Optional[BaseNoiseModel] = None

        # load hyperparameters
        self.noise_model = noise_model
        self.load_hyperparameters(kwargs)

        # TODO: dirty hack for now, cf. AbstractFitAlgo
        self.fit_metrics = fit_metrics

    @property
    def noise_model(self):
        return self._noise_model

    @noise_model.setter
    def noise_model(self, model: NoiseModelFactoryInput):
        noise_model = noise_model_factory(model)
        self.check_noise_model_compatibility(noise_model)
        self._noise_model = noise_model

    def check_noise_model_compatibility(self, model: BaseNoiseModel) -> None:
        """
        Raise a ValueError is the provided noise model isn't compatible with the model instance.
        This needs to be implemented in subclasses.
        """
        if not isinstance(model, BaseNoiseModel):
            raise LeaspyModelInputError(
                "Expected a subclass of BaselNoiseModel, but received "
                f"a {model.__class__.__name__} instead."
            )

    @abstractmethod
    def to_dict(self) -> KwargsType:
        """
        Export model as a dictionary ready for export.
        """
        return {
            'leaspy_version': __version__,
            'name': self.name,
            'features': self.features,
            'dimension': self.dimension,
            'fit_metrics': self.fit_metrics,  # TODO improve
            'noise_model': noise_model_export(self.noise_model),
            'parameters': {
                k: tensor_to_list(v)
                for k, v in (self.parameters or {}).items()
            }
        }

    def save(self, path: str, **kwargs) -> None:
        """
        Save Leaspy object as json model parameter file.

        TODO move logic upstream?

        Parameters
        ----------
        path : str
            Path to store the model's parameters.
        **kwargs
            Keyword arguments for `to_dict` child method and `json.dump` function (default to indent=2).
        """
        from inspect import signature

        export_kws = {k: kwargs.pop(k) for k in signature(self.to_dict).parameters if k in kwargs}
        model_settings = self.to_dict(**export_kws)

        # Default json.dump kwargs:
        kwargs = {'indent': 2, **kwargs}

        with open(path, 'w') as fp:
            json.dump(model_settings, fp, **kwargs)

    def load_parameters(self, parameters: KwargsType) -> None:
        """
        Instantiate or update the model's parameters.

        Parameters
        ----------
        parameters : dict[str, Any]
            Contains the model's parameters
        """
        self.parameters = copy.deepcopy(parameters)

    @abstractmethod
    def load_hyperparameters(self, hyperparameters: KwargsType) -> None:
        """
        Load model's hyperparameters.

        Parameters
        ----------
        hyperparameters : dict[str, Any]
            Contains the model's hyperparameters

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If any of the consistency checks fail.
        """

    @classmethod
    def _raise_if_unknown_hyperparameters(cls, known_hps: Iterable[str], given_hps: KwargsType) -> None:
        """
        Raises a :exc:`.LeaspyModelInputError` if any unknown hyperparameter is provided to the model.
        """
        # TODO: replace with better logic from GenericModel in the future
        unexpected_hyperparameters = set(given_hps.keys()).difference(known_hps)
        if len(unexpected_hyperparameters) > 0:
            raise LeaspyModelInputError(
                f"Only {known_hps} are valid hyperparameters for {cls.__qualname__}. "
                f"Unknown hyperparameters provided: {unexpected_hyperparameters}."
            )

    def _audit_individual_parameters(self, ips: DictParams) -> KwargsType:
        """
        Perform various consistency and compatibility (with current model) checks
        on an individual parameters dict and outputs qualified information about it.

        TODO? move to IndividualParameters class?

        Parameters
        ----------
        ips : dict[param: str, Any]
            Contains some un-trusted individual parameters.
            If representing only one individual (in a multivariate model) it could be:
                * {'tau':0.1, 'xi':-0.3, 'sources':[0.1,...]}

            Or for multiple individuals:
                * {'tau':[0.1,0.2,...], 'xi':[-0.3,0.2,...], 'sources':[[0.1,...],[0,...],...]}

            In particular, a sources vector (if present) should always be a array_like, even if it is 1D

        Returns
        -------
        ips_info : dict
            * ``'nb_inds'`` : int >= 0
                number of individuals present
            * ``'tensorized_ips'`` : dict[param:str, `torch.Tensor`]
                tensorized version of individual parameters
            * ``'tensorized_ips_gen'`` : generator
                generator providing tensorized individual parameters for all individuals present (ordered as is)

        Raises
        ------
        :exc:`.LeaspyIndividualParamsInputError`
            if any of the consistency/compatibility checks fail
        """

        def is_array_like(v):
            # abc.Collection is useless here because set, np.array(scalar) or torch.tensor(scalar)
            # are abc.Collection but are not array_like in numpy/torch sense or have no len()
            try:
                len(v)  # exclude np.array(scalar) or torch.tensor(scalar)
                return hasattr(v, '__getitem__')  # exclude set
            except Exception:
                return False

        # Model supports and needs sources?
        has_sources = (
            hasattr(self, 'source_dimension')
            and isinstance(self.source_dimension, int)
            and self.source_dimension > 0
        )

        # Check parameters names
        expected_parameters = set(['xi', 'tau'] + int(has_sources)*['sources'])
        given_parameters = set(ips.keys())
        symmetric_diff = expected_parameters.symmetric_difference(given_parameters)
        if len(symmetric_diff) > 0:
            raise LeaspyIndividualParamsInputError(
                    f'Individual parameters dict provided {given_parameters} '
                    f'is not compatible for {self.name} model. '
                    f'The expected individual parameters are {expected_parameters}.')

        # Check number of individuals present (with low constraints on shapes)
        ips_is_array_like = {k: is_array_like(v) for k, v in ips.items()}
        ips_size = {k: len(v) if ips_is_array_like[k] else 1 for k, v in ips.items()}

        if has_sources:
            if not ips_is_array_like['sources']:
                raise LeaspyIndividualParamsInputError(
                    f"Sources must be an array_like but {ips['sources']} was provided."
                )

            tau_xi_scalars = all(ips_size[k] == 1 for k in ["tau", "xi"])
            if tau_xi_scalars and (ips_size['sources'] > 1):
                # is 'sources' not a nested array? (allowed iff tau & xi are scalars)
                if not is_array_like(ips['sources'][0]):
                    # then update sources size (1D vector representing only 1 individual)
                    ips_size['sources'] = 1

            # TODO? check source dimension compatibility?

        uniq_sizes = set(ips_size.values())
        if len(uniq_sizes) != 1:
            raise LeaspyIndividualParamsInputError(
                f"Individual parameters sizes are not compatible together. Sizes are {ips_size}."
            )

        # number of individuals present
        n_inds = uniq_sizes.pop()

        # properly choose unsqueezing dimension when tensorizing array_like (useful for sources)
        unsqueeze_dim = -1  # [1,2] => [[1],[2]] (expected for 2 individuals / 1D sources)
        if n_inds == 1:
            unsqueeze_dim = 0  # [1,2] => [[1,2]] (expected for 1 individual / 2D sources)

        # tensorized (2D) version of ips
        t_ips = {k: self._tensorize_2D(v, unsqueeze_dim=unsqueeze_dim) for k, v in ips.items()}

        # construct logs
        return {
            'nb_inds': n_inds,
            'tensorized_ips': t_ips,
            'tensorized_ips_gen': (
                {k: v[i, :].unsqueeze(0) for k, v in t_ips.items()} for i in range(n_inds)
            ),
        }

    @staticmethod
    def _tensorize_2D(x, unsqueeze_dim: int, dtype=torch.float32) -> torch.Tensor:
        """
        Helper to convert a scalar or array_like into an, at least 2D, dtype tensor

        Parameters
        ----------
        x : scalar or array_like
            element to be tensorized
        unsqueeze_dim : 0 or -1
            dimension to be unsqueezed; meaningful for 1D array-like only
            (for scalar or vector of length 1 it has no matter)

        Returns
        -------
        :class:`torch.Tensor`, at least 2D

        Examples
        --------
        >>> _tensorize_2D([1, 2], 0) == tensor([[1, 2]])
        >>> _tensorize_2D([1, 2], -1) == tensor([[1], [2])
        """
        # convert to torch.Tensor if not the case
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=dtype)

        # convert dtype if needed
        if x.dtype != dtype:
            x = x.to(dtype)

        # if tensor is less than 2-dimensional add dimensions
        while x.dim() < 2:
            x = x.unsqueeze(dim=unsqueeze_dim)

        # postcondition: x.dim() >= 2
        return x

    def _get_tensorized_inputs(
        self,
        timepoints,
        individual_parameters,
        *,
        skip_ips_checks: bool = False,
    ) -> Tuple[torch.Tensor, DictParamsTorch]:
        if not skip_ips_checks:
            # Perform checks on ips and gets tensorized version if needed
            ips_info = self._audit_individual_parameters(individual_parameters)
            n_inds = ips_info['nb_inds']
            individual_parameters = ips_info['tensorized_ips']

            if n_inds != 1:
                raise LeaspyModelInputError(
                    f"Only one individual computation may be performed at a time. {n_inds} was provided."
                )

        # Convert the timepoints (list of numbers, or single number) to a 2D torch tensor
        timepoints = self._tensorize_2D(timepoints, unsqueeze_dim=0)  # 1 individual
        return timepoints, individual_parameters

    # TODO: unit tests? (functional tests covered by api.estimate)
    def compute_individual_trajectory(
        self,
        timepoints,
        individual_parameters: DictParams,
        *,
        skip_ips_checks: bool = False,
    ) -> torch.Tensor:
        """
        Compute scores values at the given time-point(s) given a subject's individual parameters.

        Parameters
        ----------
        timepoints : scalar or array_like[scalar] (list, tuple, :class:`numpy.ndarray`)
            Contains the age(s) of the subject.
        individual_parameters : dict
            Contains the individual parameters.
            Each individual parameter should be a scalar or array_like
        skip_ips_checks : bool (default: False)
            Flag to skip consistency/compatibility checks and tensorization
            of individual_parameters when it was done earlier (speed-up)

        Returns
        -------
        :class:`torch.Tensor`
            Contains the subject's scores computed at the given age(s)
            Shape of tensor is (1, n_tpts, n_features)

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            if computation is tried on more than 1 individual
        :exc:`.LeaspyIndividualParamsInputError`
            if invalid individual parameters
        """
        timepoints, individual_parameters = self._get_tensorized_inputs(
            timepoints, individual_parameters, skip_ips_checks=skip_ips_checks
        )
        return self.compute_individual_tensorized(timepoints, individual_parameters)

    # TODO: unit tests? (functional tests covered by api.estimate)
    def compute_individual_ages_from_biomarker_values(
        self,
        value: Union[float, List[float]],
        individual_parameters: DictParams,
        feature: Optional[FeatureType] = None,
    ) -> torch.Tensor:
        """
        For one individual, compute age(s) at which the given features values
        are reached (given the subject's individual parameters).

        Consistency checks are done in the main API layer.

        Parameters
        ----------
        value : scalar or array_like[scalar] (list, tuple, :class:`numpy.ndarray`)
            Contains the biomarker value(s) of the subject.

        individual_parameters : dict
            Contains the individual parameters.
            Each individual parameter should be a scalar or array_like

        feature : str (or None)
            Name of the considered biomarker (optional for univariate models,
            compulsory for multivariate models).

        Returns
        -------
        :class:`torch.Tensor`
            Contains the subject's ages computed at the given values(s)
            Shape of tensor is (1, n_values)

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            if computation is tried on more than 1 individual
        """
        value, individual_parameters = self._get_tensorized_inputs(
            value, individual_parameters, skip_ips_checks=False
        )
        return self.compute_individual_ages_from_biomarker_values_tensorized(
            value, individual_parameters, feature
        )

    @abstractmethod
    def compute_individual_ages_from_biomarker_values_tensorized(
        self,
        value: torch.Tensor,
        individual_parameters: DictParamsTorch,
        feature: Optional[FeatureType],
    ) -> torch.Tensor:
        """
        For one individual, compute age(s) at which the given features values are
        reached (given the subject's individual parameters), with tensorized inputs.

        Parameters
        ----------
        value : torch.Tensor of shape (1, n_values)
            Contains the biomarker value(s) of the subject.

        individual_parameters : dict
            Contains the individual parameters.
            Each individual parameter should be a torch.Tensor

        feature : str (or None)
            Name of the considered biomarker (optional for univariate models, compulsory for multivariate models).

        Returns
        -------
        :class:`torch.Tensor`
            Contains the subject's ages computed at the given values(s)
            Shape of tensor is (n_values, 1)
        """

    @abstractmethod
    def compute_individual_tensorized(
        self,
        timepoints: torch.Tensor,
        individual_parameters: DictParamsTorch,
        *,
        attribute_type=None,
    ) -> torch.Tensor:
        """
        Compute the individual values at timepoints according to the model.

        Parameters
        ----------
        timepoints : :class:`torch.Tensor` of shape (n_individuals, n_timepoints)

        individual_parameters : dict[param_name: str, :class:`torch.Tensor` of shape (n_individuals, n_dims_param)]

        attribute_type : Any (default None)
            Flag to ask for MCMC attributes instead of model's attributes.

        Returns
        -------
        :class:`torch.Tensor` of shape (n_individuals, n_timepoints, n_features)
        """

    @abstractmethod
    def compute_jacobian_tensorized(
        self,
        timepoints: torch.Tensor,
        individual_parameters: DictParamsTorch,
        *,
        attribute_type=None,
    ) -> DictParamsTorch:
        """
        Compute the jacobian of the model w.r.t. each individual parameter.

        This function aims to be used in :class:`.ScipyMinimize` to speed up optimization.

        TODO: as most of numerical operations are repeated when computing model & jacobian,
              we should create a single method that is able to compute model & jacobian "together" (= efficiently)
              when requested with a flag for instance.

        Parameters
        ----------
        timepoints : :class:`torch.Tensor` of shape (n_individuals, n_timepoints)

        individual_parameters : dict[param_name: str, :class:`torch.Tensor` of shape (n_individuals, n_dims_param)]

        attribute_type : Any (default None)
            Flag to ask for MCMC attributes instead of model's attributes.

        Returns
        -------
        dict[param_name: str, :class:`torch.Tensor` of shape (n_individuals, n_timepoints, n_features, n_dims_param)]
        """

    def compute_individual_attachment_tensorized(
        self,
        data: Dataset,
        param_ind: DictParamsTorch,
        *,
        attribute_type=None,
    ) -> torch.Tensor:
        """
        Compute attachment term (per subject)

        Parameters
        ----------
        data : :class:`.Dataset`
            Contains the data of the subjects, in particular the subjects'
            time-points and the mask for nan values & padded visits

        param_ind : dict
            Contain the individual parameters

        attribute_type : str or None
            Flag to ask for MCMC attributes instead of model's attributes.

        Returns
        -------
        attachment : :class:`torch.Tensor`
            Negative Log-likelihood, shape = (n_subjects,)
        """
        predictions = self.compute_individual_tensorized(
            data.timepoints, param_ind, attribute_type=attribute_type,
        )
        nll = self.noise_model.compute_nll(data, predictions)
        return nll.sum(dim=tuple(range(1, nll.ndim)))

    def compute_canonical_loss_tensorized(
        self,
        data: Dataset,
        param_ind: DictParamsTorch,
        *,
        attribute_type=None,
    ) -> torch.Tensor:
        """
        Compute canonical loss, which depends on noise-model.

        Parameters
        ----------
        data : :class:`.Dataset`
            Contains the data of the subjects, in particular the subjects'
            time-points and the mask for nan values & padded visits

        param_ind : dict
            Contain the individual parameters

        attribute_type : str or None (default)
            Flag to ask for MCMC attributes instead of model's attributes.

        Returns
        -------
        loss : :class:`torch.Tensor`
            shape = * (depending on noise-model, always summed over individuals & visits)
        """
        predictions = self.compute_individual_tensorized(
            data.timepoints, param_ind, attribute_type=attribute_type,
        )
        return self.noise_model.compute_canonical_loss(data, predictions)

    def compute_sufficient_statistics(
        self,
        data: Dataset,
        realizations: CollectionRealization,
    ) -> DictParamsTorch:
        """
        Compute sufficient statistics from realizations.

        Parameters
        ----------
        data : :class:`.Dataset`
        realizations : :class:`.CollectionRealization`

        Returns
        -------
        dict[suff_stat: str, :class:`torch.Tensor`]
        """
        suff_stats = self.compute_model_sufficient_statistics(data, realizations)
        individual_parameters = self.get_param_from_real(realizations)
        predictions = self.compute_individual_tensorized(
            data.timepoints, individual_parameters, attribute_type='MCMC'
        )
        noise_suff_stats = self.noise_model.compute_sufficient_statistics(
            data, predictions
        )

        # we add some fake sufficient statistics that are in fact convergence metrics (summed over individuals)
        # TODO proper handling of metrics
        d_regul, _ = self.compute_regularity_individual_parameters(individual_parameters, include_constant=True)
        cvg_metrics = {f'nll_regul_{param}': r.sum() for param, r in d_regul.items()}
        cvg_metrics['nll_regul_tot'] = sum(cvg_metrics.values(), 0.)
        cvg_metrics['nll_attach'] = self.noise_model.compute_nll(data, predictions).sum()
        cvg_metrics['nll_tot'] = cvg_metrics['nll_attach'] + cvg_metrics['nll_regul_tot']

        # using `dict` enforces no conflicting name in sufficient statistics
        return dict(suff_stats, **noise_suff_stats, **cvg_metrics)

    @abstractmethod
    def compute_model_sufficient_statistics(
        self,
        data: Dataset,
        realizations: CollectionRealization,
    ) -> DictParamsTorch:
        """
        Compute sufficient statistics from realizations

        Parameters
        ----------
        data : :class:`.Dataset`
        realizations : :class:`.CollectionRealization`

        Returns
        -------
        dict[suff_stat: str, :class:`torch.Tensor`]
        """

    def update_parameters_burn_in(
        self,
        data: Dataset,
        sufficient_statistics: DictParamsTorch,
    ) -> None:
        """
        Update model parameters (burn-in phase).

        Parameters
        ----------
        data : :class:`.Dataset`
        sufficient_statistics : dict[suff_stat: str, :class:`torch.Tensor`]
        """
        self.update_model_parameters_burn_in(data, sufficient_statistics)
        self.noise_model.update_parameters_from_sufficient_statistics(data, sufficient_statistics)

    @abstractmethod
    def update_model_parameters_burn_in(
        self,
        data: Dataset,
        sufficient_statistics: DictParamsTorch,
    ) -> None:
        """
        Update model parameters (burn-in phase).

        Parameters
        ----------
        data : :class:`.Dataset`
        sufficient_statistics : dict[suff_stat: str, :class:`torch.Tensor`]
        """

    def update_parameters_normal(
        self,
        data: Dataset,
        sufficient_statistics: DictParamsTorch,
    ) -> None:
        """
        Update model parameters (after burn-in phase).

        Parameters
        ----------
        data : :class:`.Dataset`
        sufficient_statistics : dict[suff_stat: str, :class:`torch.Tensor`]
        """
        self.update_model_parameters_normal(data, sufficient_statistics)
        self.noise_model.update_parameters_from_sufficient_statistics(data, sufficient_statistics)

    @abstractmethod
    def update_model_parameters_normal(
        self,
        data: Dataset,
        sufficient_statistics: DictParamsTorch,
    ) -> None:
        """
        Update model parameters (after burn-in phase).

        Parameters
        ----------
        data : :class:`.Dataset`
        sufficient_statistics : dict[suff_stat: str, :class:`torch.Tensor`]
        """

    def get_population_realization_names(self) -> List[str]:
        """
        Get names of population variables of the model.

        Returns
        -------
        list[str]
        """
        return [
            name for name, value in self.random_variable_informations().items()
            if value['type'] == 'population'
        ]

    def get_individual_realization_names(self) -> List[str]:
        """
        Get names of individual variables of the model.

        Returns
        -------
        list[str]
        """
        return [
            name for name, value in self.random_variable_informations().items()
            if value['type'] == 'individual'
        ]

    @classmethod
    def _serialize_tensor(cls, v, *, indent: str = "", sub_indent: str = "") -> str:
        """Nice serialization of floats, torch tensors (or numpy arrays)."""
        if isinstance(v, (str, bool, int)):
            return str(v)
        if isinstance(v, float) or getattr(v, 'ndim', -1) == 0:
            # for 0D tensors / arrays the default behavior is to print all digits...
            # change this!
            return f'{v:.{1+torch_print_opts.precision}g}'
        if isinstance(v, (list, frozenset, set, tuple)):
            try:
                return cls._serialize_tensor(torch.tensor(list(v)), indent=indent, sub_indent=sub_indent)
            except Exception:
                return str(v)
        if isinstance(v, dict):
            if not len(v):
                return ""
            subs = [
                f"{p} : " + cls._serialize_tensor(vp, indent="  ", sub_indent=" "*len(f"{p} : ["))
                for p, vp in v.items()
            ]
            lines = [indent + _ for _ in "\n".join(subs).split("\n")]
            return "\n" + "\n".join(lines)
        # torch.tensor, np.array, ...
        # in particular you may use `torch.set_printoptions` and `np.set_printoptions` globally
        # to tune the number of decimals when printing tensors / arrays
        v_repr = str(v)
        # remove tensor prefix & possible device/size/dtype suffixes
        v_repr = re.sub(r'^[^\(]+\(', '', v_repr)
        v_repr = re.sub(r'(?:, device=.+)?(?:, size=.+)?(?:, dtype=.+)?\)$', '', v_repr)
        # adjust justification
        return re.sub(r'\n[ ]+([^ ])', rf'\n{sub_indent}\1', v_repr)

    def __str__(self):
        output = "=== MODEL ==="
        output += self._serialize_tensor(self.parameters)

        nm_props = noise_model_export(self.noise_model)
        nm_name = nm_props.pop('name')
        output += f"\nnoise-model : {nm_name}"
        output += self._serialize_tensor(nm_props, indent="  ")

        return output

    def compute_regularity_realization(self, realization: Realization) -> torch.Tensor:
        """
        Compute regularity term for a :class:`.Realization` instance.

        Parameters
        ----------
        realization : :class:`.Realization`

        Returns
        -------
        :class:`torch.Tensor` of the same shape as `realization.tensor_realizations`
        """
        if realization.variable_type == 'population':
            # Regularization of population variables around current model values
            mean = self.parameters[realization.name]
            std = self.MCMC_toolbox['priors'][f"{realization.name}_std"]
        elif realization.variable_type == 'individual':
            # Regularization of individual parameters around mean / std from model parameters
            mean = self.parameters[f"{realization.name}_mean"]
            std = self.parameters[f"{realization.name}_std"]
        else:
            raise LeaspyModelInputError(
                f"Variable type '{realization.variable_type}' not known, "
                "should be 'population' or 'individual'."
            )
        # we do not need to include regularity constant (priors are always fixed at a given iteration)
        return self.compute_regularity_variable(
            realization.tensor_realizations, mean, std, include_constant=False
        )

    def compute_regularity_individual_parameters(
        self,
        individual_parameters: DictParamsTorch,
        *,
        include_constant: bool = False,
    ) -> Tuple[DictParamsTorch, DictParamsTorch]:
        """
        Compute the regularity terms (and their gradients if requested), per individual variable of the model.

        Parameters
        ----------
        individual_parameters : dict[str, :class:`torch.Tensor` [n_ind, n_dims_param]]
            Individual parameters as a dict of tensors.
        include_constant : bool, optional
            Whether to include a constant term or not.
            Default=False.

        Returns
        -------
        regularity : dict[param_name: str, :class:`torch.Tensor` [n_individuals]]
            Regularity of the patient(s) corresponding to the given individual parameters.

        regularity_grads : dict[param_name: str, :class:`torch.Tensor` [n_individuals, n_dims_param]]
            Gradient of regularity term with respect to individual parameters.
        """
        regularity = {}
        regularity_grads = {}

        for param_name, param_val in individual_parameters.items():
            # priors on this parameter
            priors = dict(
                mean=self.parameters[param_name+"_mean"],
                std=self.parameters[param_name+"_std"]
            )

            # TODO? create a more generic method in model `compute_regularity_variable`?
            # (at least pass the parameter name to this method to compute regularity term for non-Normal priors)
            regularity_param, regularity_grads[param_name] = self.compute_regularity_variable(
                param_val, **priors, include_constant=include_constant, with_gradient=True
            )
            # we sum on the dimension of the parameter (always 1D for now), but regularity term is per individual
            # TODO: shouldn't this summation be done directly in `compute_regularity_variable`
            #  (e.g. multivariate normal)
            regularity[param_name] = regularity_param.sum(dim=1)

        return regularity, regularity_grads

    def compute_regularity_variable(
        self,
        value: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        include_constant: bool = True,
        with_gradient: bool = False,
    ) -> torch.Tensor:
        """
        Compute regularity term (Gaussian distribution) and optionally its gradient wrt value.

        TODO: should be encapsulated in a RandomVariableSpecification class together with other specs of RV.

        Parameters
        ----------
        value, mean, std : :class:`torch.Tensor` of same shapes
        include_constant : bool (default True)
            Whether we include or not additional terms constant with respect to `value`.
        with_gradient : bool (default False)
            Whether we also return the gradient of regularity term with respect to `value`.

        Returns
        -------
        :class:`torch.Tensor` of same shape than input
        """
        # This is really slow when repeated on tiny tensors (~3x slower than direct formula!)
        #return -self.regularization_distribution_factory(mean, std).log_prob(value)

        y = (value - mean) / std
        neg_loglike = 0.5 * y * y
        if include_constant:
            neg_loglike += 0.5 * torch.log(TWO_PI * std**2)
        if not with_gradient:
            return neg_loglike
        nll_grad = y / std
        return neg_loglike, nll_grad

    def initialize_realizations_for_model(self, n_individuals: int, **init_kws) -> CollectionRealization:
        """
        Initialize a :class:`.CollectionRealization` used during model fitting or mode/mean realization personalization.

        Parameters
        ----------
        n_individuals : int
            Number of individuals to track
        **init_kws
            Keyword arguments passed to :meth:`.CollectionRealization.initialize`.
            (In particular `individual_variable_init_at_mean` to "initialize at mean"
            or `skip_variable` to filter some variables).

        Returns
        -------
        :class:`.CollectionRealization`
        """
        realizations = CollectionRealization()
        realizations.initialize(n_individuals, self, **init_kws)
        return realizations

    @abstractmethod
    def random_variable_informations(self) -> DictParams:
        """
        Information on model's random variables.

        Returns
        -------
        dict[str, Any]
            * name: str
                Name of the random variable
            * type: 'population' or 'individual'
                Individual or population random variable?
            * shape: tuple[int, ...]
                Shape of the variable (only 1D for individual and 1D or 2D for pop. are supported)
            * rv_type: str
                An indication (not used in code) on the probability distribution used for the var
                (only Gaussian is supported)
            * scale: optional float
                The fixed scale to use for initial std-dev in the corresponding sampler.
                When not defined, sampler will rely on scales estimated at model initialization.
                cf. :class:`~leaspy.algo.utils.samplers.GibbsSampler`
        """

    def smart_initialization_realizations(
        self,
        dataset: Dataset,
        realizations: CollectionRealization,
    ) -> CollectionRealization:
        """
        Smart initialization of realizations if needed (input may be modified in-place).

        Default behavior to return `realizations` as they are (no smart trick).

        Parameters
        ----------
        dataset : :class:`.Dataset`
        realizations : :class:`.CollectionRealization`

        Returns
        -------
        :class:`.CollectionRealization`
        """
        return realizations

    def _create_dictionary_of_population_realizations(self) -> dict:
        pop_dictionary: Dict[str, Realization] = {}
        for name_var, info_var in self.random_variable_informations().items():
            if info_var['type'] != "population":
                continue
            real = Realization.from_tensor(
                name_var, info_var['shape'], info_var['type'], self.parameters[name_var]
            )
            pop_dictionary[name_var] = real

        return pop_dictionary

    @staticmethod
    def time_reparametrization(
        timepoints: torch.Tensor,
        xi: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tensorized time reparametrization formula

        <!> Shapes of tensors must be compatible between them.

        Parameters
        ----------
        timepoints : :class:`torch.Tensor`
            Timepoints to reparametrize
        xi : :class:`torch.Tensor`
            Log-acceleration of individual(s)
        tau : :class:`torch.Tensor`
            Time-shift(s)

        Returns
        -------
        :class:`torch.Tensor` of same shape as `timepoints`
        """
        return torch.exp(xi) * (timepoints - tau)

    def get_param_from_real(self, realizations: CollectionRealization) -> DictParamsTorch:
        """
        Get individual parameters realizations from all model realizations

        <!> The tensors are not cloned and so a link continue to exist between the individual parameters
            and the underlying tensors of realizations.

        Parameters
        ----------
        realizations : :class:`.CollectionRealization`

        Returns
        -------
        dict[param_name: str, :class:`torch.Tensor` [n_individuals, dims_param]]
            Individual parameters
        """
        return {
            variable_ind: realizations[variable_ind].tensor_realizations
            for variable_ind in self.get_individual_realization_names()
        }

    def move_to_device(self, device: torch.device) -> None:
        """
        Move a model and its relevant attributes to the specified device.

        Parameters
        ----------
        device : torch.device
        """

        # Note that in a model, the only tensors that need offloading to a
        # particular device are in the model.parameters dict as well as in the
        # attributes and MCMC_toolbox['attributes'] objects

        for parameter in self.parameters:
            self.parameters[parameter] = self.parameters[parameter].to(device)

        if hasattr(self, "attributes"):
            self.attributes.move_to_device(device)

        if hasattr(self, "MCMC_toolbox"):
            MCMC_toolbox_attributes = self.MCMC_toolbox.get("attributes", None)
            if MCMC_toolbox_attributes is not None:
                MCMC_toolbox_attributes.move_to_device(device)

        self.noise_model.move_to_device(device)
