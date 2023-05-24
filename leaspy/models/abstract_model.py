from __future__ import annotations

import re
from abc import abstractmethod
import json
from inspect import signature
import warnings

import torch
from torch._tensor_str import PRINT_OPTS as torch_print_opts

from leaspy import __version__
from leaspy.models.base import BaseModel
from leaspy.models.obs_models import ObservationModel
from leaspy.models.utilities import tensor_to_list
from leaspy.io.data.dataset import Dataset

from leaspy.variables.specs import (
    VarName,
    VarValue,
    Hyperparameter,
    ModelParameter,
    LinkedVariable,
    DataVariable,
    LatentVariableInitType,
    NamedVariables,
    SuffStatsRO,
    SuffStatsRW,
    LVL_FT,
)
from leaspy.variables.dag import VariablesDAG
from leaspy.variables.state import State, StateForkType
from leaspy.utils.weighted_tensor import WeightedTensor

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


#  TODO? refact so to only contain methods needed for the Leaspy api + add another
#  abstract class (interface) on top of it for MCMC fittable models + one for "manifold models"

# TODO: not 100% clear to me whether:
# 1. model should have an internal state? or only provide methods to define suited states (i.e. with the right DAG) and interact with such states
# 2. model methods should have a `state: State` argument, or the state used is always the model internal one?


class AbstractModel(BaseModel):
    """
    Contains the common attributes & methods of the different probabilistic models.

    Parameters
    ----------
    name : str
        The name of the model
    obs_models : ObservationModel or Iterable[ObservationModel]
        The noise model for observations (keyword-only parameter).
    fit_metrics : dict
        Metrics that should be measured during the fit of the model
        and reported back to the user.
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
        Contains the model's parameters (read-only)
    obs_models : Tuple[ObservationModel, ...]
        The observation model(s) associated to the model.
    fit_metrics : dict
        Contains the metrics that are measured during the fit of the model and reported to the user.
    _state : State
        Private instance holding all values for model variables and their derived variables.
    """

    def __init__(
        self,
        name: str,
        *,
        # TODO? if we'd allow to pass a state there should be a all bunch of checks I guess? only "equality" of DAG is OK?
        # (WIP: cf. comment regarding inclusion of state here)
        # state: Optional[State] = None,
        # TODO? Factory of `ObservationModel` instead? (typically one would need the dimension to instantiate the `noise_std` variable of the right shape...)
        obs_models: Union[ObservationModel, Iterable[ObservationModel]],
        fit_metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)

        # observation models: one or multiple (WIP - e.g. for joint model)
        if isinstance(obs_models, ObservationModel):
            obs_models = (obs_models,)
        self.obs_models = tuple(obs_models)

        # Internal state to hold all model & data variables
        # WIP: cf. comment regarding inclusion of state here
        self._state: Optional[State] = None  # = state

        # load hyperparameters
        # <!> some may still be missing at this point (e.g. `dimension`, `source_dimension`, ...)
        # (thus we sh/could NOT instantiate the DAG right now!)
        self.load_hyperparameters(kwargs)

        # TODO: dirty hack for now, cf. AbstractFitAlgo
        self.fit_metrics = fit_metrics

    # @property
    # def noise_model(self) -> BaseNoiseModel:
    #     if self._noise_model is None:
    #         raise LeaspyModelInputError("The `noise_model` was not properly initialized.")
    #     return self._noise_model
    #
    # @noise_model.setter
    # def noise_model(self, model: NoiseModelFactoryInput):
    #     noise_model = noise_model_factory(model)
    #     self.check_noise_model_compatibility(noise_model)
    #     self._noise_model = noise_model
    #
    #def check_noise_model_compatibility(self, model: BaseNoiseModel) -> None:
    #    """
    #    Raise a LeaspyModelInputError is the provided noise model isn't compatible with the model instance.
    #    This needs to be implemented in subclasses.
    #
    #    Parameters
    #    ----------
    #    model : BaseNoiseModel
    #        The noise model with which to check compatibility.
    #    """
    #    if not isinstance(model, BaseNoiseModel):
    #        raise LeaspyModelInputError(
    #            "Expected a subclass of BaselNoiseModel, but received "
    #            f"a {type(model).__name__} instead."
    #        )

    @property
    def dag(self) -> VariablesDAG:
        assert self._state is not None, "Model state is not initialized yet."
        return self._state.dag

    @property
    def hyperparameters_names(self) -> Tuple[VarName, ...]:
        return tuple(self.dag.sorted_variables_by_type[Hyperparameter])

    @property
    def parameters_names(self) -> Tuple[VarName, ...]:
        return tuple(self.dag.sorted_variables_by_type[ModelParameter])

    # Useless as of now...
    # @property
    # def population_variables_names(self) -> Tuple[VarName, ...]:
    #     return tuple(self.dag.sorted_variables_by_type[PopulationLatentVariable])
    #
    # @property
    # def individual_variables_names(self) -> Tuple[VarName, ...]:
    #     return tuple(self.dag.sorted_variables_by_type[IndividualLatentVariable])

    @property
    def parameters(self) -> DictParamsTorch:
        """Dictionary of values for model parameters."""
        return {
            p: self._state[p]
            # TODO: a separated method for hyperparameters?
            # include hyperparameters as well for now to micmic old behavior
            for p in self.hyperparameters_names + self.parameters_names
        }

    @abstractmethod
    def to_dict(self) -> KwargsType:
        """
        Export model as a dictionary ready for export.

        Returns
        -------
        KwargsType :
            The model instance serialized as a dictionary.
        """
        return {
            'leaspy_version': __version__,
            'name': self.name,
            'features': self.features,
            'dimension': self.dimension,
            'obs_models': {
                obs_model.name: obs_model.serialized()
                for obs_model in self.obs_models
            },
            # 'obs_models': export_obs_models(self.obs_models),
            'parameters': {
                k: tensor_to_list(v)
                for k, v in (self.parameters or {}).items()
            },
            'fit_metrics': self.fit_metrics,  # TODO improve
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
        export_kws = {k: kwargs.pop(k) for k in signature(self.to_dict).parameters if k in kwargs}
        model_settings = self.to_dict(**export_kws)

        # Default json.dump kwargs:
        kwargs = {'indent': 2, **kwargs}

        with open(path, 'w') as fp:
            json.dump(model_settings, fp, **kwargs)

    def load_parameters(self, parameters: KwargsType) -> None:
        """
        Instantiate or update the model's parameters.

        It assumes that all model hyperparameters are defined.

        Parameters
        ----------
        parameters : dict[str, Any]
            Contains the model's parameters
        """
        if self._state is None:
            self.initialize_state()

        # TODO: a bit dirty due to hyperparams / params mix (cf. `.parameters` property note)

        params_names = self.parameters_names
        missing_params = set(params_names).difference(parameters)
        if len(missing_params):
            warnings.warn(f"Missing some model parameters: {missing_params}")
        extra_vars = set(parameters).difference(self.dag)
        if len(extra_vars):
            raise LeaspyModelInputError(f"Unknown model variables: {extra_vars}")
        # TODO: check no DataVariable provided???
        #extra_params = set(parameters).difference(cur_params)
        #if len(extra_params):
        #    # e.g. mixing matrix, which is a derived variable - checking their values only
        #    warnings.warn(f"Ignoring some provided values that are not model parameters: {extra_params}")

        def val_to_tensor(val, shape: Optional[tuple] = None):
            if not isinstance(val, (torch.Tensor, WeightedTensor)):
                val = torch.tensor(val)
            if shape is not None:
                val = val.view(shape)  # no expansion here
            return val

        # update parameters first (to be able to check values of derived variables afterwards)
        provided_params = {
            p: val_to_tensor(parameters[p], self.dag[p].shape)
            for p in params_names if p in parameters
        }
        for p, val in provided_params.items():
            # TODO: WeightedTensor? (e.g. batched `deltas`)
            self._state[p] = val

        # derive the population latent variables from model parameters
        # e.g. to check value of `mixing_matrix` we need `v0` and `betas` (not just `log_v0` and `betas_mean`)
        self._state.initialize_population_latent_variables(LatentVariableInitType.PRIOR_MODE)

        # check equality of other values (hyperparameters or linked variables)
        for p, val in parameters.items():
            if p in provided_params:
                continue
            # TODO: a bit dirty due to hyperparams / params mix (cf. `.parameters` property note)
            try:
                cur_val = self._state[p]
            except Exception as e:
                raise LeaspyModelInputError(
                    f"Impossible to compare value of provided value for {p} "
                    "- not computable given current state"
                ) from e
            val = val_to_tensor(val, getattr(self.dag[p], "shape", None))
            assert val.shape == cur_val.shape, (p, val.shape, cur_val.shape)
            # TODO: WeightedTensor? (e.g. batched `deltas``)
            assert torch.allclose(val, cur_val), (p, val, cur_val)

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
        timepoints: torch.Tensor,
        individual_parameters: DictParamsTorch,
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

        Nota: model uses its current internal state.

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
        raise NotImplementedError("WIP")


        timepoints, individual_parameters = self._get_tensorized_inputs(
            timepoints, individual_parameters, skip_ips_checks=skip_ips_checks
        )

        # TODO? ability to fork after several assignments?
        # or clone the state for this op?
        # otherwise if called during a fit [typically to produce plots to monitor convergence]
        # we just manually re-assign modified vars to their initial state
        # (indeed `t` is fixed during the fit, and the evolving ind. latent vars were likely not the same as those provided!)
        with self._state.auto_fork(None) as XXX:
            self._state["t"] = timepoints
            for ip, ip_v in individual_parameters.items():
                self._state[ip] = ip_v
        val = self._state["model"]
        self._state.revert(XXX) # or a similar syntax? (to tell that we want to go back to ... point)

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
        raise NotImplementedError("TODO")
        value, individual_parameters = self._get_tensorized_inputs(
            value, individual_parameters, skip_ips_checks=False
        )
        return self.compute_individual_ages_from_biomarker_values_tensorized(
            value, individual_parameters, feature
        )

    #@abstractmethod
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
        value : :class:`torch.Tensor` of shape (1, n_values)
            Contains the biomarker value(s) of the subject.

        individual_parameters : DictParamsTorch
            Contains the individual parameters.
            Each individual parameter should be a torch.Tensor

        feature : str (or None)
            Name of the considered biomarker (optional for univariate models,
            compulsory for multivariate models).

        Returns
        -------
        :class:`torch.Tensor`
            Contains the subject's ages computed at the given values(s)
            Shape of tensor is (n_values, 1)
        """
        raise NotImplementedError("TODO in child classes")

    def compute_jacobian_tensorized(
        self,
        state: State,
    ) -> DictParamsTorch:
        """
        Compute the jacobian of the model w.r.t. each individual parameter, given the input state.

        This function aims to be used in :class:`.ScipyMinimize` to speed up optimization.

        Parameters
        ----------
        state : :class:`.State`
            Instance holding values for all model variables (including latent individual variables), as well as:
            - timepoints : :class:`torch.Tensor` of shape (n_individuals, n_timepoints)

        Returns
        -------
        dict[param_name: str, :class:`torch.Tensor` of shape (n_individuals, n_timepoints, n_features, n_dims_param)]
        """
        raise NotImplementedError("TODO")
        return {
            ip: state[f"model_jacobian_{ip}"]
            for ip in self.get_individual_variable_names()
        }

    @classmethod
    def compute_sufficient_statistics(
        cls,
        state: State,
    ) -> SuffStatsRW:
        """
        Compute sufficient statistics from state.

        Parameters
        ----------
        state : :class:`.State`

        Returns
        -------
        dict[suff_stat: str, :class:`torch.Tensor`]
        """
        suff_stats = {}
        for mp_var in state.dag.sorted_variables_by_type[ModelParameter].values():
            mp_var: ModelParameter  # type-hint only
            suff_stats.update(mp_var.suff_stats(state))

        # we add some fake sufficient statistics that are in fact convergence metrics (summed over individuals)
        # TODO proper handling of metrics
        # We do not account for regularization of pop. vars since we do NOT have true Bayesian priors on them (for now)
        for k in ("nll_attach", "nll_regul_ind_sum"):
            suff_stats[k] = state[k]
        suff_stats["nll_tot"] = suff_stats["nll_attach"] + suff_stats["nll_regul_ind_sum"]  # "nll_regul_all_sum"

        return suff_stats

    @classmethod
    def update_parameters(
        cls,
        state: State,
        sufficient_statistics: SuffStatsRO,
        *,
        burn_in: bool,
    ) -> None:
        """
        Update model parameters of the provided state.

        Parameters
        ----------
        state : :class:`.State`
        sufficient_statistics : dict[suff_stat: str, :class:`torch.Tensor`]
        burn_in : bool
        """
        # <!> we should wait before updating state since some updating rules may depending on OLD state
        # (i.e. no sequential update of state but batched updates once all updated values were retrieved)
        # (+ it would be inefficient since we could recompute some derived values between updates!)
        params_updates = {}
        for mp_name, mp_var in state.dag.sorted_variables_by_type[ModelParameter].items():
            mp_var: ModelParameter  # type-hint only
            params_updates[mp_name] = mp_var.compute_update(
                state=state, suff_stats=sufficient_statistics, burn_in=burn_in
            )
        # mass update at end
        for mp, mp_updated_val in params_updates.items():
            state[mp] = mp_updated_val

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

        # TODO/WIP obs models...
        # nm_props = export_noise_model(self.noise_model)
        # nm_name = nm_props.pop('name')
        # output += f"\nnoise-model : {nm_name}"
        # output += self._serialize_tensor(nm_props, indent="  ")

        return output

    @staticmethod
    def time_reparametrization(
        *,
        t: torch.Tensor,  # TODO: TensorOrWeightedTensor?
        alpha: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tensorized time reparametrization formula

        <!> Shapes of tensors must be compatible between them.

        Parameters
        ----------
        t : :class:`torch.Tensor`
            Timepoints to reparametrize
        alpha : :class:`torch.Tensor`
            Acceleration factors of individual(s)
        tau : :class:`torch.Tensor`
            Time-shift(s)

        Returns
        -------
        :class:`torch.Tensor` of same shape as `timepoints`
        """
        return alpha * (t - tau)

    #@abstractmethod
    #def model(self, **kws) -> torch.Tensor:
    #    pass
    #
    #@abstractmethod
    #def model_jacobian(self, **kws) -> torch.Tensor:
    #    pass

    def get_variables_specs(self) -> NamedVariables:
        """Return the specifications of the variables (latent variables, derived variables, model 'parameters') that are part of the model."""
        d = NamedVariables({
            "t": DataVariable(),
            "rt": LinkedVariable(self.time_reparametrization),
            #"model": LinkedVariable(self.model),  # function arguments may depends on hyperparameters so postpone (e.g. presence of sources or not)
            #"model_jacobian_{ip}": LinkedVariable(self.model_jacobian), for ip in IndividualLatentVariables....
        })

        single_obs_model = len(self.obs_models) == 1
        for obs_model in self.obs_models:
            d.update(obs_model.get_variables_specs(named_attach_vars=not single_obs_model))

        if not single_obs_model:
            assert False, "WIP: Only 1 noise model supported for now, but to be extended"
            d.update(
                #nll_attach_full=LinkedVariable(Sum(...)),
                nll_attach_ind=LinkedVariable(Sum(...)),
                nll_attach=LinkedVariable(Sum(...)),
                # TODO Same for nll_attach_ind jacobian, w.r.t each observation var???
            )

        return d

    def initialize_state(self) -> None:
        """
        Initialize the internal state of model, as well as the underlying DAG.

        Note that all model hyperparameters (dimension, source_dimension, ...) should be defined
        in order to be able to do so.
        """
        self._state = State(
            VariablesDAG.from_dict(self.get_variables_specs()),
            auto_fork_type=StateForkType.REF
        )

    def initialize(self, dataset: Dataset, method: str = 'default') -> None:

        super().initialize(dataset, method=method)

        if self._state is not None:
            raise LeaspyModelInputError("Trying to initialize model again")
        self.initialize_state()

        # WIP: design of this may be better somehow?
        with self._state.auto_fork(None):

            # Set data variables
            # TODO/WIP: we use a regular tensor with 0 for times so that 'model' is a regular tensor
            # (to avoid having to cope with `StatelessDistributionFamily` having some `WeightedTensor` as parameters)
            # (but we might need it at some point, especially for `batched_deltas` of ordinal model for instance)
            self._state["t"] = dataset.timepoints.masked_fill((~dataset.mask.to(torch.bool)).all(dim=LVL_FT), 0.)
            for obs_model in self.obs_models:
                self._state[obs_model.name] = obs_model.getter(dataset)

            # Set model parameters
            self.initialize_model_parameters(dataset, method=method)

            # Initialize population latent variables to their mode
            self._state.initialize_population_latent_variables(LatentVariableInitType.PRIOR_MODE)

    def initialize_model_parameters(self, dataset: Dataset, method: str):
        """Initialize model parameters (in-place, in `_state`)."""
        d = self.get_initial_model_parameters(dataset, method=method)
        model_params = self.dag.sorted_variables_by_type[ModelParameter]
        assert d.keys() == set(model_params), (d.keys(), set(model_params))
        for mp, var in model_params.items():
            val = d[mp]
            if not isinstance(val, (torch.Tensor, WeightedTensor)):
                val = torch.tensor(val, dtype=torch.float)
            self._state[mp] = val.expand(var.shape)

    @abstractmethod
    def get_initial_model_parameters(self, dataset: Dataset, method: str) -> Dict[VarName, VarValue]:
        """Get initial values for model parameters."""

    def move_to_device(self, device: torch.device) -> None:
        """
        Move a model and its relevant attributes to the specified device (in-place).

        Parameters
        ----------
        device : torch.device
        """
        if self._state is None:
            return

        self._state.to_device(device)
        for hp in self.hyperparameters_names:
            self._state.dag[hp].to_device(device)
