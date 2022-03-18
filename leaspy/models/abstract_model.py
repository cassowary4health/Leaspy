from __future__ import annotations
from typing import TYPE_CHECKING

import re
import math
from abc import ABC, abstractmethod
import copy

import torch
from torch._tensor_str import PRINT_OPTS as torch_print_opts

from leaspy.io.realizations.collection_realization import CollectionRealization
from leaspy.io.realizations.realization import Realization
from leaspy.models.utils.noise_model import NoiseModel

from leaspy.exceptions import LeaspyConvergenceError, LeaspyIndividualParamsInputError, LeaspyModelInputError
from leaspy.utils.typing import FeatureType, KwargsType, DictParams, DictParamsTorch, Union, List, Dict, Tuple, Iterable, Optional

if TYPE_CHECKING:
    from leaspy.io.data.dataset import Dataset

TWO_PI = 2 * math.pi


# TODO? refact so to only contain methods needed for the Leaspy api + add another abstract class (interface) on top of it for MCMC fittable models + one for "manifold models"
class AbstractModel(ABC):
    """
    Contains the common attributes & methods of the different models.

    Parameters
    ----------
    name : str
        The name of the model
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
    noise_model : str
        The noise structure for the model.
        cf.  :class:`.NoiseModel` to see possible values.
    regularization_distribution_factory : function dist params -> :class:`torch.distributions.Distribution`
        Factory of torch distribution to compute log-likelihoods for regularization (gaussian by default)
    """

    def __init__(self, name: str, **kwargs):
        self.is_initialized: bool = False
        self.name = name
        self.features: List[FeatureType] = None
        self.dimension: int = None  # TODO: to be converted into a read-only property (cf. in GenericModel)
        self.parameters: KwargsType = None
        self.noise_model: str = None

        # TODO? shouldn't it belong to each random variable specs?
        self.regularization_distribution_factory = torch.distributions.normal.Normal

        # load hyperparameters
        # <!> in children classes with new hyperparameter you should do it manually at end of __init__ to overwrite default values
        self.load_hyperparameters(kwargs)

    @abstractmethod
    def initialize(self, dataset: Dataset, method: str = 'default') -> None:
        """
        Initialize the model given a dataset and an initialization method.

        After calling this method :attr:`is_initialized` should be True and model should be ready for use.

        Parameters
        ----------
        dataset : :class:`.Dataset`
            The dataset we want to initialize from.
        method : str
            A custom method to initialize the model
        """

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
        Load model's hyperparameters

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
        """Helper function raising a :exc:`.LeaspyModelInputError` if any unknown hyperparameter provided for model."""
        # TODO: replace with better logic from GenericModel in the future
        unexpected_hyperparameters = set(given_hps.keys()).difference(known_hps)
        if len(unexpected_hyperparameters) > 0:
            raise LeaspyModelInputError(
                    f"Only {known_hps} are valid hyperparameters for {cls.__qualname__}. "
                    f"Unknown hyperparameters provided: {unexpected_hyperparameters}.")

    @abstractmethod
    def save(self, path: str, **kwargs) -> None:
        """
        Save Leaspy object as json model parameter file.

        Parameters
        ----------
        path : str
            Path to store the model's parameters.
        **kwargs
            Keyword arguments for json.dump method.
        """

    def compute_sum_squared_per_ft_tensorized(self, dataset: Dataset, param_ind: DictParamsTorch, *,
                                              attribute_type=None) -> torch.FloatTensor:
        """
        Compute the square of the residuals per subject per feature

        Parameters
        ----------
        dataset : :class:`.Dataset`
            Contains the data of the subjects, in particular the subjects' time-points and the mask (?)
        param_ind : dict
            Contain the individual parameters
        attribute_type : Any (default None)
            Flag to ask for MCMC attributes instead of model's attributes.

        Returns
        -------
        :class:`torch.Tensor` of shape (n_individuals,dimension)
            Contains L2 residual for each subject and each feature
        """
        res = self.compute_individual_tensorized(dataset.timepoints, param_ind, attribute_type=attribute_type)
        r1 = dataset.mask.float() * (res - dataset.values) # ijk tensor (i=individuals, j=visits, k=features)
        return (r1 * r1).sum(dim=1)  # sum on visits

    def compute_sum_squared_tensorized(self, dataset: Dataset, param_ind: DictParamsTorch, *,
                                       attribute_type=None) -> torch.FloatTensor:
        """
        Compute the square of the residuals per subject

        Parameters
        ----------
        dataset : :class:`.Dataset`
            Contains the data of the subjects, in particular the subjects' time-points and the mask (?)
        param_ind : dict
            Contain the individual parameters
        attribute_type : Any (default None)
            Flag to ask for MCMC attributes instead of model's attributes.

        Returns
        -------
        :class:`torch.Tensor` of shape (n_individuals,)
            Contains L2 residual for each subject
        """
        L2_res_per_ind_per_ft = self.compute_sum_squared_per_ft_tensorized(dataset, param_ind, attribute_type=attribute_type)
        return L2_res_per_ind_per_ft.sum(dim=1)  # sum on features

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
                len(v) # exclude np.array(scalar) or torch.tensor(scalar)
                return hasattr(v, '__getitem__') # exclude set
            except Exception:
                return False

        # Model supports and needs sources?
        has_sources = hasattr(self, 'source_dimension') and isinstance(self.source_dimension, int) and self.source_dimension > 0

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
        ips_is_array_like = {k: is_array_like(v) for k,v in ips.items()}
        ips_size = {k: len(v) if ips_is_array_like[k] else 1 for k,v in ips.items()}

        if has_sources:
            s = ips['sources']

            if not ips_is_array_like['sources']:
                raise LeaspyIndividualParamsInputError(f'Sources must be an array_like but {s} was provided.')

            tau_xi_scalars = all(ips_size[k] == 1 for k in ['tau','xi'])
            if tau_xi_scalars and (ips_size['sources'] > 1):
                # is 'sources' not a nested array? (allowed iff tau & xi are scalars)
                if not is_array_like(s[0]):
                    # then update sources size (1D vector representing only 1 individual)
                    ips_size['sources'] = 1

            # TODO? check source dimension compatibility?

        uniq_sizes = set(ips_size.values())
        if len(uniq_sizes) != 1:
            raise LeaspyIndividualParamsInputError('Individual parameters sizes are not compatible together. '
                                                  f'Sizes are {ips_size}.')

        # number of individuals present
        n_inds = uniq_sizes.pop()

        # properly choose unsqueezing dimension when tensorizing array_like (useful for sources)
        unsqueeze_dim = -1 # [1,2] => [[1],[2]] (expected for 2 individuals / 1D sources)
        if n_inds == 1:
            unsqueeze_dim = 0 # [1,2] => [[1,2]] (expected for 1 individual / 2D sources)

        # tensorized (2D) version of ips
        t_ips = {k: self._tensorize_2D(v, unsqueeze_dim=unsqueeze_dim) for k,v in ips.items()}

        # construct logs
        return {
            'nb_inds': n_inds,
            'tensorized_ips': t_ips,
            'tensorized_ips_gen': ({k: v[i,:].unsqueeze(0) for k,v in t_ips.items()} for i in range(n_inds))
        }

    @staticmethod
    def _tensorize_2D(x, unsqueeze_dim: int, dtype=torch.float32) -> torch.FloatTensor:
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

    def _get_tensorized_inputs(self, timepoints, individual_parameters, *,
                               skip_ips_checks: bool = False) -> Tuple[torch.FloatTensor, DictParamsTorch]:
        if not skip_ips_checks:
            # Perform checks on ips and gets tensorized version if needed
            ips_info = self._audit_individual_parameters(individual_parameters)
            n_inds = ips_info['nb_inds']
            individual_parameters = ips_info['tensorized_ips']

            if n_inds != 1:
                raise LeaspyModelInputError('Only one individual computation may be performed at a time. '
                                           f'{n_inds} was provided.')

        # Convert the timepoints (list of numbers, or single number) to a 2D torch tensor
        timepoints = self._tensorize_2D(timepoints, unsqueeze_dim=0) # 1 individual
        return timepoints, individual_parameters

    # TODO: unit tests? (functional tests covered by api.estimate)
    def compute_individual_trajectory(self, timepoints, individual_parameters: DictParams, *,
                                      skip_ips_checks: bool = False):
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

        timepoints, individual_parameters = self._get_tensorized_inputs(timepoints, individual_parameters,
                                                                        skip_ips_checks=skip_ips_checks)
        # Compute the individual trajectory
        return self.compute_individual_tensorized(timepoints, individual_parameters)

    # TODO: unit tests? (functional tests covered by api.estimate)
    def compute_individual_ages_from_biomarker_values(self, value: Union[float, List[float]], individual_parameters: DictParams, feature: FeatureType = None):
        """
        For one individual, compute age(s) at which the given features values are reached (given the subject's
        individual parameters).

        Consistency checks are done in the main API layer.

        Parameters
        ----------
        value : scalar or array_like[scalar] (list, tuple, :class:`numpy.ndarray`)
            Contains the biomarker value(s) of the subject.

        individual_parameters : dict
            Contains the individual parameters.
            Each individual parameter should be a scalar or array_like

        feature : str (or None)
            Name of the considered biomarker (optional for univariate models, compulsory for multivariate models).

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
        value, individual_parameters = self._get_tensorized_inputs(value, individual_parameters,
                                                                   skip_ips_checks=False)
        # Compute the individual trajectory
        return self.compute_individual_ages_from_biomarker_values_tensorized(value, individual_parameters, feature)

    @abstractmethod
    def compute_individual_ages_from_biomarker_values_tensorized(self, value: torch.FloatTensor,
                                                                 individual_parameters: DictParamsTorch,
                                                                 feature: Optional[FeatureType]) -> torch.FloatTensor:
        """
        For one individual, compute age(s) at which the given features values are reached (given the subject's
        individual parameters), with tensorized inputs

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
    def compute_individual_tensorized(self, timepoints: torch.FloatTensor, individual_parameters: DictParamsTorch, *,
                                      attribute_type=None) -> torch.FloatTensor:
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
    def compute_jacobian_tensorized(self, timepoints: torch.FloatTensor, individual_parameters: DictParamsTorch, *,
                                    attribute_type=None) -> torch.FloatTensor:
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

    def compute_individual_attachment_tensorized(self, data: Dataset, param_ind: DictParamsTorch, *,
                                                 attribute_type) -> torch.FloatTensor:
        """
        Compute attachment term (per subject)

        Parameters
        ----------
        data : :class:`.Dataset`
            Contains the data of the subjects, in particular the subjects' time-points and the mask for nan values & padded visits

        param_ind : dict
            Contain the individual parameters

        attribute_type : Any
            Flag to ask for MCMC attributes instead of model's attributes.

        Returns
        -------
        attachment : :class:`torch.Tensor`
            Negative Log-likelihood, shape = (n_subjects,)

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If invalid `noise_model` for model
        """

        # TODO: this snippet could be implemented directly in NoiseModel (or subclasses depending on noise structure)
        if self.noise_model is None:
            raise LeaspyModelInputError('`noise_model` was not set correctly set.')

        elif 'gaussian' in self.noise_model:
            # diagonal noise (squared) [same for all features if it's forced to be a scalar]
            # TODO? shouldn't 'noise_std' be part of the "MCMC_toolbox" to use the one we want??
            noise_var = self.parameters['noise_std'] * self.parameters['noise_std'] # slight perf improvement over ** 2, k tensor (or scalar tensor)
            noise_var = noise_var.expand((1, data.dimension)) # 1,k tensor (for scalar products just after) # <!> this formula works with scalar noise as well

            L2_res_per_ind_per_ft = self.compute_sum_squared_per_ft_tensorized(data, param_ind, attribute_type=attribute_type) # ik tensor

            attachment = (0.5 / noise_var) @ L2_res_per_ind_per_ft.t()
            attachment += 0.5 * torch.log(TWO_PI * noise_var) @ data.n_observations_per_ind_per_ft.float().t()

        elif self.noise_model == 'bernoulli':
            pred = self.compute_individual_tensorized(data.timepoints, param_ind, attribute_type=attribute_type)
            mask = data.mask.float()

            pred = torch.clamp(pred, 1e-7, 1. - 1e-7) # safety before taking the log
            neg_crossentropy = data.values * torch.log(pred) + (1. - data.values) * torch.log(1. - pred)
            attachment = -torch.sum(mask * neg_crossentropy, dim=(1, 2))

        else:
            raise LeaspyModelInputError(f'`noise_model` should be in {NoiseModel.VALID_NOISE_STRUCTS}')

        return attachment.reshape((data.n_individuals,)) # 1D tensor of shape(n_individuals,)

    @abstractmethod
    def update_model_parameters_burn_in(self, data: Dataset, realizations: CollectionRealization) -> None:
        """
        Update model parameters (burn-in phase)

        Parameters
        ----------
        data : :class:`.Dataset`
        realizations : :class:`.CollectionRealization`
        """

    @abstractmethod
    def update_model_parameters_normal(self, data: Dataset, suff_stats: DictParamsTorch) -> None:
        """
        Update model parameters (after burn-in phase)

        Parameters
        ----------
        data : :class:`.Dataset`
        suff_stats : dict[suff_stat: str, :class:`torch.Tensor`]
        """

    @abstractmethod
    def compute_sufficient_statistics(self, data: Dataset, realizations: CollectionRealization) -> DictParamsTorch:
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

    def get_population_realization_names(self) -> List[str]:
        """
        Get names of population variables of the model.

        Returns
        -------
        list[str]
        """
        return [name for name, value in self.random_variable_informations().items()
                if value['type'] == 'population']

    def get_individual_realization_names(self) -> List[str]:
        """
        Get names of individual variables of the model.

        Returns
        -------
        list[str]
        """
        return [name for name, value in self.random_variable_informations().items()
                if value['type'] == 'individual']

    def __str__(self):
        output = "=== MODEL ==="
        for p, v in self.parameters.items():
            if isinstance(v, float) or (hasattr(v, 'ndim') and v.ndim == 0):
                # for 0D tensors / arrays the default behavior is to print all digits...
                # change this!
                v_repr = f'{v:.{1+torch_print_opts.precision}g}'
            else:
                # torch.tensor, np.array, ...
                # in particular you may use `torch.set_printoptions` and `np.set_printoptions` globally
                # to tune the number of decimals when printing tensors / arrays
                v_repr = str(v)
                # remove tensor prefix & possible dtype suffix
                v_repr = re.sub(r'^[^\(]+\(', '', v_repr)
                v_repr = re.sub(r'(?:, dtype=.+)?\)$', '', v_repr)
                # adjust justification
                spaces = " "*len(f"{p} : [")
                v_repr = re.sub(r'\n[ ]+\[', f'\n{spaces}[', v_repr)

            output += f"\n{p} : {v_repr}"
        return output

    def compute_regularity_realization(self, realization: Realization):
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
            raise LeaspyModelInputError(f"Variable type '{realization.variable_type}' not known, should be 'population' or 'individual'.")

        return self.compute_regularity_variable(realization.tensor_realizations, mean, std)

    def compute_regularity_variable(self, value: torch.FloatTensor, mean: torch.FloatTensor, std: torch.FloatTensor) -> torch.FloatTensor:
        """
        Compute regularity term (Gaussian distribution), low-level.

        TODO: should be encapsulated in a RandomVariableSpecification class together with other specs of RV.

        Parameters
        ----------
        value, mean, std : :class:`torch.Tensor` of same shapes

        Returns
        -------
        :class:`torch.Tensor` of same shape than input
        """
        return -self.regularization_distribution_factory(mean, std).log_prob(value)

    def initialize_realizations_for_model(self, n_individuals: int, **init_kws) -> CollectionRealization:
        """
        Initialize a :class:`.CollectionRealization` used during model fitting or mode/mean realization personalization.

        Parameters
        ----------
        n_individuals : int
            Number of individuals to track
        **init_kws
            Keyword arguments passed to :meth:`.CollectionRealization.initialize`.
            (In particular `individual_variable_init_at_mean` to "initialize at mean" or `skip_variable` to filter some variables)

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

    def smart_initialization_realizations(self, dataset: Dataset, realizations: CollectionRealization) -> CollectionRealization:
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

    def _create_dictionary_of_population_realizations(self):
        pop_dictionary: Dict[str, Realization] = {}
        for name_var, info_var in self.random_variable_informations().items():
            if info_var['type'] != "population":
                continue
            real = Realization.from_tensor(name_var, info_var['shape'], info_var['type'], self.parameters[name_var])
            pop_dictionary[name_var] = real

        return pop_dictionary

    @staticmethod
    def time_reparametrization(timepoints: torch.FloatTensor, xi: torch.FloatTensor, tau: torch.FloatTensor) -> torch.FloatTensor:
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

    @staticmethod
    def _compute_std_from_var(variance: torch.FloatTensor, *, varname: str, tol: float = 1e-5) -> torch.FloatTensor:
        """
        Check that variance is strictly positive and return its square root, otherwise fail with a convergence error.

        If variance is multivariate check that all components are strictly positive.

        TODO? a full Bayesian setting with good priors on all variables should prevent such convergence issues.

        Parameters
        ----------
        var : :class:`torch.Tensor`
            The variance we would like to convert to a std-dev.
        varname : str
            The name of the variable - to display a nice error message.
        tol : float
            The lower bound on variance, under which the converge error is raised.

        Returns
        -------
        torch.FloatTensor

        Raises
        ------
        :exc:`.LeaspyConvergenceError`
        """
        if (variance < tol).any():
            raise LeaspyConvergenceError(f"The parameter '{varname}' collapsed to zero, which indicates a convergence issue.\n"
                                         "Start by investigating what happened in the logs of your calibration and try to double check:"
                                         "\n- your training dataset (not enough subjects and/or visits? too much missing data?)"
                                         "\n- the hyperparameters of your Leaspy model (`source_dimension` too low or too high? "
                                         "`noise_model` not suited to your data?)"
                                         "\n- the hyperparameters of your calibration algorithm"
                                        )

        return variance.sqrt()

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
