from __future__ import annotations
from typing import TYPE_CHECKING
import math
from abc import ABC, abstractmethod
import copy

import torch

from leaspy.models.utils import DEFAULT_LOSS, VALID_LOSSES
from leaspy.io.realizations.collection_realization import CollectionRealization
from leaspy.io.realizations.realization import Realization

from leaspy.exceptions import LeaspyIndividualParamsInputError, LeaspyModelInputError
from leaspy.utils.typing import FeatureType, KwargsType, DictParams, DictParamsTorch, Union, List, Dict, Tuple, Optional

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
    loss : str
        The loss to optimize (``'MSE'``, ``'MSE_diag_noise'`` or ``'crossentropy'``)
    distribution : :class:`torch.distributions.normal.Normal`
        Gaussian distribution object to compute variables regularization
    """

    def __init__(self, name: str):
        self.is_initialized: bool = False
        self.name = name
        self.features: List[FeatureType] = None
        self.parameters: KwargsType = None
        self.loss: str = DEFAULT_LOSS  # default value, changes when a fit / personalize algo is called, TODO: change to MSE_diag_noise ?
        self.distribution = torch.distributions.normal.Normal(loc=0., scale=1.)

    @abstractmethod
    def initialize(self, dataset: Dataset, method: str = 'default'):
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
        pass

    def load_parameters(self, parameters: KwargsType):
        """
        Instantiate or update the model's parameters.

        Parameters
        ----------
        parameters : dict[str, Any]
            Contains the model's parameters
        """
        self.parameters = copy.deepcopy(parameters)

    @abstractmethod
    def load_hyperparameters(self, hyperparameters: KwargsType):
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
        pass

    @abstractmethod
    def save(self, path: str, **kwargs):
        """
        Save Leaspy object as json model parameter file.

        Parameters
        ----------
        path : str
            Path to store the model's parameters.
        **kwargs
            Keyword arguments for json.dump method.
        """
        pass

    def get_individual_variable_name(self):
        """
        Return list of names of the individual variables from the model.

        Duplicate of :meth:`.get_individual_realization_names`

        TODO delete one of them

        Returns
        -------
        individual_variable_name : list [str]
            Contains the individual variables' names
        """
        return self.get_individual_realization_names()

    def compute_sum_squared_per_ft_tensorized(self, data: Dataset, param_ind: DictParamsTorch, attribute_type=None) -> torch.FloatTensor:
        """
        Compute the square of the residuals per subject per feature

        Parameters
        ----------
        data : :class:`.Dataset`
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
        res: torch.FloatTensor = self.compute_individual_tensorized(data.timepoints, param_ind, attribute_type)
        r1 = data.mask.float() * (res - data.values) # ijk tensor (i=individuals, j=visits, k=features)
        return torch.sum(r1 * r1, dim=1)

    def compute_sum_squared_tensorized(self, data: Dataset, param_ind: DictParamsTorch, attribute_type=None) -> torch.FloatTensor:
        """
        Compute the square of the residuals per subject

        Parameters
        ----------
        data : :class:`.Dataset`
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
        L2_res_per_ind_per_ft = self.compute_sum_squared_per_ft_tensorized(data, param_ind, attribute_type)
        return torch.sum(L2_res_per_ind_per_ft, dim=1) # sum on features

    def _audit_individual_parameters(self, ips: DictParams) -> KwargsType:
        """
        Perform various consistency and compatibility (with current model) checks
        on an individual parameters dict and outputs qualified information about it.

        TODO? move to IndividualParameters class?

        Parameters
        ----------
        ips : dict[param: str, Any]
            Contains some untrusted individual parameters.
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
    def compute_individual_ages_from_biomarker_values_tensorized(self, value: torch.FloatTensor, individual_parameters: DictParamsTorch,
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
        pass

    @abstractmethod
    def compute_individual_tensorized(self, timepoints: torch.FloatTensor, individual_parameters: DictParamsTorch, attribute_type=None) -> torch.FloatTensor:
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
        pass

    @abstractmethod
    def compute_jacobian_tensorized(self, timepoints: torch.FloatTensor, ind_parameters: DictParamsTorch, attribute_type=None) -> torch.FloatTensor:
        """
        Compute the jacobian of the model w.r.t. each individual parameter.

        This function aims to be used in :class:`.ScipyMinimize` to speed up optimization.

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
        pass

    def compute_individual_attachment_tensorized_mcmc(self, data: Dataset, realizations: CollectionRealization):
        """
        Compute MCMC attachment of all subjects? One subject? One visit?
        TODO: complete

        Parameters
        ----------
        data : :class:`.Dataset`
            Contains the data of the subjects, in particular the subjects' time-points and the mask (?)
        realizations : :class:`.CollectionRealization`

        Returns
        -------
        attachment : :class:`torch.Tensor`
            The subject attachment (?)
        """
        param_ind = self.get_param_from_real(realizations)
        attachment = self.compute_individual_attachment_tensorized(data, param_ind, attribute_type='MCMC')
        return attachment

    def compute_individual_attachment_tensorized(self, data: Dataset, param_ind: DictParamsTorch, attribute_type) -> torch.FloatTensor:
        """
        Compute attachment term (per subject)

        Parameters
        ----------
        data : :class:`.Dataset`
            Contains the data of the subjects, in particular the subjects' time-points and the mask for nan values & padded visits

        param_ind : dict
            Contain the individual parameters

        attribute_type : Any, optional
            Flag to ask for MCMC attributes instead of model's attributes.

        Returns
        -------
        attachment : :class:`torch.Tensor`
            Negative Log-likelihood, shape = (n_subjects,)

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If invalid loss for model
        """

        if 'MSE' in self.loss:
            # diagonal noise (squared) [same for all features if it's forced to be a scalar]
            noise_var = self.parameters['noise_std'] * self.parameters['noise_std'] # slight perf improvement over ** 2, k tensor (or scalar tensor)
            noise_var = noise_var.expand((1, data.dimension)) # 1,k tensor (for scalar products just after) # <!> this formula works with scalar noise as well

            L2_res_per_ind_per_ft = self.compute_sum_squared_per_ft_tensorized(data, param_ind, attribute_type) # ik tensor

            attachment = (0.5 / noise_var) @ L2_res_per_ind_per_ft.t()
            attachment += 0.5 * torch.log(TWO_PI * noise_var) @ data.n_observations_per_ind_per_ft.float().t()

        elif self.loss == 'crossentropy':
            pred = self.compute_individual_tensorized(data.timepoints, param_ind, attribute_type)
            mask = data.mask.float()

            pred = torch.clamp(pred, 1e-38, 1. - 1e-7) # safety before taking the log
            neg_crossentropy = data.values * torch.log(pred) + (1. - data.values) * torch.log(1. - pred)
            attachment = -torch.sum(mask * neg_crossentropy, dim=(1, 2))

        else:
            raise LeaspyModelInputError(f'Model loss should be in {VALID_LOSSES}')

        return attachment.reshape((data.n_individuals,)) # 1D tensor of shape(n_individuals,)

    def update_model_parameters(self, data: Dataset, reals_or_suff_stats: Union[CollectionRealization, DictParamsTorch], burn_in_phase=True):
        """
        Update model parameters (high-level function)

        Under-the-hood call :meth:`.update_model_parameters_burn_in` or :meth:`.update_model_parameters_normal` depending on the phase of the fit algorithm

        Parameters
        ----------
        data : :class:`.Dataset`
        reals_or_suff_stats :
            If during burn-in phase will be realizations:
                :class:`.CollectionRealization`
            If after burn-in phase will be sufficient statistics:
                dict[suff_stat: str, :class:`torch.Tensor`]
        """

        # Memoryless part of the algorithm
        if burn_in_phase:
            self.update_model_parameters_burn_in(data, reals_or_suff_stats)
        # Stochastic sufficient statistics used to update the parameters of the model
        else:
            self.update_model_parameters_normal(data, reals_or_suff_stats)
        self.attributes.update(['all'], self.parameters)

    @abstractmethod
    def update_model_parameters_burn_in(self, data: Dataset, realizations: CollectionRealization):
        """
        Update model parameters (burn-in phase)

        Parameters
        ----------
        data : :class:`.Dataset`
        realizations : :class:`.CollectionRealization`
        """
        pass

    @abstractmethod
    def update_model_parameters_normal(self, data: Dataset, suff_stats: DictParamsTorch):
        """
        Update model parameters (after burn-in phase)

        Parameters
        ----------
        data : :class:`.Dataset`
        suff_stats : dict[suff_stat: str, :class:`torch.Tensor`]
        """
        pass

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
        pass

    def get_population_realization_names(self):
        """
        Get names of population variales of the model.

        Returns
        -------
        list[str]
        """
        return [name for name, value in self.random_variable_informations().items()
                if value['type'] == 'population']

    def get_individual_realization_names(self):
        """
        Get names of individual variales of the model.

        Returns
        -------
        list[str]
        """
        return [name for name, value in self.random_variable_informations().items()
                if value['type'] == 'individual']

    def __str__(self):
        output = "=== MODEL ===\n"
        for key in self.parameters.keys():
            output += f"{key} : {self.parameters[key]}\n"
        return output

    def compute_regularity_realization(self, realization: Realization):
        """
        Compute regularity term for a :class:`.Realization` instance.

        Parameters
        ----------
        realization : :class:`.Realization`

        Returns
        -------
        :class:`torch.Tensor`
        """

        # Instanciate torch distribution
        if realization.variable_type == 'population':
            mean = self.parameters[realization.name]
            # TODO : Sure it is only MCMC_toolbox?
            std = self.MCMC_toolbox['priors'][f"{realization.name}_std"]
        elif realization.variable_type == 'individual':
            mean = self.parameters[f"{realization.name}_mean"]
            std = self.parameters[f"{realization.name}_std"]
        else:
            raise LeaspyModelInputError(f"Variable type '{realization.variable_type}' not known, should be 'population' or 'individual'.")

        return self.compute_regularity_variable(realization.tensor_realizations, mean, std)

    def compute_regularity_variable(self, value: torch.FloatTensor, mean: torch.FloatTensor, std: torch.FloatTensor) -> torch.FloatTensor:
        """
        Compute regularity term (Gaussian distribution), low-level.

        Parameters
        ----------
        value, mean, std : :class:`torch.Tensor` of same shapes
        """

        # TODO change to static ???
        # Instanciate torch distribution
        # distribution = torch.distributions.normal.Normal(loc=mean, scale=std)

        self.distribution.loc = mean
        self.distribution.scale = std

        return -self.distribution.log_prob(value)

    def get_realization_object(self, n_individuals: int) -> CollectionRealization:
        """
        Initialization of a :class:`.CollectionRealization` used during model fitting.

        Parameters
        ----------
        n_individuals : int
            Number of individuals to track

        Returns
        -------
        :class:`.CollectionRealization`
        """

        # TODO : CollectionRealizations should probably get self.get_info_var rather than all self
        realizations = CollectionRealization()
        realizations.initialize(n_individuals, self)
        return realizations

    @abstractmethod
    def random_variable_informations(self) -> DictParams:
        """
        Informations on model's random variables.

        Returns
        -------
        dict[str, Any]
        """
        pass

    def smart_initialization_realizations(self, data: Dataset, realizations: CollectionRealization):
        """
        Smart initialization of realizations if needed.

        Default behavior to return `realizations` as they are (no smart trick).

        Parameters
        ----------
        data : :class:`.Dataset`
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

        Parameters
        ----------
        realizations : :class:`.CollectionRealization`

        Returns
        -------
        dict[param_name: str, :class:`torch.Tensor` [n_individuals, dims_param]]
            Individual parameters
        """

        individual_parameters = dict.fromkeys(self.get_individual_variable_name())

        for variable_ind in self.get_individual_variable_name():
            if variable_ind == "sources" and getattr(self, 'source_dimension', 0) == 0:
                individual_parameters[variable_ind] = None
            else:
                individual_parameters[variable_ind] = realizations[variable_ind].tensor_realizations

        return individual_parameters
