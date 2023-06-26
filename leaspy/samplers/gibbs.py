import abc
from random import shuffle
from typing import ClassVar
from collections.abc import Sequence

import torch
from numpy import ndindex

from .base import AbstractPopulationSampler, AbstractIndividualSampler
from leaspy.exceptions import LeaspyInputError
from leaspy.utils.typing import Union, Tuple, List
from leaspy.variables.state import State


IteratorIndicesType = List[Tuple[int, ...]]


class GibbsSamplerMixin:
    """
    Mixin class for all Gibbs samplers (individual and population).

    This class contains the logic common to all Gibbs samplers.

    Parameters
    ----------
    name : :obj:`str`
        The name of the random variable to sample.
    shape : :obj:`tuple` of :obj:`int`
        The shape of the random variable to sample.
    scale : :obj:`float` > 0 or :class:`torch.FloatTensor` > 0
        An approximate scale for the variable.
        It will be used to scale the initial adaptive std-dev used in sampler.
        An extra factor will be applied on top of this scale (hyperparameters):
            * 1% for population parameters (:attr:`.AbstractPopulationGibbsSampler.STD_SCALE_FACTOR`)
            * 50% for individual parameters (:attr:`.IndividualGibbsSampler.STD_SCALE_FACTOR`)
        Note that if you pass a :class:`torch.Tensor`, its shape should be compatible with shape of the variable.
    random_order_dimension : :obj:`bool` (default True)
        This parameter controls whether we randomize the order of indices during the sampling loop.
        (only for population variables, since we perform group sampling for individual variables)
        Article https://proceedings.neurips.cc/paper/2016/hash/e4da3b7fbbce2345d7772b0674a318d5-Abstract.html
        gives a rationale on why we should activate this flag.
    mean_acceptation_rate_target_bounds : :obj:`tuple`[lower_bound: float, upper_bound: float] with 0 < lower_bound < upper_bound < 1
        Bounds on mean acceptation rate.
        Outside this range, the adaptation of the std-dev of sampler is triggered
        so to maintain a target acceptation rate in between these too bounds (e.g: ~30%).
    adaptive_std_factor : :obj:`float` in ]0, 1[
        Factor by which we increase or decrease the std-dev of sampler when we are out of
        the custom bounds for the mean acceptation rate. We decrease it by `1 - factor` if too low,
        and increase it with `1 + factor` if too high.
    **base_sampler_kws
        Keyword arguments passed to `AbstractSampler` init method.
        In particular, you may pass the `acceptation_history_length` hyperparameter.

    Attributes
    ----------
    In addition to the attributes present in :class:`.AbstractSampler`:

    std : :class:`torch.Tensor`
        Adaptive std-dev of variable

    Raises
    ------
    :exc:`.LeaspyInputError`
    """

    STD_SCALE_FACTOR: ClassVar[float]

    def __init__(
        self,
        name: str,
        shape: tuple,
        *,
        scale: Union[float, torch.FloatTensor],
        random_order_dimension: bool = True,
        mean_acceptation_rate_target_bounds: Tuple[float, float] = (0.2, 0.4),
        adaptive_std_factor: float = 0.1,
        **base_sampler_kws,
    ):
        super().__init__(name, shape, **base_sampler_kws)
        self.scale = self.validate_scale(scale)
        self.std = self.STD_SCALE_FACTOR * self.scale * torch.ones(self.shape_adapted_std)
        # Internal counter to trigger adaptation of std based on mean acceptation rate
        self._counter: int = 0
        # Parameters of the sampler
        self._random_order_dimension = random_order_dimension
        self._set_acceptation_bounds(mean_acceptation_rate_target_bounds)
        self._set_adaptive_std_factor(adaptive_std_factor)

    @property
    @abc.abstractmethod
    def shape_adapted_std(self) -> Tuple[int, ...]:
        """Shape of adaptative variance."""

    @property
    def shape_acceptation(self) -> Tuple[int, ...]:
        return self.shape_adapted_std

    def _set_acceptation_bounds(self, mean_acceptation_rate_target_bounds: Tuple[float, float]) -> None:
        if not (
            isinstance(mean_acceptation_rate_target_bounds, Sequence)
            and len(mean_acceptation_rate_target_bounds) == 2
            and 0 < mean_acceptation_rate_target_bounds[0] < mean_acceptation_rate_target_bounds[1] < 1
        ):
            raise LeaspyInputError(
                "`mean_acceptation_rate_target_bounds` should be a tuple (lower_bound, upper_bound) "
                f"with 0 < lower_bound < upper_bound < 1, not '{mean_acceptation_rate_target_bounds}'"
            )
        (
            self._mean_acceptation_lower_bound_before_adaptation,
            self._mean_acceptation_upper_bound_before_adaptation,
        ) = mean_acceptation_rate_target_bounds

    def _set_adaptive_std_factor(self, adaptive_std_factor: float) -> None:
        if not (0 < adaptive_std_factor < 1):
            raise LeaspyInputError(
                f"`adaptive_std_factor` should be a float in ]0, 1[, not '{adaptive_std_factor}'"
            )
        self._adaptive_std_factor = adaptive_std_factor

    def validate_scale(self, scale: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Validate user provided scale in :obj:`float` or :class:`torch.Tensor` form.

        Scale of variable should always be positive (component-wise if multidimensional).

        Parameters
        ----------
        scale : :obj:`float` or :class:`torch.Tensor`
            The scale to be validated.

        Returns
        -------
        :class:`torch.Tensor` :
            Valid scale.
        """
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
        scale = scale.float()
        if (scale <= 0).any():
            raise LeaspyInputError(
                f"Scale of variable '{self.name}' should be positive, not `{scale}`."
            )
        return scale

    def __str__(self):
        mean_std = self.std[self._meaningful_indices].mean()
        mean_acceptation_rate = self.acceptation_history[(slice(None),) + self._meaningful_indices].mean()
        return f"{self.name} rate : {mean_acceptation_rate.item():.1%}, std: {mean_std.item():.1e}"

    @property
    def _meaningful_indices(self) -> Tuple[torch.Tensor, ...]:
        """
        Return the subset of indices that are relevant for both adapted-variance
        and acceptations tensors.

        By default, this is an empty :obj:`tuple` which corresponds to no masking at all
        of adapted-variances nor acceptations tensors.

        Returns
        -------
        :obj:`tuple` of :obj:`int` :
            The meaningful indices.
        """
        return ()

    def _update_std(self):
        """
        Update standard deviation of sampler according to current frequency of acceptation.

        Adaptive std is known to improve sampling performances.

        For default parameters:
            - `std-dev` is increased if frequency of acceptation is > 40%
            - `std-dev` is decreased if frequency of acceptation is < 20%
              (so as to stay close to 30%).
        """
        self._counter += 1

        if self._counter % self.acceptation_history_length == 0:
            mean_acceptation = self.acceptation_history.mean(dim=0)

            # nota: for masked elements in full Gibbs, std will always remain = 0
            idx_toolow = mean_acceptation < self._mean_acceptation_lower_bound_before_adaptation
            idx_toohigh = mean_acceptation > self._mean_acceptation_upper_bound_before_adaptation

            self.std[idx_toolow] *= (1 - self._adaptive_std_factor)
            self.std[idx_toohigh] *= (1 + self._adaptive_std_factor)


class AbstractPopulationGibbsSampler(GibbsSamplerMixin, AbstractPopulationSampler):
    """
    Abstract class for all Gibbs samplers for population variables.

    Parameters
    ----------
    name : str
        The name of the random variable to sample.
    shape : tuple[int, ...]
        The shape of the random variable to sample.
    scale : float > 0 or :class:`torch.FloatTensor` > 0
        An approximate scale for the variable.
        It will be used to scale the initial adaptive std-dev used in sampler.
        An extra 1% factor will be applied on top of this scale (STD_SCALE_FACTOR)
        Note that if you pass a torch tensor, its shape should be compatible with shape of the variable.
    random_order_dimension : bool (default True)
        This parameter controls whether we randomize the order of indices during the sampling loop.
        Article https://proceedings.neurips.cc/paper/2016/hash/e4da3b7fbbce2345d7772b0674a318d5-Abstract.html
        gives a rationale on why we should activate this flag.
    mean_acceptation_rate_target_bounds : tuple[lower_bound: float, upper_bound: float] with 0 < lower_bound < upper_bound < 1
        Bounds on mean acceptation rate.
        Outside this range, the adaptation of the std-dev of sampler is triggered
        so to maintain a target acceptation rate in between these too bounds (e.g: ~30%).
    adaptive_std_factor : float in ]0, 1[
        Factor by which we increase or decrease the std-dev of sampler when we are out of
        the custom bounds for the mean acceptation rate. We decrease it by `1 - factor` if too low,
        and increase it with `1 + factor` if too high.
    **base_sampler_kws
        Keyword arguments passed to `AbstractSampler` init method.
        In particular, you may pass the `acceptation_history_length` hyperparameter.
    """

    STD_SCALE_FACTOR = 0.01

    def validate_scale(self, scale: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Validate user provided scale in :obj:`float` or :class:`torch.Tensor` form.

        If necessary, scale is casted to a :class:`torch.Tensor`.

        Parameters
        ----------
        scale : :obj:`float` or :class:`torch.Tensor`
            The scale to be validated.

        Returns
        -------
        :class:`torch.Tensor` :
            Valid scale.
        """
        scale = super().validate_scale(scale)
        if scale.ndim > len(self.shape_adapted_std):
            # we take the mean of grouped dimension in this case
            scale_squeezed_dims = tuple(range(len(self.shape_adapted_std), scale.ndim))
            scale = scale.mean(dim=scale_squeezed_dims)
        return scale

    def sample(
        self,
        state: State,
        *,
        temperature_inv: float,
    ) -> None:
        """
        For each dimension (1D or 2D) of the population variable, compute current attachment and regularity.

        Propose a new value for the given dimension of the given population variable,
        and compute new attachment and regularity thanks to State.

        Do a MH step, keeping if better, or if worse with a probability.

        Parameters
        ----------
        state : :class:`.State`
            Instance holding values for all model variables (including latent variables), as well as:
            - timepoints : :class:`torch.Tensor` of shape (n_individuals, n_timepoints)
            - dataset : ...
                Contains the data of the subjects, in particular the subjects'
                time-points and the mask for nan values & padded visits
        temperature_inv : :obj:`float` > 0
            The temperature to use.
        """
        accepted_array = torch.zeros_like(self.std)

        def compute_attachment_regularity():
            # Mask for regularity is handled directly by `WeightedTensor` logic
            return state["nll_attach"], state[f"nll_regul_{self.name}"]

        for idx in self._get_shuffled_iterator_indices():
            previous_attachment, previous_regularity = compute_attachment_regularity()
            # with state.auto_fork():  # not needed since state already have auto_fork on
            state.put(
                self.name,
                self._proposed_change_idx(idx),
                indices=idx,
                accumulate=True,  # out-of-place addition
            )

            # Update (and caching) of derived model attributes (orthonormal basis, ...) is done in state
            new_attachment, new_regularity = compute_attachment_regularity()
            alpha = torch.exp(
                -1 * (
                    (new_regularity - previous_regularity) * temperature_inv +
                    (new_attachment - previous_attachment)
                )
            )
            accepted = self._metropolis_step(alpha)
            accepted_array[idx] = accepted

            if not accepted:
                state.revert()

        self._update_acceptation_rate(accepted_array)
        self._update_std()

    def _get_shuffled_iterator_indices(self) -> IteratorIndicesType:
        indices = self._get_iterator_indices()
        if self._random_order_dimension:
            shuffle(indices)  # shuffle in-place!
        return indices

    def _get_iterator_indices(self) -> IteratorIndicesType:
        """
        When a mask is set on the random variable and when we sample without any grouping
        (i.e. regular Gibbs sampler - i.e. shape of `std` is same as shape of `mask`)
        then we'll only loop on coordinates not masked (we skip the un-needed indices)

        Depending on sampler type we will loop on all coordinates or (partially) group them:
        Example for variable of shape (2, 3)
            * for 'Gibbs' sampler: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
            * for 'FastGibbs' sampler: [(0,), (1,)]
            * for 'Metropolis-Hastings' sampler: [()]

        Returns
        -------
        IteratorIndicesType :
            The indices for the iterator.
        """
        return list(ndindex(self.shape_adapted_std))

    @property
    def _should_mask_changes(self) -> bool:
        return self.mask is not None

    def _proposed_change_idx(self, idx: Tuple[int, ...]) -> torch.Tensor:
        """
        The previous version with `_proposal` was not incorrect but computationally inefficient:
        because we were sampling on the full shape of `std` whereas we only needed `std[idx]` (smaller)

        We don't directly mask the `new_val_idx` since it may be infinite, producing nans
        when trying to multiply them by 0.

        Returns
        -------
        :class:`torch.Tensor` :
            The proposed change for the given index.
        """
        shape_idx = self.shape[slice(len(idx), self.ndim)]
        change_idx = self.std[idx] * torch.randn(shape_idx)
        if self._should_mask_changes:
            change_idx = change_idx * self.mask[idx].float()
        return change_idx


class PopulationGibbsSampler(AbstractPopulationGibbsSampler):
    """
    Gibbs sampler for population variables.

    The sampling is done iteratively for all coordinate values.

    Parameters
    ----------
    name : :obj:`str`
        The name of the random variable to sample.
    shape : :obj:`tuple` of :obj:`int`
        The shape of the random variable to sample.
    scale : :obj:`float` > 0 or :class:`torch.FloatTensor` > 0
        An approximate scale for the variable.
        It will be used to scale the initial adaptive std-dev used in sampler.
        An extra 1% factor will be applied on top of this scale (STD_SCALE_FACTOR)
        Note that if you pass a torch tensor, its shape should be compatible with shape of the variable.
    random_order_dimension : :obj:`bool` (default True)
        This parameter controls whether we randomize the order of indices during the sampling loop.
        Article https://proceedings.neurips.cc/paper/2016/hash/e4da3b7fbbce2345d7772b0674a318d5-Abstract.html
        gives a rationale on why we should activate this flag.
    mean_acceptation_rate_target_bounds : :obj:`tuple`[lower_bound: float, upper_bound: float] with 0 < lower_bound < upper_bound < 1
        Bounds on mean acceptation rate.
        Outside this range, the adaptation of the std-dev of sampler is triggered
        so to maintain a target acceptation rate in between these too bounds (e.g: ~30%).
    adaptive_std_factor : :obj:`float` in ]0, 1[
        Factor by which we increase or decrease the std-dev of sampler when we are out of
        the custom bounds for the mean acceptation rate. We decrease it by `1 - factor` if too low,
        and increase it with `1 + factor` if too high.
    **base_sampler_kws
        Keyword arguments passed to :meth:`.AbstractSampler.__init__` method.
        In particular, you may pass the `acceptation_history_length` hyperparameter.
    """
    def __init__(
        self,
        name: str,
        shape: tuple,
        *,
        scale: Union[float, torch.FloatTensor],
        random_order_dimension: bool = True,
        mean_acceptation_rate_target_bounds: Tuple[float, float] = (0.2, 0.4),
        adaptive_std_factor: float = 0.1,
        **base_sampler_kws,
    ):
        super().__init__(
            name,
            shape,
            scale=scale,
            random_order_dimension=random_order_dimension,
            mean_acceptation_rate_target_bounds=mean_acceptation_rate_target_bounds,
            adaptive_std_factor=adaptive_std_factor,
            **base_sampler_kws
        )
        # adapted-variance of mask elements are meaningless (not taken into account for aggregated stats)
        if self.mask is not None:
            self.std[~self.mask] = 0

    @property
    def _should_mask_changes(self) -> bool:
        """
        Nota: for full Gibbs, strictly speaking we never need to applying any masking on proposed changes
        (since we already forced std=0 on masked elements if any; it would not hurt but would be slightly less efficient).
        """
        return False

    @property
    def shape_adapted_std(self) -> tuple:
        return self.shape

    @property
    def _meaningful_indices(self) -> tuple:
        if self.mask is not None:
            return (self.mask,)
        return ()

    def _get_iterator_indices(self) -> IteratorIndicesType:
        """
        Example for variable of shape (2,3) with mask = [[1,1,0],[1,1,1]]
            --> iterator_indices = [(0,0), (0, 1), (1, 0), (1, 1), (1, 2)]

        Returns
        -------
        IteratorIndicesType :
            The indices for the iterator.
        """
        if self.mask is not None:
            return list(map(tuple, self.mask.nonzero(as_tuple=False).tolist()))
        return super()._get_iterator_indices()


class PopulationFastGibbsSampler(AbstractPopulationGibbsSampler):
    """
    Fast Gibbs sampler for population variables.

    .. note::
        The sampling batches along the dimensions except the first one.
        This speeds up sampling process for 2 dimensional parameters.

    Parameters
    ----------
    name : :obj:`str`
        The name of the random variable to sample.
    shape : :obj:`tuple` of :obj:`int`
        The shape of the random variable to sample.
    scale : :obj:`float` > 0 or :class:`torch.FloatTensor` > 0
        An approximate scale for the variable.
        It will be used to scale the initial adaptive std-dev used in sampler.
        An extra 1% factor will be applied on top of this scale (STD_SCALE_FACTOR)
        Note that if you pass a torch tensor, its shape should be compatible with shape of the variable.
    random_order_dimension : :obj:`bool` (default True)
        This parameter controls whether we randomize the order of indices during the sampling loop.
        Article https://proceedings.neurips.cc/paper/2016/hash/e4da3b7fbbce2345d7772b0674a318d5-Abstract.html
        gives a rationale on why we should activate this flag.
    mean_acceptation_rate_target_bounds : :obj:`tuple`[lower_bound: float, upper_bound: float] with 0 < lower_bound < upper_bound < 1
        Bounds on mean acceptation rate.
        Outside this range, the adaptation of the std-dev of sampler is triggered
        so to maintain a target acceptation rate in between these too bounds (e.g: ~30%).
    adaptive_std_factor : :obj:`float` in ]0, 1[
        Factor by which we increase or decrease the std-dev of sampler when we are out of
        the custom bounds for the mean acceptation rate. We decrease it by `1 - factor` if too low,
        and increase it with `1 + factor` if too high.
    **base_sampler_kws
        Keyword arguments passed to `AbstractSampler` init method.
        In particular, you may pass the `acceptation_history_length` hyperparameter.
    """

    @property
    def shape_adapted_std(self) -> tuple:
        return (self.shape[0], )


class PopulationMetropolisHastingsSampler(AbstractPopulationGibbsSampler):
    """
    Metropolis-Hastings sampler for population variables.

    .. note::
        The sampling is done for all values at once.
        This speeds up considerably sampling but usually requires more iterations.

    Parameters
    ----------
    name : :obj:`str`
        The name of the random variable to sample.
    shape : :obj:`tuple` of :obj:`int`
        The shape of the random variable to sample.
    scale : :obj:`float` > 0 or :class:`torch.FloatTensor` > 0
        An approximate scale for the variable.
        It will be used to scale the initial adaptive std-dev used in sampler.
        An extra 1% factor will be applied on top of this scale (STD_SCALE_FACTOR)
        Note that if you pass a torch tensor, its shape should be compatible with shape of the variable.
    random_order_dimension : :obj:`bool` (default True)
        This parameter controls whether we randomize the order of indices during the sampling loop.
        Article https://proceedings.neurips.cc/paper/2016/hash/e4da3b7fbbce2345d7772b0674a318d5-Abstract.html
        gives a rationale on why we should activate this flag.
    mean_acceptation_rate_target_bounds : :obj:`tuple`[lower_bound: float, upper_bound: float] with 0 < lower_bound < upper_bound < 1
        Bounds on mean acceptation rate.
        Outside this range, the adaptation of the std-dev of sampler is triggered
        so to maintain a target acceptation rate in between these too bounds (e.g: ~30%).
    adaptive_std_factor : :obj:`float` in ]0, 1[
        Factor by which we increase or decrease the std-dev of sampler when we are out of
        the custom bounds for the mean acceptation rate. We decrease it by `1 - factor` if too low,
        and increase it with `1 + factor` if too high.
    **base_sampler_kws
        Keyword arguments passed to `AbstractSampler` init method.
        In particular, you may pass the `acceptation_history_length` hyperparameter.
    """

    @property
    def shape_adapted_std(self) -> tuple:
        return ()


class IndividualGibbsSampler(GibbsSamplerMixin, AbstractIndividualSampler):
    """
    Gibbs sampler for individual variables.

    Individual variables are handled with a grouped Gibbs sampler.
    There is currently no other sampler available for individual variables.

    Parameters
    ----------
    name : :obj:`str`
        The name of the random variable to sample.
    shape : :obj:`tuple` of :obj:`int`
        The shape of the random variable to sample.
    n_patients : :obj:`int`
        Number of patients.
    scale : :obj:`float` > 0 or :class:`torch.FloatTensor` > 0
        An approximate scale for the variable.
        It will be used to scale the initial adaptive std-dev used in sampler.
        An extra 1% factor will be applied on top of this scale (STD_SCALE_FACTOR)
        Note that if you pass a torch tensor, its shape should be compatible with shape of the variable.
    random_order_dimension : :obj:`bool` (default True)
        This parameter controls whether we randomize the order of indices during the sampling loop.
        Article https://proceedings.neurips.cc/paper/2016/hash/e4da3b7fbbce2345d7772b0674a318d5-Abstract.html
        gives a rationale on why we should activate this flag.
    mean_acceptation_rate_target_bounds : :obj:`tuple`[lower_bound: float, upper_bound: float] with 0 < lower_bound < upper_bound < 1
        Bounds on mean acceptation rate.
        Outside this range, the adaptation of the std-dev of sampler is triggered
        so to maintain a target acceptation rate in between these too bounds (e.g: ~30%).
    adaptive_std_factor : :obj:`float` in ]0, 1[
        Factor by which we increase or decrease the std-dev of sampler when we are out of
        the custom bounds for the mean acceptation rate. We decrease it by `1 - factor` if too low,
        and increase it with `1 + factor` if too high.
    **base_sampler_kws
        Keyword arguments passed to `AbstractSampler` init method.
        In particular, you may pass the `acceptation_history_length` hyperparameter.
    """

    STD_SCALE_FACTOR = 0.5

    def __init__(
        self,
        name: str,
        shape: tuple,
        *,
        n_patients: int,
        scale: Union[float, torch.FloatTensor],
        random_order_dimension: bool = True,
        mean_acceptation_rate_target_bounds: Tuple[float, float] = (0.2, 0.4),
        adaptive_std_factor: float = 0.1,
        **base_sampler_kws,
    ):
        super().__init__(
            name,
            shape,
            n_patients=n_patients,
            scale=scale,
            random_order_dimension=random_order_dimension,
            mean_acceptation_rate_target_bounds=mean_acceptation_rate_target_bounds,
            adaptive_std_factor=adaptive_std_factor,
            **base_sampler_kws
        )

    def validate_scale(self, scale: Union[float, torch.Tensor]) -> torch.Tensor:
        scale = super().validate_scale(scale)
        # <!> scale should always be a scalar tensor for individual sampler
        return scale.mean()

    @property
    def shape_adapted_std(self) -> tuple:
        # <!> We do not take into account the dimensionality of
        # individual parameter for acceptation / adaptation
        return (self.n_patients,)

    def _proposed_change(self) -> torch.Tensor:
        """
        Proposal change around the current value with sampler standard deviation.

        .. warning::
            Not to be used for scalar sampling (in population samplers)
            since it would be inefficient!

        Parameters
        ----------
        val : torch.Tensor

        Returns
        -------
        :class:`torch.Tensor` :
            Tensor of shape (self.n_patients, *self.shape).
        """
        std_broadcasting = (slice(None),) + (None,) * self.ndim
        return self.std[std_broadcasting] * torch.randn((self.n_patients, *self.shape))

    def sample(
        self,
        state: State,
        *,
        temperature_inv: float,
    ) -> None:
        """
        For each individual variable, compute current patient-batched attachment and regularity.

        Propose a new value for the individual variable,
        and compute new patient-batched attachment and regularity.

        Do a MH step, keeping if better, or if worse with a probability.

        Notes
        -----
        The population variables are fixed during this sampling step (since we update an individual parameter), but:
        - if we are in a calibration: we may have updated them just before and have NOT yet propagated these changes
          into the master model parameters, so we SHOULD use the MCMC state for model computations (default)
        - if we are in a personalization (mode/mean real): we are not updating the population parameters any more
          so we should NOT use a MCMC state any more (not defined, only the "fitted model" state is defined)

        Parameters
        ----------
        state : :class:`.State`
            Instance holding values for all model variables (including latent variables), as well as:
            - timepoints : :class:`torch.Tensor` of shape (n_individuals, n_timepoints)
            - dataset : ...
                Contains the data of the subjects, in particular the subjects'
                time-points and the mask for nan values & padded visits
        temperature_inv : :obj:`float` > 0
            The temperature to use.
        """

        def compute_attachment_regularity():
            # compute neg log-likelihood of just the given variable (tau, xi or sources)
            # (per subject; all dimensions of the individual parameter are summed together)
            return state["nll_attach_ind"], state[f"nll_regul_{self.name}_ind"]

        previous_attachment, previous_regularity = compute_attachment_regularity()
        # with state.auto_fork():
        state.put(
            self.name,
            self._proposed_change(),
            accumulate=True,  # out-of-place addition
        )

        # alpha is per individual and > 0, shape = (n_individuals,)
        # if new is "better" than previous, then alpha > 1 so it will always be accepted in `_group_metropolis_step`
        new_attachment, new_regularity = compute_attachment_regularity()
        alpha = torch.exp(
            -1 * (
                (new_regularity - previous_regularity) * temperature_inv +
                (new_attachment - previous_attachment)
            )
        )
        accepted = self._group_metropolis_step(alpha)

        # Partial reversion where proposals were not accepted
        state.revert(~accepted)

        self._update_acceptation_rate(accepted.float())
        self._update_std()