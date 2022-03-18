from random import shuffle
from collections.abc import Sequence

import torch
from numpy import ndindex

from .abstract_sampler import AbstractSampler
from leaspy.exceptions import LeaspyInputError
from leaspy.utils.docs import doc_with_super
from leaspy.utils.typing import Union, Tuple


@doc_with_super()
class GibbsSampler(AbstractSampler):
    """
    Gibbs sampler class.

    Parameters
    ----------
    info : dict[str, Any]
        The dictionary describing the random variable to sample.
        It should contains the following entries:
            * name : str
            * shape : tuple[int, ...]
            * type : 'population' or 'individual'
    n_patients : int > 0
        Number of patients (useful for individual variables)
    scale : float > 0 or :class:`torch.FloatTensor` > 0
        An approximate scale for the variable.
        It will be used to scale the initial adaptive std-dev used in sampler.
        An extra factor will be applied on top of this scale (hyperparameters):
            * 1% for population parameters (:attr:`.GibbsSampler.STD_SCALE_FACTOR_POP`)
            * 50% for individual parameters (:attr:`.GibbsSampler.STD_SCALE_FACTOR_IND`)
        Note that if you pass a torch tensor, its shape should be compatible with shape of the variable.
    random_order_dimension : bool (default True)
        This parameter controls whether we randomize the order of indices during the sampling loop.
        (only for population variables, since we perform group sampling for individual variables)
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

    Attributes
    ----------
    In addition to the attributes present in :class:`.AbstractSampler`:

    std : torch.FloatTensor
        Adaptative std-dev of variable

    Raises
    ------
    :exc:`.LeaspyInputError`
    """

    # Cf. note on `scale` parameter above (heuristic values)
    STD_SCALE_FACTOR_POP = .01
    STD_SCALE_FACTOR_IND = .5

    def __init__(self, info: dict, n_patients: int, *, scale: Union[float, torch.FloatTensor],
                 random_order_dimension: bool = True,
                 mean_acceptation_rate_target_bounds: Tuple[float, float] = (0.2, 0.4),
                 adaptive_std_factor: float = 0.1,
                 **base_sampler_kws):

        super().__init__(info, n_patients, **base_sampler_kws)

        # Scale of variable should always be positive (component-wise if multidimensional)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
        scale = scale.float()
        if (scale <= 0).any():
            raise LeaspyInputError(f"Scale of variable '{info['name']}' should be positive, not `{scale}`.")

        if info["type"] == "population":
            # Proposition variance is adapted independently on each dimension of the population variable
            self.std = self.STD_SCALE_FACTOR_POP * scale * torch.ones(self.shape)
        elif info["type"] == "individual":
            # Proposition variance is adapted independently on each patient
            true_shape = (n_patients, *self.shape)
            self.std = self.STD_SCALE_FACTOR_IND * scale * torch.ones(true_shape)
        else:
            raise LeaspyInputError(f"Unknown variable type '{info['type']}'.")

        # Internal counter to trigger adaptation of std based on mean acceptation rate
        self._counter: int = 0

        # Torch distribution: all modifications will be in-place on `self.std`
        # So there will be no need to update this distribution!
        self._distribution = torch.distributions.normal.Normal(loc=0.0, scale=self.std)

        # Parameters of the sampler
        self._random_order_dimension = random_order_dimension

        if not (
            isinstance(mean_acceptation_rate_target_bounds, Sequence)
            and len(mean_acceptation_rate_target_bounds) == 2
            and 0 < mean_acceptation_rate_target_bounds[0] < mean_acceptation_rate_target_bounds[1] < 1
        ):
            raise LeaspyInputError("`mean_acceptation_rate_target_bounds` should be a tuple (lower_bound, upper_bound) "
                                   f"with 0 < lower_bound < upper_bound < 1, not '{mean_acceptation_rate_target_bounds}'")
        self._mean_acceptation_lower_bound_before_adaptation, self._mean_acceptation_upper_bound_before_adaptation = mean_acceptation_rate_target_bounds

        if not (0 < adaptive_std_factor < 1):
            raise LeaspyInputError(f"`adaptive_std_factor` should be a float in ]0, 1[, not '{adaptive_std_factor}'")
        self._adaptive_std_factor = adaptive_std_factor

    def __str__(self):
        mean_acceptation_rate = self.acceptation_history.mean().item()  # mean on all dimensions!
        return f"{self.name} rate : {mean_acceptation_rate:.1%}, std: {self.std.mean():.1e}"

    def _proposal(self, val):
        """
        Proposal value around the current value with sampler standard deviation.

        <!> Not to be used for scalar sampling (in `_sample_population_realizations`)
            since it would be inefficient!

        Parameters
        ----------
        val : torch.FloatTensor

        Returns
        -------
        torch.FloatTensor of shape broadcasted_shape(val.shape, self.std.shape)
            value around `val`
        """
        return val + self._distribution.sample()  # sample_shape=val.shape

    def _update_std(self):
        """
        Update standard deviation of sampler according to current frequency of acceptation.

        Adaptive std is known to improve sampling performances.
        For default parameters: std-dev is increased if frequency of acceptation is > 40%,
        and decreased if < 20%, so as to stay close to 30%.
        """
        self._counter += 1

        if self._counter % self.acceptation_history_length == 0:
            mean_acceptation = self.acceptation_history.mean(dim=0)

            idx_toolow = mean_acceptation < self._mean_acceptation_lower_bound_before_adaptation
            idx_toohigh = mean_acceptation > self._mean_acceptation_upper_bound_before_adaptation

            self.std[idx_toolow] *= (1 - self._adaptive_std_factor)
            self.std[idx_toohigh] *= (1 + self._adaptive_std_factor)

    def _sample_population_realizations(self, data, model, realizations, temperature_inv, **attachment_computation_kws):
        """
        For each dimension (1D or 2D) of the population variable, compute current attachment and regularity.
        Propose a new value for the given dimension of the given population variable,
        and compute new attachment and regularity.
        Do a MH step, keeping if better, or if worse with a probability.

        Parameters
        ----------
        data : :class:`.Dataset`
        model : :class:`~.models.abstract_model.AbstractModel`
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
        temperature_inv : float > 0
        **attachment_computation_kws
            Currently not used for population parameters.

        Returns
        -------
        attachment, regularity_var : `torch.FloatTensor` 0D (scalars)
            The attachment and regularity (only for the current variable) at the end of this sampling step (summed on all individuals).
        """
        realization = realizations[self.name]
        shape_current_variable = realization.shape
        accepted_array = torch.zeros(shape_current_variable)
        iterator_indices = list(ndindex(shape_current_variable))
        if self._random_order_dimension:
            shuffle(iterator_indices)  # shuffle in-place!

        # retrieve the individual parameters from realizations once for all to speed-up computations,
        # since they are fixed during the sampling of this population variable!
        ind_params = model.get_param_from_real(realizations)

        def compute_attachment_regularity():
            # model attributes used are the ones from the MCMC toolbox that we are currently changing!
            attachment = model.compute_individual_attachment_tensorized(data, ind_params, attribute_type='MCMC').sum()
            # regularity is always computed with model.parameters (not "temporary MCMC parameters")
            regularity = model.compute_regularity_realization(realization).sum()
            return attachment, regularity

        previous_attachment = previous_regularity = None

        for idx in iterator_indices:
            # Compute the attachment and regularity
            if previous_attachment is None:
                previous_attachment, previous_regularity = compute_attachment_regularity()

            # Keep previous realizations and sample new ones
            old_val_idx = realization.tensor_realizations[idx].clone()
            # the previous version with `_proposal` was not incorrect but computationally inefficient:
            # because we were sampling on the full shape of `std` whereas we only needed `std[idx]` (scalar)
            new_val_idx = old_val_idx + self.std[idx] * torch.randn(())
            realization.set_tensor_realizations_element(new_val_idx, idx)

            # Update derived model attributes if necessary (orthonormal basis, ...)
            model.update_MCMC_toolbox([self.name], realizations)

            # Compute the attachment and regularity
            new_attachment, new_regularity = compute_attachment_regularity()

            alpha = torch.exp(-((new_regularity - previous_regularity) * temperature_inv +
                                (new_attachment - previous_attachment)))

            accepted = self._metropolis_step(alpha)
            accepted_array[idx] = accepted

            if accepted:
                previous_attachment, previous_regularity = new_attachment, new_regularity
            else:
                # Revert modification of realization at idx and its consequences
                realization.set_tensor_realizations_element(old_val_idx, idx)
                # Update (back) derived model attributes if necessary
                # TODO: Shouldn't we backup the old MCMC toolbox instead to avoid heavy computations?
                # (e.g. orthonormal basis re-computation just for a single change)
                model.update_MCMC_toolbox([self.name], realizations)
                # force re-compute on next iteration:
                # not performed since it is useless, since we rolled back to the starting state!
                # previous_attachment = previous_regularity = None

        self._update_acceptation_rate(accepted_array)
        self._update_std()

        # Return last attachment and regularity_var
        return previous_attachment, previous_regularity

    def _sample_individual_realizations(self, data, model, realizations, temperature_inv, **attachment_computation_kws):
        """
        For each individual variable, compute current patient-batched attachment and regularity.
        Propose a new value for the individual variable,
        and compute new patient-batched attachment and regularity.
        Do a MH step, keeping if better, or if worse with a probability.

        Parameters
        ----------
        data : :class:`.Dataset`
        model : :class:`~.models.abstract_model.AbstractModel`
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
        temperature_inv : float > 0
        **attachment_computation_kws
            Optional keyword arguments for attachment computations.
            As of now, we only use it for individual variables, and only `attribute_type`.
            It is used to know whether to compute attachments from the MCMC toolbox (esp. during fit)
            or to compute it from regular model parameters (esp. during personalization in mean/mode realization)

        Returns
        -------
        attachment, regularity_var : `torch.FloatTensor` 1D (n_individuals,)
            The attachment and regularity (only for the current variable) at the end of this sampling step, per individual.
        """

        # Compute the attachment and regularity for all subjects
        realization = realizations[self.name]

        # the population variables are fixed during this sampling step (since we update an individual parameter), but:
        # - if we are in a calibration: we may have updated them just before and have NOT yet propagated these changes
        #   into the master model parameters, so we SHOULD use the MCMC toolbox for model computations (default)
        # - if we are in a personalization (mode/mean real): we are not updating the population parameters any more
        #   so we should NOT use a MCMC_toolbox (not proper)
        attribute_type = attachment_computation_kws.get('attribute_type', 'MCMC')

        def compute_attachment_regularity():
            # current realizations => individual parameters
            ind_params = model.get_param_from_real(realizations)

            # individual parameters => compare reconstructions vs values (per subject)
            attachment = model.compute_individual_attachment_tensorized(data, ind_params, attribute_type=attribute_type)

            # compute log-likelihood of just the given parameter (tau or xi or sources)
            # (per subject; all dimensions of the individual parameter are summed together)
            # regularity is always computed with model.parameters (not "temporary MCMC parameters")
            regularity = model.compute_regularity_realization(realization)
            regularity = regularity.sum(dim=self.ind_param_dims_but_individual).reshape(data.n_individuals)

            return attachment, regularity

        previous_attachment, previous_regularity = compute_attachment_regularity()

        # Keep previous realizations and sample new ones
        previous_reals = realization.tensor_realizations.clone()
        # Add perturbations to previous observations
        realization.tensor_realizations = self._proposal(realization.tensor_realizations)

        # Compute the attachment and regularity
        new_attachment, new_regularity = compute_attachment_regularity()

        # alpha is per patient and > 0, shape = (n_individuals,)
        # if new is "better" than previous, then alpha > 1 so it will always be accepted in `_group_metropolis_step`
        alpha = torch.exp(-((new_regularity - previous_regularity) * temperature_inv +
                            (new_attachment - previous_attachment)))

        accepted = self._group_metropolis_step(alpha)
        self._update_acceptation_rate(accepted)
        self._update_std()

        # compute attachment & regularity
        attachment = accepted*new_attachment + (1-accepted)*previous_attachment
        regularity = accepted*new_regularity + (1-accepted)*previous_regularity

        # we accept together all dimensions of individual parameter
        accepted = accepted.unsqueeze(-1)  # shape = (n_individuals, 1)
        realization.tensor_realizations = accepted*realization.tensor_realizations + (1-accepted)*previous_reals

        return attachment, regularity
