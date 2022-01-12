import itertools

import torch

from .abstract_sampler import AbstractSampler
from leaspy.exceptions import LeaspyInputError
from leaspy.utils.docs import doc_with_super
from leaspy.utils.typing import Optional, Union


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
        It will be used to scale the initial adaptative std-dev used in sampler.
        An extra factor will be applied on top of this scale (hyperparameters):
            * 1% for population parameters (:attr:`.GibbsSampler.STD_SCALE_FACTOR_POP`)
            * 50% for individual parameters (:attr:`.GibbsSampler.STD_SCALE_FACTOR_IND`)
        Note that if you pass a torch tensor, its shape should be compatible with shape of the variable.

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

    def __init__(self, info: dict, n_patients: int, *, scale: Union[float, torch.FloatTensor]):

        super().__init__(info, n_patients)

        # Scale of variable should always be positive (component-wise if multidimensional)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float32)
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

        # Acceptation rate
        self._counter_acceptation: int = 0

        # Torch distribution: all modifications will be in-place on `self.std`
        # So there will be no need to update this distribution!
        self._distribution = torch.distributions.normal.Normal(loc=0.0, scale=self.std)

        # Previous computations to speed-up population variable sampling
        self._previous_attachment: Optional[torch.FloatTensor] = None
        self._previous_regularity: Optional[torch.FloatTensor] = None

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

        Adaptative std is known to improve sampling performances.
        Std is increased if frequency of acceptation > 40%, and decreased if <20%,
        so as to stay close to 30%.
        """
        self._counter_acceptation += 1

        if self._counter_acceptation == self.temp_length:
            mean_acceptation = self.acceptation_temp.mean(dim=0)

            idx_toolow = mean_acceptation < 0.2
            idx_toohigh = mean_acceptation > 0.4

            self.std[idx_toolow] *= 0.9
            self.std[idx_toohigh] *= 1.1

            # reset acceptation temp list
            self._counter_acceptation = 0

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
        """
        realization = realizations[self.name]
        shape_current_variable = realization.shape
        index = [e for e in itertools.product(*[range(s) for s in shape_current_variable])]

        accepted_array = []

        # retrieve the individual parameters from realizations once for all to speed-up computations,
        # since they are fixed during the sampling of this population variable!
        ind_params = model.get_param_from_real(realizations)

        def compute_attachment_regularity():
            # model attributes used are the ones from the MCMC toolbox that we are currently changing!
            attachment = model.compute_individual_attachment_tensorized(data, ind_params, attribute_type='MCMC').sum()
            # regularity is always computed with model.parameters (not "temporary MCMC parameters")
            regularity = model.compute_regularity_realization(realization).sum()
            return attachment, regularity

        # TODO: shouldn't we loop randomly here so there is no order in dimensions?
        for idx in index:
            # Compute the attachment and regularity
            if self._previous_attachment is None:
                assert self._previous_regularity is None
                self._previous_attachment, self._previous_regularity = compute_attachment_regularity()

            # Keep previous realizations and sample new ones
            old_val_idx = realization.tensor_realizations[idx].clone()
            # the previous version with `_proposal` was not incorrect but computationnally inefficient:
            # because we were sampling on the full shape of `std` whereas we only needed `std[idx]` (scalar)
            new_val_idx = old_val_idx + self.std[idx] * torch.randn(())
            realization.set_tensor_realizations_element(new_val_idx, idx)

            # Update derived model attributes if necessary (orthonormal basis, ...)
            model.update_MCMC_toolbox([self.name], realizations)

            # Compute the attachment and regularity
            new_attachment, new_regularity = compute_attachment_regularity()

            alpha = torch.exp(-((new_regularity - self._previous_regularity) * temperature_inv +
                                (new_attachment - self._previous_attachment)))

            accepted = self._metropolis_step(alpha)
            accepted_array.append(accepted)

            if not accepted:
                # Revert modification of realization at idx and its consequences
                realization.set_tensor_realizations_element(old_val_idx, idx)
                # Update (back) derived model attributes if necessary
                # TODO: Shouldn't we backup the old MCMC toolbox instead to avoid heavy computations?
                # (e.g. orthonormal basis re-computation just for a single change)
                model.update_MCMC_toolbox([self.name], realizations)
                # force re-compute on next iteration:
                # not performed since it is useless, since we rolled back to the starting state!
                # self._previous_attachment = self._previous_regularity = None
            else:
                self._previous_attachment, self._previous_regularity = new_attachment, new_regularity

        accepted_array = torch.tensor(accepted_array, dtype=torch.float32).reshape(shape_current_variable)
        self._update_acceptation_rate(accepted_array)
        self._update_std()

        # Reset previous attachment and regularity for next sampling loop!
        self._previous_attachment = self._previous_regularity = None

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
            Optional keyword arguments for attachement computations.
            As of now, we only use it for individual variables, and only `attribute_type`.
            It is used to know whether to compute attachments from the MCMC toolbox (esp. during fit)
            or to compute it from regular model parameters (esp. during personalization in mean/mode realization)
        """

        # Compute the attachment and regularity for all subjects
        realization = realizations[self.name]

        # the population variables during this sampling step (since we update an individual parameter), but:
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

        # we accept together all dimensions of individual parameter
        accepted = accepted.unsqueeze(-1)  # shape = (n_individuals, 1)

        realization.tensor_realizations = accepted*realization.tensor_realizations + (1-accepted)*previous_reals
