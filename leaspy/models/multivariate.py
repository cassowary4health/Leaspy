import torch

from typing import Iterable, Optional
from leaspy.models.abstract_multivariate_model import AbstractMultivariateModel
from leaspy.io.data.dataset import Dataset
from leaspy.utils.weighted_tensor import WeightedTensor, TensorOrWeightedTensor
from leaspy.utils.docs import doc_with_super  # doc_with_
# from leaspy.utils.subtypes import suffixed_method
from leaspy.exceptions import LeaspyModelInputError

from leaspy.variables.state import State
from leaspy.variables.specs import (
    NamedVariables,
    ModelParameter,
    PopulationLatentVariable,
    LinkedVariable,
    Hyperparameter,
    SuffStatsRW,
    VariablesValuesRO,
)
from leaspy.variables.distributions import Normal
from leaspy.utils.functional import Exp, Sqr, OrthoBasis
from leaspy.utils.weighted_tensor import unsqueeze_right


# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either
# model values or model values + jacobians wrt individual parameters

# TODO refact? subclass or other proper code technique to extract model's concrete
#  formulation depending on if linear, logistic, mixed log-lin, ...


@doc_with_super()
class MultivariateModel(AbstractMultivariateModel):
    """
    Manifold model for multiple variables of interest (logistic or linear formulation).

    Parameters
    ----------
    name : :obj:`str`
        The name of the model.
    **kwargs
        Hyperparameters of the model (including `noise_model`)

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        * If `name` is not one of allowed sub-type: 'univariate_linear' or 'univariate_logistic'
        * If hyperparameters are inconsistent
    """

    SUBTYPES_SUFFIXES = {
        'linear': '_linear',
        'logistic': '_logistic',
        'mixed_linear-logistic': '_mixed',
    }

    def __init__(self, name: str, variables_to_track: Optional[Iterable[str]] = None, **kwargs):
        super().__init__(name, **kwargs)

        # TODO: remove this, use children classes instead (more proper)
        self._subtype_suffix = self._check_subtype()

        variables_to_track = variables_to_track or (
            "log_g_mean",
            "log_v0_mean",
            "noise_std",
            "tau_mean",
            "tau_std",
            "xi_mean",
            "xi_std",
            "nll_attach",
            "nll_regul_log_g",
            "nll_regul_log_v0",
        )
        self.tracked_variables = self.tracked_variables.union(set(variables_to_track))

    def _check_subtype(self) -> str:
        if self.name not in self.SUBTYPES_SUFFIXES.keys():
            raise LeaspyModelInputError(
                f"{type(self).__name__} name should be among these valid sub-types: "
                f"{list(self.SUBTYPES_SUFFIXES.keys())}."
            )

        return self.SUBTYPES_SUFFIXES[self.name]

    """
    @suffixed_method
    def compute_individual_tensorized(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> torch.Tensor:
        pass

    def compute_individual_tensorized_linear(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> torch.Tensor:
        # Population parameters
        positions, velocities, mixing_matrix = self._get_attributes(attribute_type)
        xi, tau = individual_parameters['xi'], individual_parameters['tau']

        # TODO: use rt instead
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1)  # for automatic broadcast on n_features (last dim)

        # Model expected value
        model = positions + velocities * reparametrized_time

        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(mixing_matrix.t())
            model += wi.unsqueeze(-2)

        return model  # (n_individuals, n_timepoints, n_features)

    def compute_individual_tensorized_logistic(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> torch.Tensor:
        # Population parameters
        g, v0, a_matrix = self._get_attributes(attribute_type)
        g_plus_1 = 1. + g
        b = g_plus_1 * g_plus_1 / g

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1)  # (n_individuals, n_timepoints, n_features)

        if self.is_ordinal:
            # add an extra dimension for the levels of the ordinal item
            reparametrized_time = reparametrized_time.unsqueeze(-1)
            g = g.unsqueeze(-1)
            b = b.unsqueeze(-1)
            v0 = v0.unsqueeze(-1)
            deltas = self._get_deltas(attribute_type)  # (features, max_level)
            deltas = deltas.unsqueeze(0).unsqueeze(0)  # add (ind, timepoints) dimensions
            # infinite deltas (impossible ordinal levels) will induce model = 0 which is intended
            reparametrized_time = reparametrized_time - deltas.cumsum(dim=-1)

        LL = v0 * reparametrized_time

        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(a_matrix.t()).unsqueeze(-2)  # unsqueeze for (n_timepoints)
            if self.is_ordinal:
                wi = wi.unsqueeze(-1)
            LL += wi

        # TODO? more efficient & accurate to compute `torch.exp(-t*b + log_g)` since we
        #  directly sample & stored log_g
        LL = 1. + g * torch.exp(-LL * b)
        model = 1. / LL

        # For ordinal models, compute pdf instead of survival function if needed
        model = self.compute_appropriate_ordinal_model(model)

        return model  # (n_individuals, n_timepoints, n_features [, extra_dim_ordinal_models])

    @suffixed_method
    def compute_individual_ages_from_biomarker_values_tensorized(
        self,
        value: torch.Tensor,
        individual_parameters: dict,
        feature: str,
    ) -> torch.Tensor:
        pass

    def compute_individual_ages_from_biomarker_values_tensorized_logistic(
        self,
        value: torch.Tensor,
        individual_parameters: dict,
        feature: str,
    ) -> torch.Tensor:
        if value.dim() != 2:
            raise LeaspyModelInputError(f"The biomarker value should be dim 2, not {value.dim()}!")

        if self.is_ordinal:
            return self._compute_individual_ages_from_biomarker_values_tensorized_logistic_ordinal(
                value, individual_parameters, feature
            )
        # avoid division by zero:
        value = value.masked_fill((value == 0) | (value == 1), float('nan'))

        # 1/ get attributes
        g, v0, a_matrix = self._get_attributes(None)
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(a_matrix.t())
        else:
            wi = 0

        # get feature value for g, v0 and wi
        feat_ind = self.features.index(feature)  # all consistency checks were done in API layer
        g = torch.tensor([g[feat_ind]])  # g and v0 were shape: (n_features in the multivariate model)
        v0 = torch.tensor([v0[feat_ind]])
        if self.source_dimension != 0:
            wi = wi[0, feat_ind].item()  # wi was shape (1, n_features)

        # 2/ compute age
        ages = tau + (torch.exp(-xi) / v0) * ((g / (g + 1) ** 2) * torch.log(g/(1 / value - 1)) - wi)
        # assert ages.shape == value.shape

        return ages

    def _compute_individual_ages_from_biomarker_values_tensorized_logistic_ordinal(
        self,
        value: torch.Tensor,
        individual_parameters: dict,
        feature: str,
    ) -> torch.Tensor:
        ""
        For one individual, compute age(s) breakpoints at which the given features
        levels are the most likely (given the subject's individual parameters).

        Consistency checks are done in the main :term:`API` layer.

        Parameters
        ----------
        value : :class:`torch.Tensor`
            Contains the :term:`biomarker` level value(s) of the subject.

        individual_parameters : :obj:`dict`
            Contains the individual parameters.
            Each individual parameter should be a scalar or array_like.

        feature : :obj:`str`
            Name of the considered :term:`biomarker`

            .. note::
                Optional for :class:`.UnivariateModel`, compulsory
                for :class:`.MultivariateModel`.

        Returns
        -------
        :class:`torch.Tensor`
            Contains the subject's ages computed at the given values(s)
            Shape of tensor is ``(1, n_values)``.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            if computation is tried on more than 1 individual
        ""
        # 1/ get attributes
        g, v0, a_matrix = self._get_attributes(None)
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(a_matrix.t())
        else:
            wi = 0

        # get feature value for g, v0 and wi
        feat_ind = self.features.index(feature)  # all consistency checks were done in API layer
        g = torch.tensor([g[feat_ind]])  # g and v0 were shape: (n_features in the multivariate model)
        v0 = torch.tensor([v0[feat_ind]])
        if self.source_dimension != 0:
            wi = wi[0, feat_ind].item()  # wi was shape (1, n_features)

        # 2/ compute age
        ages_0 = tau + (torch.exp(-xi) / v0) * ((g / (g + 1) ** 2) * torch.log(g) - wi)
        deltas_ft = self._get_deltas(None)[feat_ind]
        delta_max = deltas_ft[torch.isfinite(deltas_ft)].sum()
        ages_max = tau + (torch.exp(-xi) / v0) * ((g / (g + 1) ** 2) * torch.log(g) - wi + delta_max)

        grid_timepoints = torch.linspace(ages_0.item(), ages_max.item(), 1000)

        return self._ordinal_grid_search_value(
            grid_timepoints,
            value,
            individual_parameters=individual_parameters,
            feat_index=feat_ind,
        )

    @suffixed_method
    def compute_jacobian_tensorized(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> DictParamsTorch:
        pass

    def compute_jacobian_tensorized_linear(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> DictParamsTorch:
        # Population parameters
        _, v0, mixing_matrix = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1)  # (n_individuals, n_timepoints, n_features)

        alpha = torch.exp(xi).reshape(-1, 1, 1)
        dummy_to_broadcast_n_ind_n_tpts = torch.ones_like(reparametrized_time)

        # Jacobian of model expected value w.r.t. individual parameters
        derivatives = {
            'xi': (v0 * reparametrized_time).unsqueeze(-1),  # add a last dimension for len param
            'tau': (v0 * -alpha * dummy_to_broadcast_n_ind_n_tpts).unsqueeze(-1),  # same
        }

        if self.source_dimension > 0:
            derivatives['sources'] = (
                mixing_matrix.expand((1, 1, -1, -1)) * dummy_to_broadcast_n_ind_n_tpts.unsqueeze(-1)
            )

        # dict[param_name: str, torch.Tensor of shape(n_ind, n_tpts, n_fts, n_dims_param)]
        return derivatives

    def compute_jacobian_tensorized_logistic(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> DictParamsTorch:
        # TODO: refact highly inefficient (many duplicated code from `compute_individual_tensorized_logistic`)

        # Population parameters
        g, v0, a_matrix = self._get_attributes(attribute_type)
        g_plus_1 = 1. + g
        b = g_plus_1 * g_plus_1 / g

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1)  # (n_individuals, n_timepoints, n_features)
        alpha = torch.exp(xi).reshape(-1, 1, 1)

        if self.is_ordinal:
            # add an extra dimension for the levels of the ordinal item
            LL = reparametrized_time.unsqueeze(-1)
            g = g.unsqueeze(-1)
            b = b.unsqueeze(-1)
            deltas = self._get_deltas(attribute_type)  # (features, max_level)
            deltas = deltas.unsqueeze(0).unsqueeze(0)  # add (ind, timepoints) dimensions
            LL = LL - deltas.cumsum(dim=-1)
            LL = v0.unsqueeze(-1) * LL

        else:
            LL = v0 * reparametrized_time

        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(a_matrix.t()).unsqueeze(-2)  # unsqueeze for (n_timepoints)
            if self.is_ordinal:
                wi = wi.unsqueeze(-1)
            LL += wi
        LL = 1. + g * torch.exp(-LL * b)
        model = 1. / LL

        # Jacobian of model expected value w.r.t. individual parameters
        c = model * (1. - model) * b

        derivatives = {
            'xi': (v0 * reparametrized_time).unsqueeze(-1),
            'tau': (-v0 * alpha).unsqueeze(-1),
        }
        if self.source_dimension > 0:
            derivatives['sources'] = a_matrix.expand((1, 1, -1, -1))

        if self.is_ordinal:
            ordinal_lvls_shape = c.shape[-1]
            for param in derivatives:
                derivatives[param] = derivatives[param].unsqueeze(-2).repeat(1, 1, 1, ordinal_lvls_shape, 1)

        # Multiply by common constant, and post-process derivative for ordinal models if needed
        for param in derivatives:
            derivatives[param] = self.compute_appropriate_ordinal_model(
                c.unsqueeze(-1) * derivatives[param]
            )

        # dict[param_name: str, torch.Tensor of shape(n_ind, n_tpts, n_fts [, extra_dim_ordinal_models], n_dims_param)]
        return derivatives
    """

    ##############################
    ### MCMC-related functions ###
    ##############################

    @classmethod
    def _center_xi_realizations(cls, state: State) -> None:
        """
        Center the ``xi`` realizations in place.

        .. note::
            This operation does not change the orthonormal basis
            (since the resulting ``v0`` is collinear to the previous one)
            Nor all model computations (only ``v0 * exp(xi_i)`` matters),
            it is only intended for model identifiability / ``xi_i`` regularization
            <!> all operations are performed in "log" space (``v0`` is log'ed)

        Parameters
        ----------
        realizations : :class:`.CollectionRealization`
            The realizations to use for updating the :term:`MCMC` toolbox.
        """
        mean_xi = torch.mean(state['xi'])
        state["xi"] = state["xi"] - mean_xi
        state["log_v0"] = state["log_v0"] + mean_xi

        # TODO: find a way to prevent re-computation of orthonormal basis since it should not have changed (v0_collinear update)
        #self.update_MCMC_toolbox({'v0_collinear'}, realizations)

    @classmethod
    def compute_sufficient_statistics(
        cls,
        state: State,
    ) -> SuffStatsRW:
        """
        Compute the model's :term:`sufficient statistics`.

        Parameters
        ----------
        state : :class:`.State`
            The state to pick values from.

        Returns
        -------
        SuffStatsRW :
            The computed sufficient statistics.
        """
        # <!> modify 'xi' and 'log_v0' realizations in-place
        # TODO: what theoretical guarantees for this custom operation?
        cls._center_xi_realizations(state)

        return super().compute_sufficient_statistics(state)

    @classmethod
    def model_with_sources(
        cls,
        *,
        rt: TensorOrWeightedTensor[float],
        space_shifts: TensorOrWeightedTensor[float],
        metric: TensorOrWeightedTensor[float],
        v0: TensorOrWeightedTensor[float],
        log_g: TensorOrWeightedTensor[float],
    ) -> torch.Tensor:
        """Returns a model with sources."""
        # TODO WIP: logistic model only for now
        # Shape: (Ni, Nt, Nfts)
        pop_s = (None, None, ...)
        rt = unsqueeze_right(rt, ndim=1)  # .filled(float('nan'))
        w_model_logit = metric[pop_s] * (v0[pop_s] * rt + space_shifts[:, None, ...]) - log_g[pop_s]
        model_logit, weights = WeightedTensor.get_filled_value_and_weight(w_model_logit, fill_value=0.)
        return WeightedTensor(torch.sigmoid(model_logit), weights).weighted_value

    @classmethod
    def model_no_sources(
        cls,
        *,
        rt: TensorOrWeightedTensor[float],
        metric: TensorOrWeightedTensor[float],
        v0: TensorOrWeightedTensor[float],
        log_g: TensorOrWeightedTensor[float],
    ) -> torch.Tensor:
        """Returns a model without source. A bit dirty?"""
        return cls.model_with_sources(
            rt=rt,
            metric=metric,
            v0=v0,
            log_g=log_g,
            space_shifts=torch.zeros((1, 1)),
        )

    @staticmethod
    def metric(*, g: TensorOrWeightedTensor[float]) -> TensorOrWeightedTensor[float]:
        """Used to define the corresponding variable."""
        return (g + 1) ** 2 / g

    def get_variables_specs(self) -> NamedVariables:
        """
        Return the specifications of the variables (latent variables, derived variables,
        model 'parameters') that are part of the model.

        Returns
        -------
        NamedVariables :
            The specifications of the model's variables.
        """
        d = super().get_variables_specs()

        #if self._subtype_suffix != "_logistic":
        #    raise NotImplementedError("WIP: Only implemented for logistic models.")

        d.update(
            # PRIORS
            log_v0_mean=ModelParameter.for_pop_mean(
                "log_v0",
                shape=(self.dimension,),
            ),
            log_v0_std=Hyperparameter(0.01),
            xi_mean=Hyperparameter(0.),
            # LATENT VARS
            log_v0=PopulationLatentVariable(
                Normal("log_v0_mean", "log_v0_std"),
            ),
            # DERIVED VARS
            v0=LinkedVariable(
                Exp("log_v0"),
            ),
            metric=LinkedVariable(self.metric),  # for linear model: metric & metric_sqr are fixed = 1.
        )

        if self.source_dimension >= 1:
            d.update(
                model=LinkedVariable(self.model_with_sources),
                metric_sqr=LinkedVariable(Sqr("metric")),
                orthonormal_basis=LinkedVariable(OrthoBasis("v0", "metric_sqr")),
            )
        else:
            d['model'] = LinkedVariable(self.model_no_sources)

        # TODO: WIP
        #variables_info.update(self.get_additional_ordinal_population_random_variable_information())
        #self.update_ordinal_population_random_variable_information(variables_info)

        return d

    def get_initial_model_parameters(self, dataset: Dataset, method: str) -> VariablesValuesRO:
        """Get initial values for model parameters."""

        # TODO - WIP (put back `initialize_parameters` function in this one, with possible utils functions in models.utilities?)
        from leaspy.models.utils.initialization.model_initialization import initialize_parameters
        params, obs_model_params = initialize_parameters(self, dataset, method=method)
        # all parameters in one
        params.update(obs_model_params)

        return params

"""
# document some methods (we cannot decorate them at method creation since they are
# not yet decorated from `doc_with_super`)
doc_with_(MultivariateModel.compute_individual_tensorized_linear,
          MultivariateModel.compute_individual_tensorized,
          mapping={'the model': 'the model (linear)'})
doc_with_(MultivariateModel.compute_individual_tensorized_logistic,
          MultivariateModel.compute_individual_tensorized,
          mapping={'the model': 'the model (logistic)'})

# doc_with_(MultivariateModel.compute_individual_tensorized_mixed,
#          MultivariateModel.compute_individual_tensorized,
#          mapping={'the model': 'the model (mixed logistic-linear)'})

doc_with_(MultivariateModel.compute_jacobian_tensorized_linear,
          MultivariateModel.compute_jacobian_tensorized,
          mapping={'the model': 'the model (linear)'})
doc_with_(MultivariateModel.compute_jacobian_tensorized_logistic,
          MultivariateModel.compute_jacobian_tensorized,
          mapping={'the model': 'the model (logistic)'})
# doc_with_(MultivariateModel.compute_jacobian_tensorized_mixed,
#          MultivariateModel.compute_jacobian_tensorized,
#          mapping={'the model': 'the model (mixed logistic-linear)'})

# doc_with_(MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized_linear,
#          MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized,
#          mapping={'the model': 'the model (linear)'})
doc_with_(MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized_logistic,
          MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized,
          mapping={'the model': 'the model (logistic)'})
# doc_with_(MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized_mixed,
#          MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized,
#          mapping={'the model': 'the model (mixed logistic-linear)'})
"""
