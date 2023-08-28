import torch

from leaspy.models.abstract_multivariate_model import AbstractMultivariateModel
from leaspy.io.data.dataset import Dataset

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
class OrdinalModel(AbstractMultivariateModel):
    """
    Ordinal model based on logistics.

    Parameters
    ----------
    name : :obj:`str`
        The name of the model.
    **kwargs
        Hyperparameters of the model (including `noise_model`)

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        * If `name` is not one of allowed sub-type: 'ordinal'
        * If hyperparameters are inconsistent
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

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
    def model_with_sources(cls, *, rt: torch.Tensor, space_shifts: torch.Tensor, metric, v0, log_g) -> torch.Tensor:
        """Returns a model with sources."""
        # TODO WIP: logistic model only for now
        # Shape: (Ni, Nt, Nfts)
        pop_s = (None, None, ...)
        rt = unsqueeze_right(rt, ndim=1)  # .filled(float('nan'))
        model_logit = metric[pop_s] * (v0[pop_s] * rt + space_shifts[:, None, ...]) - log_g[pop_s]
        return torch.sigmoid(model_logit)

    @classmethod
    def model_no_sources(cls, *, rt: torch.Tensor, metric, v0, log_g) -> torch.Tensor:
        """Returns a model without source. A bit dirty?"""
        return cls.model_with_sources(
            rt=rt,
            metric=metric,
            v0=v0,
            log_g=log_g,
            space_shifts=torch.zeros((1, 1)),
        )

    @staticmethod
    def metric(*, g: torch.Tensor) -> torch.Tensor:
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

        if self._subtype_suffix != "_logistic":
            raise NotImplementedError("WIP: Only implemented for logistic models.")

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
