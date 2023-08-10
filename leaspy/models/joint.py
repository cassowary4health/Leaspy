import torch

from leaspy.models.multivariate import MultivariateModel
from leaspy.io.data.dataset import Dataset

from leaspy.utils.docs import doc_with_super  # doc_with_
# from leaspy.utils.subtypes import suffixed_method
from leaspy.exceptions import LeaspyModelInputError

from leaspy.variables.state import State
from leaspy.variables.specs import (
    DataVariable,
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
class JointModel(MultivariateModel):
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


    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)



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

        d.update(

            # Input data
            event_time = DataVariable(),
            event_bool = DataVariable(),

            # PRIORS
            n_log_nu_mean=ModelParameter.for_pop_mean(
                "n_log_nu",
                shape=(self.dimension,),
            ),
            n_log_nu_std=Hyperparameter(0.01),

            log_rho_mean=ModelParameter.for_pop_mean(
                "log_rho",
                shape=(self.dimension,),
            ),
            log_rho_std=Hyperparameter(0.01),

            # LATENT VARS
            n_log_nu=PopulationLatentVariable(
                Normal("n_log_nu_mean", "n_log_nu_std"),
            ),
            log_rho=PopulationLatentVariable(
                Normal("log_rho_mean", "log_rho_std"),
            ),
            # DERIVED VARS
            nu=LinkedVariable(
                Exp(-"n_log_nu"),
            ),
            rho=LinkedVariable(
                Exp("log_rho"),
            ),
            nu_rep=LinkedVariable(
                self.nu_rep,
            ),
            event_shifted=LinkedVariable(
                self.event_shifted,
            ),
        )

        if self.source_dimension >= 1:
            # TODO: How to handle w in survival process
            pass
        else:
            pass

        # TODO: WIP
        # variables_info.update(self.get_additional_ordinal_population_random_variable_information())
        # self.update_ordinal_population_random_variable_information(variables_info)

        return d

    ##############################
    ### MCMC-related functions ###
    ##############################

    @staticmethod
    def event_shifted(
            *,
            event_time: torch.Tensor,  # TODO: TensorOrWeightedTensor?
            tau: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tensorized time reparametrization formula.

        .. warning::
            Shapes of tensors must be compatible between them.

        Parameters
        ----------
        t : :class:`torch.Tensor`
            Timepoints to reparametrize
        alpha : :class:`torch.Tensor`
            Acceleration factors of individual(s)
        tau : :class:`torch.Tensor`
            Time-shift(s).

        Returns
        -------
        :class:`torch.Tensor` of same shape as `timepoints`
        """
        return (event_time - tau)

    @staticmethod
    def nu_rep(
            *,
            nu: torch.Tensor,  # TODO: TensorOrWeightedTensor?
            alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tensorized time reparametrization formula.

        .. warning::
            Shapes of tensors must be compatible between them.

        Parameters
        ----------
        t : :class:`torch.Tensor`
            Timepoints to reparametrize
        alpha : :class:`torch.Tensor`
            Acceleration factors of individual(s)
        tau : :class:`torch.Tensor`
            Time-shift(s).

        Returns
        -------
        :class:`torch.Tensor` of same shape as `timepoints`
        """
        return alpha/nu

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
        state["n_log_nu"] = state["n_log_nu"] + mean_xi

        # TODO: find a way to prevent re-computation of orthonormal basis since it should not have changed (v0_collinear update)
        #self.update_MCMC_toolbox({'v0_collinear'}, realizations)


    def get_initial_model_parameters(self, dataset: Dataset, method: str) -> VariablesValuesRO:
        """Get initial values for model parameters."""

        # TODO - WIP (put back `initialize_parameters` function in this one, with possible utils functions in models.utilities?)
        from leaspy.models.utils.initialization.model_initialization import initialize_parameters
        params, obs_model_params = initialize_parameters(self, dataset, method=method)
        # all parameters in one
        params.update(obs_model_params)

        return params
