import torch
from lifelines import WeibullFitter

from leaspy.models.univariate import UnivariateModel
from leaspy.io.data.dataset import Dataset

from leaspy.utils.docs import doc_with_super  # doc_with_
# from leaspy.utils.subtypes import suffixed_method
from leaspy.exceptions import LeaspyInputError, LeaspyModelInputError
from leaspy.utils.weighted_tensor import WeightedTensor, TensorOrWeightedTensor
from leaspy.models.utils.initialization.model_initialization import (
    compute_patient_slopes_distribution,
    compute_patient_values_distribution,
    compute_patient_time_distribution,
    get_log_velocities,
    _torch_round)

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
from leaspy.utils.functional import Exp, Sqr, OrthoBasis, Sum
from leaspy.utils.weighted_tensor import unsqueeze_right

from leaspy.models.obs_models import observation_model_factory


# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either
# model values or model values + jacobians wrt individual parameters

# TODO refact? subclass or other proper code technique to extract model's concrete
#  formulation depending on if linear, logistic, mixed log-lin, ...


@doc_with_super()
class UnivariateJointModel(UnivariateModel):
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
        'univariate_joint': '_joint',
    }

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.obs_models += (observation_model_factory('weibull-right-censored', nu = 'nu', rho = 'rho', xi = 'xi', tau = 'tau'),)
        variables_to_track = (
            "n_log_nu_mean",
            "log_rho_mean",
        )
        self.tracked_variables = self.tracked_variables.union(set(variables_to_track))

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
            nu = LinkedVariable(self.exp_neg_n_log_nu),
            rho=LinkedVariable(
                Exp("log_rho"),
            ),
        )

        d.update(
            nll_attach_xi_ind=LinkedVariable(Sum("nll_attach_y_ind", "nll_attach_event_ind")),
            nll_attach_tau_ind=LinkedVariable(Sum("nll_attach_y_ind")),
            nll_attach=LinkedVariable(Sum("nll_attach_y", "nll_attach_event")),
        )

        return d

    @staticmethod
    def exp_neg_n_log_nu(
            *,
            n_log_nu: torch.Tensor,  # TODO: TensorOrWeightedTensor?
    ) -> torch.Tensor:
        return torch.exp(-1 * n_log_nu)

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

    ##############################
    ###      Initialisation    ###
    ##############################

    def _estimate_initial_longitudinal_parameters(self, dataset: Dataset) -> VariablesValuesRO:

        # Hardcoded
        XI_STD = .5
        TAU_STD = 5.
        NOISE_STD = .1

        # Get dataframe
        df = dataset.to_pandas().dropna(how='all').sort_index()[dataset.headers]

        # Make basic data checks
        assert df.index.is_unique
        assert df.index.to_frame().notnull().all(axis=None)
        if self.features != df.columns:
            raise LeaspyInputError(f"Features mismatch between model and dataset: {model.features} != {df.columns}")

        # Get mean time
        t0, _ = compute_patient_time_distribution(df)

        # Enforce values are between 0 and 1
        values, _ = compute_patient_values_distribution(df)
        values = values.clamp(min=1e-2, max=1 - 1e-2)  # always "works" for ordinal (values >= 1)

        # Extract lopes and convert into parameters
        slopes, _ = compute_patient_slopes_distribution(df)
        v0_array = get_log_velocities(slopes, self.features)
        g_array = torch.log(
            1. / values - 1.)  # cf. Igor thesis; <!> exp is done in Attributes class for logistic models

        # Create smart initialization dictionary
        parameters = {
                'log_g_mean': g_array.squeeze(),
                'log_v0_mean': v0_array.squeeze(),
                'tau_mean': t0,
                'tau_std': torch.tensor(TAU_STD),
                'xi_std': torch.tensor(XI_STD),
            'noise_std' : torch.tensor(NOISE_STD)
            }
        return parameters

    def _estimate_initial_event_parameters(self, dataset: Dataset) -> VariablesValuesRO:
        wbf = WeibullFitter().fit(dataset.event_time,
                                  dataset.event_bool)
        parameters = {
            'log_rho_mean': torch.log(torch.tensor(wbf.rho_)),
            'n_log_nu_mean': -torch.log(torch.tensor(wbf.lambda_))
        }
        return parameters

    def get_initial_model_parameters(self, dataset: Dataset, method: str) -> VariablesValuesRO:
        """Get initial values for model parameters."""

        # Estimate initial parameters from the data
        params = self._estimate_initial_longitudinal_parameters(dataset)
        params_event = self._estimate_initial_event_parameters(dataset)
        params.update(params_event)

        # convert to float 32 bits & add a rounding step on the initial parameters to ensure full reproducibility
        rounded_parameters = {
            str(p): _torch_round(v.to(torch.float32))
            for p, v in params.items()
        }

        return rounded_parameters
