import torch
from lifelines import WeibullFitter

from leaspy.models.univariate import UnivariateModel
from leaspy.io.data.dataset import Dataset
from leaspy.exceptions import LeaspyModelInputError

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
from leaspy.utils.typing import (

    DictParams,
)
import pandas as pd
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
    init_tolerance = 0.3
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        obs_models_to_string = [o.to_string() for o in self.obs_models]
        if "gaussian-scalar" not in obs_models_to_string:
            self.obs_models += (observation_model_factory("gaussian-scalar", dimension = 1),)
        if "weibull-right-censored" not in obs_models_to_string:
            self.obs_models += (observation_model_factory("weibull-right-censored", nu = 'nu', rho = 'rho', xi = 'xi', tau = 'tau'),)

        variables_to_track = (
            "n_log_nu_mean",
            "log_rho_mean",
            "nll_attach_y",
            "nll_attach_event",
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
            nll_attach=LinkedVariable(Sum("nll_attach_y", "nll_attach_event")),
            nll_attach_ind = LinkedVariable(Sum("nll_attach_y_ind", "nll_attach_event_ind")),
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

    def initialize_individual_parameters(self, state, dataset):

        df = dataset.to_pandas().reset_index('TIME').groupby('ID').min()

        # Initialise individual parameters if they are not already initialised
        if not state.are_variables_set(('xi', 'tau')):
            df_ind = df["TIME"].to_frame(name='tau')
            df_ind['xi'] = 0.
        else:
            df_ind = pd.DataFrame(torch.concat([state['xi'], state['tau']], axis=1),
                columns=['xi', 'tau'], index=df.index)

        # Set the right initialisation point fpr barrier methods
        df_inter = pd.concat([df["EVENT_TIME"] - self.init_tolerance, df_ind['tau']], axis=1)
        df_ind['tau'] = df_inter.min(axis=1)

        with state.auto_fork(None):
            state.put_individual_latent_variables(df = df_ind)
        return state

    ##############################
    ###      Estimation        ###
    ##############################


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
        timepoints : scalar or array_like[scalar] (:obj:`list`, :obj:`tuple`, :class:`numpy.ndarray`)
            Contains the age(s) of the subject.
        individual_parameters : :obj:`dict`
            Contains the individual parameters.
            Each individual parameter should be a scalar or array_like.
        skip_ips_checks : :obj:`bool` (default: ``False``)
            Flag to skip consistency/compatibility checks and tensorization
            of ``individual_parameters`` when it was done earlier (speed-up).

        Returns
        -------
        :class:`torch.Tensor`
            Contains the subject's scores computed at the given age(s)
            Shape of tensor is ``(1, n_tpts, n_features)``.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If computation is tried on more than 1 individual.
        :exc:`.LeaspyIndividualParamsInputError`
            if invalid individual parameters.
        """
        self.check_individual_parameters_provided(individual_parameters.keys())
        timepoints, individual_parameters = self._get_tensorized_inputs(
            timepoints, individual_parameters, skip_ips_checks=skip_ips_checks
        )

        # TODO? ability to revert back after **several** assignments?
        # instead of cloning the state for this op?
        local_state = self.state.clone(disable_auto_fork=True)

        self._put_data_timepoints(local_state, timepoints)
        local_state.put('event', (timepoints, torch.zeros(timepoints.shape).bool()))

        for ip, ip_v in individual_parameters.items():
            local_state[ip] = ip_v
        # reshape survival_event from (len(timepoints)) to (1, len(timepoints), 1) so it is compatible with the
        # model shape 
        return torch.cat((local_state["model"],
                   torch.exp(local_state["survival_event"]).reshape(-1,1).expand((1,-1,-1))),2)