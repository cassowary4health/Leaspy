import warnings

import torch

from leaspy.models.abstract_model import AbstractModel
from leaspy.models.obs_models import FullGaussianObs
# WIP
# from leaspy.models.utils.initialization.model_initialization import initialize_parameters
# from leaspy.models.utils.ordinal import OrdinalModelMixin
from leaspy.io.data.dataset import Dataset
from leaspy.variables.specs import (
    NamedVariables,
    ModelParameter,
    Hyperparameter,
    PopulationLatentVariable,
    IndividualLatentVariable,
    LinkedVariable,
)
from leaspy.variables.distributions import Normal
from leaspy.utils.functional import (
    Exp,
    MatMul,
)

from leaspy.utils.typing import KwargsType
from leaspy.utils.docs import doc_with_super
from leaspy.exceptions import LeaspyModelInputError


@doc_with_super()
class AbstractMultivariateModel(AbstractModel):  # OrdinalModelMixin,
    """
    Contains the common attributes & methods of the multivariate models.

    Parameters
    ----------
    name : :obj:`str`
        Name of the model.
    **kwargs
        Hyperparameters for the model (including `noise_model`).

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        If inconsistent hyperparameters.
    """
    def __init__(self, name: str, **kwargs):

        self.source_dimension: int = None

        # TODO / WIP / TMP: dirty for now...
        # Should we:
        # - use factory of observation models instead? dataset -> ObservationModel
        # - or refact a bit `ObservationModel` structure? (lazy init of its variables...)
        # (cf. note in AbstractModel as well)
        dimension = kwargs.get('dimension', None)
        if 'features' in kwargs:
            dimension = len(kwargs['features'])

        obs_model = kwargs.get("obs_models", None)
        if isinstance(obs_model, str):
            if obs_model == "gaussian-diagonal":
                assert dimension is not None, "WIP: dimension / features should be provided to init the obs_model = 'gaussian-diagonal'"
                kwargs["obs_models"] = FullGaussianObs.with_noise_std_as_model_parameter(dimension)
            elif obs_model == "gaussian-scalar":
                kwargs["obs_models"] = FullGaussianObs.with_noise_std_as_model_parameter(1)
            else:
                raise NotImplementedError("WIP...")

        if dimension is not None:
            kwargs.setdefault("obs_models", FullGaussianObs.with_noise_std_as_model_parameter(dimension))
        # END TMP

        super().__init__(name, **kwargs)


    def get_variables_specs(self) -> NamedVariables:
        """Return the specifications of the variables (latent variables, derived variables, model 'parameters') that are part of the model."""
        d = super().get_variables_specs()

        d.update(
            # PRIORS
            log_g_mean=ModelParameter.for_pop_mean("log_g", shape=(self.dimension,)),
            log_g_std=Hyperparameter(0.01),

            tau_mean=ModelParameter.for_ind_mean("tau", shape=(1,)),
            tau_std=ModelParameter.for_ind_std("tau", shape=(1,)),
            #xi_mean=Hyperparameter(0.),  # depends on model sub-type (parallel or not)
            xi_std=ModelParameter.for_ind_std("xi", shape=(1,)),

            # LATENT VARS
            log_g=PopulationLatentVariable(Normal("log_g_mean", "log_g_std")),
            xi=IndividualLatentVariable(Normal("xi_mean", "xi_std")),
            tau=IndividualLatentVariable(Normal("tau_mean", "tau_std")),

            # DERIVED VARS
            g=LinkedVariable(Exp("log_g")),
            alpha=LinkedVariable(Exp("xi")),
            # rt=LinkedVariable(self.time_reparametrization),  # in super class...
        )

        if self.source_dimension >= 1:
            d.update(
                # PRIORS
                betas_mean=ModelParameter.for_pop_mean("betas", shape=(self.dimension - 1, self.source_dimension)),
                betas_std=Hyperparameter(0.01),
                sources_mean=Hyperparameter(torch.zeros((self.source_dimension,))),
                sources_std=Hyperparameter(1.),
                # LATENT VARS
                betas=PopulationLatentVariable(
                    Normal("betas_mean", "betas_std"),
                    sampling_kws={"scale": .5},   # cf. GibbsSampler (for retro-compat)
                ),
                sources=IndividualLatentVariable(Normal("sources_mean", "sources_std")),
                # DERIVED VARS
                mixing_matrix=LinkedVariable(MatMul("orthonormal_basis", "betas").then(torch.t)),  # shape: (Ns, Nfts)
                space_shifts=LinkedVariable(MatMul("sources", "mixing_matrix")),                   # shape: (Ni, Nfts)
            )

        return d

    def initialize(self, dataset: Dataset, method: str = 'default') -> None:
        """
        Overloads base initialization of model (base method takes care of features consistency checks).

        Parameters
        ----------
        dataset : :class:`.Dataset`
            Input :class:`.Dataset` from which to initialize the model.
        method : :obj:`str`, optional
            The initialization method to be used.
            Default='default'.
        """

        # WIP: a bit dirty this way...
        # TODO? split method in two so that it would overwritting of method would be cleaner?
        if self.source_dimension is None:
            self.source_dimension = int(dataset.dimension ** .5)
            warnings.warn('You did not provide `source_dimension` hyperparameter for multivariate model, '
                          f'setting it to ⌊√dimension⌋ = {self.source_dimension}.')

        elif not (isinstance(self.source_dimension, int) and 0 <= self.source_dimension < dataset.dimension):
            raise LeaspyModelInputError(f"Sources dimension should be an integer in [0, dimension - 1[ "
                                        f"but you provided `source_dimension` = {self.source_dimension} whereas `dimension` = {dataset.dimension}")

        super().initialize(dataset, method=method)


    #def load_parameters(self, parameters: KwargsType) -> None:
    #    """
    #    Updates all model parameters from the provided parameters.
    #
    #    Parameters
    #    ----------
    #    parameters : KwargsType
    #        The parameters to be loaded.
    #    """
    #    self.parameters = {}
    #    for k, v in parameters.items():
    #        if k in ('mixing_matrix',):
    #            # The mixing matrix will always be recomputed from `betas`
    #            # and the other needed model parameters (g, v0)
    #            continue
    #        if not isinstance(v, torch.Tensor):
    #            v = torch.tensor(v)
    #        self.parameters[k] = v
    #
    #    self._check_ordinal_parameters_consistency()

    def load_hyperparameters(self, hyperparameters: KwargsType) -> None:
        """
        Updates all model hyperparameters from the provided hyperparameters.

        Parameters
        ----------
        hyperparameters : KwargsType
            The hyperparameters to be loaded.
        """
        expected_hyperparameters = ('features', 'dimension', 'source_dimension')

        if 'features' in hyperparameters:
            self.features = hyperparameters['features']

        if 'dimension' in hyperparameters:
            if self.features and hyperparameters['dimension'] != len(self.features):
                raise LeaspyModelInputError(
                    f"Dimension provided ({hyperparameters['dimension']}) does not match "
                    f"features ({len(self.features)})"
                )
            self.dimension = hyperparameters['dimension']

        if 'source_dimension' in hyperparameters:
            if not (
                isinstance(hyperparameters['source_dimension'], int)
                and (hyperparameters['source_dimension'] >= 0)
                and (self.dimension is None or hyperparameters['source_dimension'] <= self.dimension - 1)
            ):
                raise LeaspyModelInputError(
                    f"Source dimension should be an integer in [0, dimension - 1], "
                    f"not {hyperparameters['source_dimension']}"
                )
            self.source_dimension = hyperparameters['source_dimension']

        # WIP
        ## special hyperparameter(s) for ordinal model
        #expected_hyperparameters += self._handle_ordinal_hyperparameters(hyperparameters)

        self._raise_if_unknown_hyperparameters(expected_hyperparameters, hyperparameters)

    def to_dict(self, *, with_mixing_matrix: bool = True) -> KwargsType:
        """
        Export ``Leaspy`` object as dictionary ready for :term:`JSON` saving.

        Parameters
        ----------
        with_mixing_matrix : :obj:`bool` (default ``True``)
            Save the :term:`mixing matrix` in the exported file in its 'parameters' section.

            .. warning::
                It is not a real parameter and its value will be overwritten at model loading
                (orthonormal basis is recomputed from other "true" parameters and mixing matrix
                is then deduced from this orthonormal basis and the betas)!
                It was integrated historically because it is used for convenience in
                browser webtool and only there...

        Returns
        -------
        KwargsType :
            The object as a dictionary.
        """
        model_settings = super().to_dict()
        model_settings['source_dimension'] = self.source_dimension

        if with_mixing_matrix and self.source_dimension >= 1:
            # transposed compared to previous version
            model_settings['parameters']['mixing_matrix'] = self.state['mixing_matrix'].tolist()

        # self._export_extra_ordinal_settings(model_settings)

        return model_settings
