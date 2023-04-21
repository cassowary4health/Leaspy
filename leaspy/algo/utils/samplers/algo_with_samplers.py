from typing import Dict, Optional
import warnings

from .abstract_sampler import AbstractSampler
from .gibbs_sampler import GibbsSampler

from leaspy.exceptions import LeaspyAlgoInputError
from leaspy.models.abstract_model import AbstractModel
from leaspy.io.data.dataset import Dataset


class AlgoWithSamplersMixin:
    """
    Mixin to use in algorithms needing `samplers`; inherit from this class first.

    Note that this mixin is to be used with a class inheriting from `AbstractAlgo`
    (and in particular that have a `algo_parameters` attribute)

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        The specifications of the algorithm as a :class:`.AlgorithmSettings` instance.

        Please note that you can customize the number of memory-less (burn-in) iterations by setting either:
        * `n_burn_in_iter` directly (deprecated but has priority over following setting, not defined by default)
        * `n_burn_in_iter_frac`, such that duration of burn-in phase is a ratio of algorithm `n_iter` (default of 90%)

    Attributes
    ----------
    samplers : dict[ str, :class:`~.algo.samplers.abstract_sampler.AbstractSampler` ]
        Dictionary of samplers per each variable

    current_iteration : int, default 0
        Current iteration of the algorithm.
        The first iteration will be 1 and the last one `n_iter`.

    random_order_variables : bool (default True)
        This attribute controls whether we randomize the order of variables at each iteration.
        Article https://proceedings.neurips.cc/paper/2016/hash/e4da3b7fbbce2345d7772b0674a318d5-Abstract.html
        gives a rationale on why we should activate this flag.
    """

    def __init__(self, settings):
        super().__init__(settings)

        self.samplers: Dict[str, AbstractSampler] = None

        self.random_order_variables = self.algo_parameters.get('random_order_variables', True)

        self.current_iteration: int = 0

        # Dynamic number of iterations for burn-in phase
        n_burn_in_iter_frac = self.algo_parameters['n_burn_in_iter_frac']

        if self.algo_parameters.get('n_burn_in_iter', None) is None:
            if n_burn_in_iter_frac is None:
                raise LeaspyAlgoInputError(
                    "You should NOT have both `n_burn_in_iter_frac` and `n_burn_in_iter` None."
                    "\nPlease set a value for at least one of those settings.")

            self.algo_parameters['n_burn_in_iter'] = int(n_burn_in_iter_frac * self.algo_parameters['n_iter'])

        elif n_burn_in_iter_frac is not None:
            warnings.warn("`n_burn_in_iter` setting is deprecated in favour of `n_burn_in_iter_frac` - "
                          "which defines the duration of the burn-in phase as a ratio of the total number of iterations."
                          "\nPlease use the new setting to suppress this warning "
                          "or explicitly set `n_burn_in_iter_frac=None`."
                          "\nNote that while `n_burn_in_iter` is supported "
                          "it will always have priority over `n_burn_in_iter_frac`.", FutureWarning)

    def _is_burn_in(self) -> bool:
        """
        Check if current iteration is in burn-in (= memory-less) phase.

        Returns
        -------
        bool
        """
        return self.current_iteration <= self.algo_parameters['n_burn_in_iter']

    ###########################
    # Output
    ###########################

    def _get_progress_str(self) -> str:
        # The algorithm must define a progress string (thanks to `self.current_iteration`)
        iter_str = super()._get_progress_str()
        if self._is_burn_in():
            iter_str += " (memory-less phase)"
        else:
            iter_str += " (with memory)"
        return iter_str

    def __str__(self):
        out = super().__str__()
        out += "\n= Samplers ="
        for sampler in self.samplers.values():
            out += f"\n    {str(sampler)}"
        return out

    def _initialize_samplers(self, model: AbstractModel, dataset: Dataset) -> None:
        """
        Instantiate samplers as a dictionary samplers {variable_name: sampler}

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
        dataset : :class:`.Dataset`
        """
        self.samplers = {}
        self._initialize_individual_samplers(model, dataset)
        self._initialize_population_samplers(model, dataset)

    def _initialize_individual_samplers(self, model: AbstractModel, dataset: Dataset) -> None:
        # TODO: per variable and not just per type of variable?
        sampler = self.get_individual_sampler()
        sampler_kws = self.algo_parameters.get('sampler_ind_params', {})
        for variable, info in model.get_individual_random_variable_information().items():
            # To enforce a fixed scale for a given var, one should put it in the random var specs
            # But note that for individual parameters the model parameters ***_std should always be OK (> 0)
            if sampler == "Gibbs":
                self.samplers[variable] = GibbsSampler(
                    info,
                    dataset.n_individuals,
                    scale=info.get('scale', model.parameters[f'{variable}_std']),
                    sampler_type=sampler,
                    **sampler_kws,
                )

    def _initialize_population_samplers(self, model: AbstractModel, dataset: Dataset) -> None:
        sampler = self.get_population_sampler()
        sampler_kws = self.algo_parameters.get('sampler_pop_params', {})
        for variable, info in model.get_population_random_variable_information().items():
            # To enforce a fixed scale for a given var, one should put it in the random var specs
            # For instance: for betas & deltas, it is a good idea to define them this way
            # since they'll probably be = 0 just after initialization!
            # We have priors which should be better than the variable initial value no ?
            # model.MCMC_toolbox['priors'][f'{variable}_std']
            if sampler in ("Gibbs", "FastGibbs", "Metropolis-Hastings"):
                self.samplers[variable] = GibbsSampler(
                    info,
                    dataset.n_individuals,
                    scale=info.get('scale', model.parameters[variable].abs()),
                    sampler_type=sampler,
                    **sampler_kws,
                )

    def get_individual_sampler(self) -> Optional[str]:
        """
        Return a valid individual sampler.

        Allow sampler to be None when the corresponding variables are not needed
        e.g. for personalization algorithms (mode & mean real), we do not need to
        sample pop variables any more!

        Returns
        -------
        sampler : str
            The name of the individual sampler.
        """
        sampler = self.algo_parameters.get('sampler_ind', None)
        if sampler not in (None, "Gibbs"):
            raise NotImplementedError(
                "Only 'Gibbs' sampler is supported for individual variables for now, "
                "please open an issue on Gitlab if needed."
            )
        return sampler

    def get_population_sampler(self) -> Optional[str]:
        """
        Return a valid population sampler.
        """
        sampler = self.algo_parameters.get('sampler_pop', None)
        if sampler not in (None, "Gibbs", "FastGibbs", "Metropolis-Hastings"):
            raise NotImplementedError(
                "Only 'Gibbs', 'FastGibbs' and 'Metropolis-Hastings' samplers are "
                "supported for population variables for now, "
                "please open an issue on Gitlab if needed."
            )
        return sampler
