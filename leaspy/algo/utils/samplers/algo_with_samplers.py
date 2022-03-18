from typing import Dict
import warnings

from .abstract_sampler import AbstractSampler
from .gibbs_sampler import GibbsSampler
#from .hmc_sampler import HMCSampler  # legacy


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
                raise ValueError("You should NOT have both `n_burn_in_iter_frac` and `n_burn_in_iter` None."
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

    def __str__(self):
        out = super().__str__()
        # TODO? separate mixin for algorithms with nb of iterations & burn-in phase?
        out += f"\nIteration {self.current_iteration} / {self.algo_parameters['n_iter']}"
        if self._is_burn_in():
            out += " (memory-less phase)"
        out += "\n= Samplers ="
        for sampler in self.samplers.values():
            out += f"\n    {str(sampler)}"
        return out

    def _initialize_samplers(self, model, dataset):
        """
        Instantiate samplers as a dictionary samplers {variable_name: sampler}

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
        dataset : :class:`.Dataset`
        """
        # fetch additional hyperparameters for samplers
        # TODO: per variable and not just per type of variable?
        sampler_ind = self.algo_parameters.get('sampler_ind', None)
        sampler_ind_kws = self.algo_parameters.get('sampler_ind_params', {})
        sampler_pop = self.algo_parameters.get('sampler_pop', None)
        sampler_pop_kws = self.algo_parameters.get('sampler_pop_params', {})

        # allow sampler ind or pop to be None when the corresponding variables are not needed
        # e.g. for personalization algorithms (mode & mean real), we do not need to sample pop variables any more!
        if sampler_ind not in [None, 'Gibbs']:
            raise NotImplementedError("Only 'Gibbs' sampler is supported for individual variables for now, "
                                      "please open an issue on Gitlab if needed.")

        if sampler_pop not in [None, 'Gibbs']:
            raise NotImplementedError("Only 'Gibbs' sampler is supported for population variables for now, "
                                      "please open an issue on Gitlab if needed.")

        self.samplers = {}
        for variable, info in model.random_variable_informations().items():

            if info["type"] == "individual":

                # To enforce a fixed scale for a given var, one should put it in the random var specs
                # But note that for individual parameters the model parameters ***_std should always be OK (> 0)
                scale_param = info.get('scale', model.parameters[f'{variable}_std'])

                if sampler_ind == 'Gibbs':
                    self.samplers[variable] = GibbsSampler(info, dataset.n_individuals, scale=scale_param, **sampler_ind_kws)
                #elif sampler_ind == 'HMC':  # legacy
                    #self.samplers[variable] = HMCSampler(info, data.n_individuals, self.algo_parameters['eps'])
            else:

                # To enforce a fixed scale for a given var, one should put it in the random var specs
                # For instance: for betas & deltas, it is a good idea to define them this way
                # since they'll probably be = 0 just after initialization!
                scale_param = info.get('scale', model.parameters[variable].abs())

                if sampler_pop == 'Gibbs':
                    self.samplers[variable] = GibbsSampler(info, dataset.n_individuals, scale=scale_param, **sampler_pop_kws)
                #elif sampler_pop == 'HMC':  # legacy
                    #self.samplers[variable] = HMCSampler(info, data.n_individuals, self.algo_parameters['eps'])
