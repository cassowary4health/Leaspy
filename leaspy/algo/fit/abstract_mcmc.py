import torch

from .abstract_fit_algo import AbstractFitAlgo
from ..samplers.gibbs_sampler import GibbsSampler
from ..samplers.hmc_sampler import HMCSampler

from leaspy.models.utils import DEFAULT_LOSS
from leaspy.exceptions import LeaspyAlgoInputError


class AbstractFitMCMC(AbstractFitAlgo):
    """
    Abstract class containing common method for all `fit` algorithm classes based on `Monte-Carlo Markov Chains` (MCMC).

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        MCMC fit algorithm settings

    Attributes
    ----------
    samplers : dict[ str, :class:`~.algo.samplers.abstract_sampler.AbstractSampler` ]
        Dictionary of samplers per each variable
    TODO add missing

    See Also
    --------
    :mod:`leaspy.algo.samplers`
    """

    def __init__(self, settings):

        super().__init__()

        # Algorithm parameters
        self.algo_parameters = settings.parameters
        self.seed = settings.seed
        self.loss = settings.loss  # to be removed?!

        # Realizations and samplers
        self.realizations = None
        #self.task = None
        self.samplers = None
        self.current_iteration = 0

        # Annealing
        self.temperature_inv = 1
        self.temperature = 1

        # Select the device on which computation will be perfomed
        # (defaults to the CPU in the AlgorithmSettings method)
        self.device = torch.device(settings.device)

    ###########################
    ## Initialization
    ###########################

    def _initialize_algo(self, data, model, realizations):
        """
        Initialize the samplers, annealing, MCMC toolbox and sufficient statistics.

        Parameters
        ----------
        data : :class:`.Dataset`
        model : :class:`~.models.abstract_model.AbstractModel`
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`

        Raises
        ------
        :exc:`.LeaspyAlgoInputError`
            If inconsistent parameter between model & algo
        """

        # TODO: change this behavior: loss should only come from algo!
        # handling loss, a bit dirty...
        if model.loss != DEFAULT_LOSS: # non default loss from model
            if self.loss not in [DEFAULT_LOSS, model.loss]:
                raise LeaspyAlgoInputError(f"You provided inconsistent loss: '{model.loss}' for model and '{self.loss}' for algo.")
            # set algo loss to the one from model
            self.loss = model.loss
        else:
            # set model loss from algo
            model.loss = self.loss

        # MCMC toolbox (cache variables for speed-ups + tricks)
        model.initialize_MCMC_toolbox()

        # Samplers
        self._initialize_samplers(model, data)
        self._initialize_sufficient_statistics(data, model, realizations)
        if self.algo_parameters['annealing']['do_annealing']:
            self._initialize_annealing()
        return realizations

    def _initialize_annealing(self):
        """
        Initialize annealing, setting initial temperature and number of iterations.
        """
        if self.algo_parameters['annealing']['do_annealing']:
            if self.algo_parameters['annealing']['n_iter'] is None:
                self.algo_parameters['annealing']['n_iter'] = int(self.algo_parameters['n_iter'] / 2)

        self.temperature = self.algo_parameters['annealing']['initial_temperature']
        self.temperature_inv = 1 / self.temperature

    def _initialize_samplers(self, model, data):
        """
        Instanciate samplers for Gibbs / HMC sampling as a dictionnary samplers {variable_name: sampler}

        Parameters
        ----------
        data : :class:`.Dataset`
        model : :class:`~.models.abstract_model.AbstractModel`
        """
        infos_variables = model.random_variable_informations()
        self.samplers = dict.fromkeys(infos_variables.keys())
        for variable, info in infos_variables.items():
            if info["type"] == "individual":
                if self.algo_parameters['sampler_ind'] == 'Gibbs':
                    self.samplers[variable] = GibbsSampler(info, data.n_individuals, device=self.device)
                else:
                    self.samplers[variable] = HMCSampler(info, data.n_individuals, self.algo_parameters['eps'])
            else:
                if self.algo_parameters['sampler_pop'] == 'Gibbs':
                    self.samplers[variable] = GibbsSampler(info, data.n_individuals, device=self.device)
                else:
                    self.samplers[variable] = HMCSampler(info, data.n_individuals, self.algo_parameters['eps'])

    def _initialize_sufficient_statistics(self, data, model, realizations):
        """
        Initialize the sufficient statistics.

        Parameters
        ----------
        data : :class:`.Dataset`
        model : :class:`~.models.abstract_model.AbstractModel`
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
        """
        suff_stats = model.compute_sufficient_statistics(data, realizations)
        self.sufficient_statistics = {k: torch.zeros(v.shape, dtype=torch.float32, device=self.device) for k, v in suff_stats.items()}

    ###########################
    ## Getters / Setters
    ###########################

    ###########################
    ## Core
    ###########################

    def iteration(self, data, model, realizations):
        """
        MCMC-SAEM iteration.

        1. Sample : MC sample successively of the populatin and individual variales
        2. Maximization step : update model parameters from current population/individual variables values.

        Parameters
        ----------
        data : :class:`.Dataset`
        model : :class:`~.models.abstract_model.AbstractModel`
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
        """

        # Sample step
        for key in realizations.reals_pop_variable_names:
            self.samplers[key].sample(data, model, realizations, self.temperature_inv)
        for key in realizations.reals_ind_variable_names:
            self.samplers[key].sample(data, model, realizations, self.temperature_inv)

        # Maximization step
        self._maximization_step(data, model, realizations)
        model.update_MCMC_toolbox(['all'], realizations)

        # Update the likelihood with the new noise_var
        # TODO likelihood is computed 2 times, remove this one, and update it in maximization step ?
        # TODO or ar the update of all sufficient statistics ???
        # self.likelihood.update_likelihood(data, model, realizations)

        # Annealing
        if self.algo_parameters['annealing']['do_annealing']:
            self._update_temperature()

    def _update_temperature(self):
        """
        Update the temperature according to a plateau annealing scheme.
        """
        if self.current_iteration <= self.algo_parameters['annealing']['n_iter']:
            # If we cross a plateau step
            if self.current_iteration % int(
                    self.algo_parameters['annealing']['n_iter'] / self.algo_parameters['annealing'][
                        'n_plateau']) == 0:
                # Decrease temperature linearly
                self.temperature -= self.algo_parameters['annealing']['initial_temperature'] / \
                                    self.algo_parameters['annealing']['n_plateau']
                self.temperature = max(self.temperature, 1)
                self.temperature_inv = 1 / self.temperature

    ###########################
    ## Output
    ###########################

    def __str__(self):
        out = ""
        out += "=== ALGO ===\n"
        out += f"Instance of {self.name} algo\n"
        out += f"Iteration {self.current_iteration}\n"
        out += "=Samplers\n"
        for sampler_name, sampler in self.samplers.items():
            acceptation_rate = torch.mean(sampler.acceptation_temp.detach()).item()
            out += f"    {sampler_name} rate : {acceptation_rate:.2%}, std: {sampler.std.mean():.5f}\n"

        if self.algo_parameters['annealing']['do_annealing']:
            out += "Annealing\n"
            out += f"Temperature : {self.temperature}"
        return out

    #############
    ## HMC
    #############
