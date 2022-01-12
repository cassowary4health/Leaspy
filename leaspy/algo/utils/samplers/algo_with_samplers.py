from typing import Dict

from .abstract_sampler import AbstractSampler
from .gibbs_sampler import GibbsSampler
#from .hmc_sampler import HMCSampler  # legacy


class AlgoWithSamplersMixin:
    """
    Mixin to use in algorithms needing `samplers`; inherit from this class first.

    Note that this mixin is to be used with a class inheriting from `AbstractAlgo`
    (and in particular that have a `algo_parameters` attribute )

    Parameters
    ----------
    *args
        Positional arguments to pass to super __init__ function
    **kwargs
        Keyword arguments to pass to super __init__ function

    Attributes
    ----------
    samplers : dict[ str, :class:`~.algo.samplers.abstract_sampler.AbstractSampler` ]
        Dictionary of samplers per each variable
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments to next base classes init method
        self.samplers: Dict[str, AbstractSampler] = None

    def _initialize_samplers(self, model, data):
        """
        Instantiate samplers as a dictionnary samplers {variable_name: sampler}

        Parameters
        ----------
        data : :class:`.Dataset`
        model : :class:`~.models.abstract_model.AbstractModel`
        """
        self.samplers = {}
        import warnings
        for variable, info in model.random_variable_informations().items():

            if info["type"] == "individual":

                # To enforce a fixed scale for a given var, one should put it in the random var specs
                # But note that for individual parameters the model parameters ***_std should always be OK (> 0)
                scale_param = info.get('scale', model.parameters[f'{variable}_std'])

                if self.algo_parameters['sampler_ind'] == 'Gibbs':
                    self.samplers[variable] = GibbsSampler(info, data.n_individuals, scale=scale_param)
                #elif self.algo_parameters['sampler_ind'] == 'HMC':  # legacy
                    #self.samplers[variable] = HMCSampler(info, data.n_individuals, self.algo_parameters['eps'])
                else:
                    raise NotImplementedError('Only "Gibbs" sampler is supported for now, please open an issue on Gitlab if needed.')
            else:

                # To enforce a fixed scale for a given var, one should put it in the random var specs
                # For instance: for betas & deltas, it is a good idea to define them this way
                # since they'll probably be = 0 just after initialization!
                scale_param = info.get('scale', model.parameters[variable].abs())

                if self.algo_parameters['sampler_pop'] == 'Gibbs':
                    self.samplers[variable] = GibbsSampler(info, data.n_individuals, scale=scale_param)
                #elif self.algo_parameters['sampler_pop'] == 'HMC':  # legacy
                    #self.samplers[variable] = HMCSampler(info, data.n_individuals, self.algo_parameters['eps'])
                else:
                    raise NotImplementedError('Only "Gibbs" sampler is supported for now, please open an issue on Gitlab if needed.')
