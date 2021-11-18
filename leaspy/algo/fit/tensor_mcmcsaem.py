from leaspy.algo.fit.abstract_mcmc import AbstractFitMCMC


class TensorMCMCSAEM(AbstractFitMCMC):
    """
    Main algorithm for MCMC-SAEM.

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        MCMC fit algorithm settings

    See Also
    --------
    :class:`.AbstractFitMCMC`
    """

    def __init__(self, settings):
        super().__init__(settings)
        self.name = "MCMC_SAEM (tensor)"
