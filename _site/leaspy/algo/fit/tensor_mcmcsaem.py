from leaspy.algo.fit.abstract_mcmc import AbstractFitMCMC


class TensorMCMCSAEM(AbstractFitMCMC):

    def __init__(self, settings):
        super().__init__(settings)
        self.name = "MCMC_SAEM (tensor)"
