import torch
from src.algo.abstract_mcmc import AbstractMCMC


class TensorMCMCSAEM(AbstractMCMC):

    def __init__(self, settings):
        super().__init__(settings)
        self.name = "MCMC_SAEM (tensor)"
