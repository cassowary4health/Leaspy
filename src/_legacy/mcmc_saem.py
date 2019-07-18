import torch
from src.algo.abstract_mcmc import AbstractMCMC
import os
from src.inputs.algorithm_settings import AlgorithmSettings
from src.utils.sampler import Sampler
import numpy as np




class MCMCSAEM(AbstractMCMC):

    def __init__(self):
        super().__init__()