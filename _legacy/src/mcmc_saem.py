import torch
from leaspy.algo.abstract_mcmc import AbstractMCMC
import os
from leaspy.inputs.algorithm_settings import AlgorithmSettings
from leaspy import default_algo_dir
from leaspy.utils.sampler import Sampler
import numpy as np




class MCMCSAEM(AbstractMCMC):

    def __init__(self):
        super().__init__()