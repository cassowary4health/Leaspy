import torch
from src.algo.abstract_mcmc import AbstractMCMC
import os
from src.inputs.algo_settings import AlgoSettings
from src import default_algo_dir
from src.utils.sampler import Sampler
import numpy as np




class MCMCSAEM(AbstractMCMC):

    def __init__(self):
        super().__init__()