import unittest

import pandas as pd
import torch

from leaspy import AlgorithmSettings, Data, Leaspy

from tests import example_data_path, binary_data_path

class GPUModelFit(unittest.TestCase):

    def test_all_model_gpu_run(self):
        """
        Check if the following models run with the following algorithms, on a GPU device.
        """
        for model_name in ('linear', 'univariate_logistic', 'univariate_linear', 'logistic', 'logistic_parallel'):

            leaspy = Leaspy(model_name)
            settings = AlgorithmSettings('mcmc_saem', n_iter=200, seed=0, device=torch.device("cuda"))

            df = pd.read_csv(example_data_path)
            if model_name == 'univariate_linear' or model_name == 'univariate_logistic':
                df = df.iloc[:, :3]
            data = Data.from_dataframe(df)

            leaspy.fit(data, settings)

            methods = ['mode_real', 'mean_real', 'scipy_minimize']

            for method in methods:
                burn_in_kw = dict() # not for all algos
                if '_real' in method:
                    burn_in_kw = dict(n_burn_in_iter=90, )
                settings = AlgorithmSettings(method, n_iter=100, seed=0, **burn_in_kw)
                result = leaspy.personalize(data, settings)

    def test_all_model_gpu_run_crossentropy(self):
        """
        Check if the following models run with the following algorithms, on a GPU device.
        """
        for model_name in ('linear', 'univariate_logistic', 'univariate_linear', 'logistic', 'logistic_parallel'):
            leaspy = Leaspy(model_name, loss="crossentropy")
            settings = AlgorithmSettings('mcmc_saem', n_iter=200, seed=0, device=torch.device("cuda"))

            df = pd.read_csv(binary_data_path)
            if model_name == 'univariate_linear' or model_name == 'univariate_logistic':
                df = df.iloc[:, :3]
            data = Data.from_dataframe(df)

            leaspy.fit(data, settings)

            for method in ['scipy_minimize']:
                burn_in_kw = dict() # not for all algos
                if '_real' in method:
                    burn_in_kw = dict(n_burn_in_iter=90, )
                settings = AlgorithmSettings(method, n_iter=100, seed=0, **burn_in_kw)
                result = leaspy.personalize(data, settings)
