import torch
from leaspy.main import Leaspy

import unittest
from leaspy.inputs.data.data import Data

from leaspy.inputs.settings.algorithm_settings import AlgorithmSettings

from tests import example_logisticmodel_path
from tests import example_data_path

class LeaspyPersonalizeTest(unittest.TestCase):

    ## Test MCMC-SAEM



    def test_personalize_meanrealization(self):
        """
        Load logistic model from file, and personalize it to data from ...
        :return:
        """
        # Inputs
        data = Data.from_csv_file(example_data_path)

        # Initialize
        leaspy = Leaspy.load(example_logisticmodel_path)

        # Launch algorithm
        algo_personalize_settings = AlgorithmSettings('mean_real', seed=0)
        result = leaspy.personalize(data, settings=algo_personalize_settings)

        self.assertAlmostEqual(result.noise_std,  0.1584, delta=0.01)


    def test_personalize_scipy(self):
        """
        Load logistic model from file, and personalize it to data from ...
        :return:
        """
        # Inputs
        data = Data.from_csv_file(example_data_path)

        # Initialize
        leaspy = Leaspy.load(example_logisticmodel_path)

        # Launch algorithm
        algo_personalize_settings = AlgorithmSettings('scipy_minimize')
        result = leaspy.personalize(data, settings=algo_personalize_settings)

        self.assertAlmostEqual(result.noise_std,  0.1169, delta=0.01)

    def test_personalize_gradientdescent(self):
        """
        Load logistic model from file, and personalize it to data from ...
        :return:
        """
        # Inputs
        data = Data.from_csv_file(example_data_path)

        # Initialize
        leaspy = Leaspy.load(example_logisticmodel_path)

        # Launch algorithm
        algo_personalize_settings = AlgorithmSettings('gradient_descent_personalize')
        result = leaspy.personalize(data, settings=algo_personalize_settings)

        self.assertAlmostEqual(result.noise_std,  0.17925, delta=0.01)



    def test_personalize_modereal(self):
        """
        Load logistic model from file, and personalize it to data from ...
        :return:
        """
        # Inputs
        data = Data.from_csv_file(example_data_path)

        # Initialize
        leaspy = Leaspy.load(example_logisticmodel_path)

        # Launch algorithm
        algo_personalize_settings = AlgorithmSettings('mode_real')
        result = leaspy.personalize(data, settings=algo_personalize_settings)

        self.assertAlmostEqual(result.noise_std,  0.17925, delta=0.01)