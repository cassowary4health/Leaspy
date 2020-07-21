import os
import unittest

from leaspy import Leaspy, Data, AlgorithmSettings
from tests import example_data_path
from tests import example_logisticmodel_path


class LeaspyPersonalizeTest(unittest.TestCase):

    # Test MCMC-SAEM

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
        path_settings = os.path.join(os.path.dirname(__file__), "data/settings_mean_real.json")
        algo_personalize_settings = AlgorithmSettings.load(path_settings)
        ips = leaspy.personalize(data, settings=algo_personalize_settings)

        # TODO REFORMAT: compute the noise std afterwards
        #self.assertAlmostEqual(result.noise_std,  0.108, delta=0.01)


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
        algo_personalize_settings = AlgorithmSettings('scipy_minimize', seed=0)
        ips = leaspy.personalize(data, settings=algo_personalize_settings)
        # TODO REFORMAT: compute the noise std afterwards
        #self.assertAlmostEqual(result.noise_std,  0.1169, delta=0.01)

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
        path_settings = os.path.join(os.path.dirname(__file__), "data/settings_mode_real.json")
        algo_personalize_settings = AlgorithmSettings.load(path_settings)
        ips = leaspy.personalize(data, settings=algo_personalize_settings)
        # TODO REFORMAT: compute the noise std afterwards
        #self.assertAlmostEqual(result.noise_std,   0.12152, delta=0.01)

    # TODO : problem with nans
    """
    def test_personalize_gradientdescent(self):
        # Inputs
        data = Data.from_csv_file(example_data_path)

        # Initialize
        leaspy = Leaspy.load(example_logisticmodel_path)

        # Launch algorithm
        algo_personalize_settings = AlgorithmSettings('gradient_descent_personalize', seed=2)
        result = leaspy.personalize(data, settings=algo_personalize_settings)

        self.assertAlmostEqual(result.noise_std,  0.17925, delta=0.01)"""
