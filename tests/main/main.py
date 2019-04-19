import os
import unittest

from tests import test_data_dir
from src.main import Leaspy
from src.models.univariate_model import UnivariateModel
from src.inputs.model_parameters_reader import ModelParametersReader
from src.inputs.data_reader import DataReader
from src.inputs.data import Data

from src.algo.algo_factory import AlgoFactory
from src.inputs.algo_reader import AlgoReader

from src.algo.gradient_descent import GradientDescent

import torch


class LeaspyTest(unittest.TestCase):

    def test_constructor(self):
        leaspy = Leaspy('univariate')
        self.assertEqual(leaspy.type, 'univariate')
        self.assertEqual(type(leaspy.model), UnivariateModel)
        self.assertEqual(leaspy.model.model_parameters['p0'], 0.5)
        self.assertEqual(leaspy.model.model_parameters['tau_mean'], 70)
        self.assertEqual(leaspy.model.model_parameters['tau_std'], 5)
        self.assertEqual(leaspy.model.model_parameters['xi_mean'], -2)
        self.assertEqual(leaspy.model.model_parameters['xi_std'], 0.1)

    def test_constructor_from_parameters(self):
        path_to_model_parameters = os.path.join(test_data_dir, 'model_parameters.json')
        leaspy = Leaspy.from_parameters(path_to_model_parameters)
        self.assertEqual(leaspy.type, "univariate")
        self.assertEqual(leaspy.model.model_parameters['p0'], 0.3)
        self.assertEqual(leaspy.model.model_parameters['tau_mean'], 50)
        self.assertEqual(leaspy.model.model_parameters['tau_std'], 2)
        self.assertEqual(leaspy.model.model_parameters['xi_mean'], -10)
        self.assertEqual(leaspy.model.model_parameters['xi_std'], 0.8)

    def test_gaussian_distribution_model(self):
        path_to_model_parameters = os.path.join(test_data_dir, '_gaussiandistribution_gradientdescent','model_parameters.json')
        leaspy = Leaspy.from_parameters(path_to_model_parameters)

        self.assertEqual(leaspy.type, 'gaussian_distribution')
        # Create the data
        data_path = os.path.join(test_data_dir, 'univariate_data.csv')
        reader = DataReader()
        data = reader.read(data_path)

        leaspy.fit(data, os.path.join(test_data_dir,
                                      '_gaussiandistribution_gradientdescent', "algorithm_settings.json"),
                   seed=0)
        self.assertAlmostEqual(leaspy.model.model_parameters['mu'], 0.16181408, delta=0.01)
        self.assertAlmostEqual(leaspy.model.model_parameters['intercept_var'], 0.011426399, delta=0.001)


    """

    def test_univariate_model(self):
        path_to_model_parameters = os.path.join(test_data_dir, '_univariate_gradientdescent', 'model_parameters.json')
        leaspy = Leaspy.from_parameters(path_to_model_parameters)

        self.assertEqual(leaspy.type, 'univariate')
        # Create the data
        data_path = os.path.join(test_data_dir, 'univariate_data.csv')
        reader = DataReader()
        data = reader.read(data_path)

        leaspy.fit(data, os.path.join(test_data_dir, '_univariate_gradientdescent', "algorithm_settings.json"),
                   seed=0)        
        """






