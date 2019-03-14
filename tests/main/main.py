import os
import unittest

from tests import test_data_dir
from src.main import Leaspy
from src.models.univariate_model import UnivariateModel
from src.inputs.model_parameters_reader import ModelParametersReader


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