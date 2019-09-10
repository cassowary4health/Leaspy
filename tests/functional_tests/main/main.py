import os
import unittest

from tests import test_data_dir
from leaspy import Leaspy
from leaspy.models.univariate_model import UnivariateModel


class LeaspyTest(unittest.TestCase):

    def test_constructor(self):
        leaspy = Leaspy('univariate')
        self.assertEqual(leaspy.type, 'univariate')
        self.assertEqual(type(leaspy.model), UnivariateModel)
        self.assertEqual(leaspy.model.model_parameters['p0'], None)
        self.assertEqual(leaspy.model.model_parameters['tau_mean'], None)
        self.assertEqual(leaspy.model.model_parameters['tau_var'], None)
        self.assertEqual(leaspy.model.model_parameters['xi_mean'], None)
        self.assertEqual(leaspy.model.model_parameters['xi_var'], None)

    def test_constructor_from_parameters(self):
        path_to_model_parameters = os.path.join(test_data_dir, 'model_settings_univariate.json')
        leaspy = Leaspy.load(path_to_model_parameters)
        self.assertEqual(leaspy.type, "univariate")
        self.assertEqual(leaspy.model.model_parameters['p0'], [0.3])
        self.assertEqual(leaspy.model.model_parameters['tau_mean'], 50)
        self.assertEqual(leaspy.model.model_parameters['tau_var'], 2)
        self.assertEqual(leaspy.model.model_parameters['xi_mean'], -10)
        self.assertEqual(leaspy.model.model_parameters['xi_var'], 0.8)









