import unittest

from leaspy.models.univariate_model import UnivariateModel


class UnivariateModelTest(unittest.TestCase):

    def test_constructor(self):
        model = UnivariateModel()
        self.assertEqual(model.model_parameters['p0'], None)
        self.assertEqual(model.model_parameters['tau_mean'], None)
        self.assertEqual(model.model_parameters['tau_var'], None)
        self.assertEqual(model.model_parameters['xi_mean'], None)
        self.assertEqual(model.model_parameters['xi_var'], None)