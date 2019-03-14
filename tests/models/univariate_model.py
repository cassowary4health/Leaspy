import unittest

from src.models.univariate_model import UnivariateModel


class UnivariateModelTest(unittest.TestCase):

    def test_constructor(self):
        model = UnivariateModel()
        self.assertEqual(model.model_parameters['p0'], 0.5)
        self.assertEqual(model.model_parameters['tau_mean'], 70)
        self.assertEqual(model.model_parameters['tau_std'], 5)
        self.assertEqual(model.model_parameters['xi_mean'], -2)
        self.assertEqual(model.model_parameters['xi_std'], 0.1)