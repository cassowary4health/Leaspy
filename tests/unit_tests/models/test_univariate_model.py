import unittest

from leaspy.models.abstract_model import AbstractModel
from leaspy.models.univariate_model import UnivariateModel


class UnivariateModelTest(unittest.TestCase):

    def test_univariate_constructor(self, model=None):
        """
        Test attribute's initialization of leaspy univariate model
        """
        print('\nUnit-test UnivariateModel\n')

        if model is None:
            model = UnivariateModel('univariate')
            self.assertEqual(type(model), UnivariateModel)
            self.assertEqual(model.dimension, 1)
            self.assertEqual(model.source_dimension, 0)

        self.assertTrue(issubclass(model.__class__, AbstractModel))

        self.assertEqual(model.attributes, None)
        self.assertEqual(model.bayesian_priors, None)

        self.assertEqual(model.parameters['g'], None)
        self.assertEqual(model.parameters['noise_std'], None)
        self.assertEqual(model.parameters['tau_mean'], None)
        self.assertEqual(model.parameters['tau_std'], None)
        self.assertEqual(model.parameters['xi_mean'], None)
        self.assertEqual(model.parameters['xi_std'], None)

        self.assertEqual(model.MCMC_toolbox['attributes'], None)
        self.assertEqual(model.MCMC_toolbox['priors']['g_std'], None)
