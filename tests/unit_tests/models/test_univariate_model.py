from leaspy.models.abstract_model import AbstractModel
from leaspy.models.univariate_model import UnivariateModel

from tests import LeaspyTestCase


class ManifoldModelTest_Mixin(LeaspyTestCase):

    def check_common_attrs(self, model):
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

class UnivariateModelTest(ManifoldModelTest_Mixin):

    def test_univariate_constructor(self):
        """
        Test attribute's initialization of leaspy univariate model
        """
        for name in ['univariate_linear', 'univariate_logistic']:

            model = UnivariateModel(name)
            self.assertIsInstance(model, UnivariateModel)
            self.assertEqual(model.name, name)
            self.assertEqual(model.dimension, 1)
            self.assertEqual(model.source_dimension, 0)

            self.check_common_attrs(model)

