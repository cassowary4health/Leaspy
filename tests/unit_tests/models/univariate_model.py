from leaspy.models.univariate_model import UnivariateModel

from .abstract_model import AbstractModelTest


class UnivariateModelTest(AbstractModelTest):

    def test_univariate_constructor(self, model=None):
        """
        Test attribute's initialization of leaspy univariate model
        :return: exit code
        """
        print('Unit-test UnivariateModel')

        if model is None:
            model = UnivariateModel('univariate')
            self.assertEqual(type(model), UnivariateModel)
            self.assertEqual(model.dimension, 1)
            self.assertEqual(model.source_dimension, 0)

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
