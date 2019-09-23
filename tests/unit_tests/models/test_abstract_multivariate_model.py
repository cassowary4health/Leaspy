from .test_abstract_model import AbstractModelTest
from .test_univariate_model import UnivariateModelTest

from leaspy.models.abstract_multivariate_model import AbstractMultivariateModel


class AbstractMultivariateModelTest(AbstractModelTest):

    def test_constructor_abstract_multivariate(self, model=None):
        """
        Test attribute's initialization of leaspy abstract multivariate model
        :param model: leaspy model object
        :return: exit code
        """
        print('Unit-test constructor AbstractMultivariateModel')

        if model is None:
            # Abstract Multivariate Model
            model = AbstractMultivariateModel('dummy')
            self.assertEqual(type(model), AbstractMultivariateModel)
            self.assertEqual(model.name, 'dummy')

        # Test common initialization with univariate
        UnivariateModelTest().test_univariate_constructor(model)

        # Test specific multivariate initialization
        self.assertEqual(model.dimension, None)
        self.assertEqual(model.source_dimension, None)
        self.assertEqual(model.parameters['betas'], None)
        self.assertEqual(model.parameters['sources_mean'], None)
        self.assertEqual(model.parameters['sources_std'], None)
        self.assertEqual(model.MCMC_toolbox['priors']['betas_std'], None)
