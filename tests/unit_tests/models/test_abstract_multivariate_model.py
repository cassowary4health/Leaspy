import unittest

from leaspy.models.abstract_multivariate_model import AbstractMultivariateModel
from .test_univariate_model import UnivariateModelTest


class AbstractMultivariateModelTest(unittest.TestCase):

    def test_constructor_abstract_multivariate(self, model=None):
        """
        Test attribute's initialization of leaspy abstract multivariate model

        Parameters
        ----------
        model : leaspy.models.abstract_model.AbstractModel, optional (default None)
            An instance of a subclass of leaspy AbstractModel.
        """
        print('\nUnit-test AbstractMultivariateModel\n')

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

