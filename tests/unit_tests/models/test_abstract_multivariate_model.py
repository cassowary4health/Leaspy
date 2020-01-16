import pandas as pd

from leaspy import AlgorithmSettings, Data, Leaspy
from leaspy.models.abstract_multivariate_model import AbstractMultivariateModel
from tests import example_data_path
from .test_abstract_model import AbstractModelTest
from .test_univariate_model import UnivariateModelTest


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

    def test_multivariate_run(self):
        logistic_leaspy = Leaspy('logistic')
        settings = AlgorithmSettings('mcmc_saem', n_iter=200, seed=0)

        df = pd.read_csv(example_data_path)
        data = Data.from_dataframe(df)

        logistic_leaspy.fit(data, settings)

        for method in ('mode_real', 'mean_real', 'scipy_minimize', 'gradient_descent_personalize'):
            settings = AlgorithmSettings(method, n_iter=100, n_burn_in_iter=90, seed=0)
            logistic_result = logistic_leaspy.personalize(data, settings)

    def test_multivariate_parallel_run(self):
        logistic_leaspy = Leaspy('logistic_parallel')
        settings = AlgorithmSettings('mcmc_saem', n_iter=200, seed=0)

        df = pd.read_csv(example_data_path)
        data = Data.from_dataframe(df)

        logistic_leaspy.fit(data, settings)

        for method in ('mode_real', 'mean_real', 'scipy_minimize', 'gradient_descent_personalize'):
            settings = AlgorithmSettings(method, n_iter=100, n_burn_in_iter=90, seed=0)
            logistic_result = logistic_leaspy.personalize(data, settings)