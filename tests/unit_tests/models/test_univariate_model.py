import pandas as pd

from leaspy import AlgorithmSettings, Data, Leaspy
from leaspy.models.univariate_model import UnivariateModel
from tests import example_data_path
from .test_abstract_model import AbstractModelTest


class UnivariateModelTest(AbstractModelTest):

    def test_univariate_constructor(self, model=None):
        """
        Test attribute's initialization of leaspy univariate model
        :return: exit code
        """
        print('Unit-test constructor UnivariateModel')

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

    def test_univariate_run(self):
        univariate_leaspy = Leaspy('univariate')
        settings = AlgorithmSettings('mcmc_saem', n_iter=200, seed=0)

        df = pd.read_csv(example_data_path)
        df = df[['ID', 'TIME', 'Y0']]
        data = Data.from_dataframe(df)

        univariate_leaspy.fit(data, settings)

        for method in ('mode_real', 'mean_real', 'scipy_minimize', 'gradient_descent_personalize'):
            settings = AlgorithmSettings(method, n_iter=100, n_burn_in_iter=90, seed=0)
            univariate_result = univariate_leaspy.personalize(data, settings)
