import unittest

import pandas as pd
import torch

from leaspy import AlgorithmSettings, Data, Leaspy
from leaspy.models.abstract_model import AbstractModel
from tests import example_data_path
from tests import example_logisticmodel_path


class AbstractModelTest(unittest.TestCase):

    def test_abstract_model_constructor(self):
        """
        Test initialization of abstract model class object

        Returns
        -------
        Exit code
        """
        print("Unit-test constructor AbstractModel")

        model = AbstractModel("dummy_abstractmodel")
        self.assertFalse(model.is_initialized)
        self.assertEqual(model.name, "dummy_abstractmodel")
        self.assertEqual(model.parameters, None)
        self.assertEqual(type(model.distribution), torch.distributions.normal.Normal)

        # Test the presence of all these essential methods
        main_methods = ['load_parameters', 'get_individual_variable_name', 'compute_sum_squared_tensorized',
                        'compute_individual_attachment_tensorized_mcmc', 'compute_individual_attachment_tensorized',
                        'update_model_parameters', 'update_model_parameters_burn_in',
                        'get_population_realization_names', 'get_individual_realization_names',
                        'compute_regularity_realization', 'compute_regularity_variable', 'get_realization_object']

        present_attributes = [_ for _ in dir(model) if _[:2] != '__']  # Get the present method

        for attribute in main_methods:
            self.assertTrue(attribute in present_attributes)
        # TODO: use python's hasattr and issubclass

    def test_load_parameters(self):
        """
        Test the method load_parameters

        Returns
        -------
        Exit code
        """
        leaspy_object = Leaspy.load(example_logisticmodel_path)

        abstract_model = AbstractModel("dummy_model")

        abstract_model.load_parameters(leaspy_object.model.parameters)

        self.assertTrue(torch.equal(abstract_model.parameters['g'],
                                    torch.tensor([1.8669992685317993, 2.4921786785125732,
                                                  2.471605062484741, 2.1240732669830322])))
        self.assertTrue(torch.equal(abstract_model.parameters['v0'],
                                    torch.tensor([-2.8300716876983643, -3.3241398334503174,
                                                  -3.4701175689697266, -2.6136295795440674])))
        self.assertTrue(torch.equal(abstract_model.parameters['betas'],
                                    torch.tensor([[0.011530596762895584, 0.06039918214082718],
                                                  [0.008324957452714443, 0.048168670386075974],
                                                  [0.01144738681614399, 0.0822334811091423]])))
        self.assertTrue(torch.equal(abstract_model.parameters['tau_mean'], torch.tensor(75.30111694335938)))
        self.assertTrue(torch.equal(abstract_model.parameters['tau_std'], torch.tensor(7.103002071380615)))
        self.assertTrue(torch.equal(abstract_model.parameters['xi_mean'], torch.tensor(0.0)))
        self.assertTrue(torch.equal(abstract_model.parameters['xi_std'], torch.tensor(0.2835913300514221)))
        self.assertTrue(torch.equal(abstract_model.parameters['sources_mean'], torch.tensor(0.0)))
        self.assertTrue(torch.equal(abstract_model.parameters['sources_std'], torch.tensor(1.0)))
        self.assertTrue(torch.equal(abstract_model.parameters['noise_std'], torch.tensor(0.1988248974084854)))

    def test_all_model_run(self):
        """
        Check if the following models ru with the following algorithms.
        """
        for model_name in ('linear', 'univariate', 'logistic', 'logistic_parallel'):
            logistic_leaspy = Leaspy(model_name)
            settings = AlgorithmSettings('mcmc_saem', n_iter=200, seed=0)

            df = pd.read_csv(example_data_path)
            if model_name == 'univariate':
                df = df.iloc[:, :3]
            data = Data.from_dataframe(df)

            logistic_leaspy.fit(data, settings)

            for method in ('mode_real', 'mean_real', 'scipy_minimize', 'gradient_descent_personalize'):
                settings = AlgorithmSettings(method, n_iter=100, n_burn_in_iter=90, seed=0)
                logistic_result = logistic_leaspy.personalize(data, settings)
