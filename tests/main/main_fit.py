import os
from tests import test_data_dir
from src.main import Leaspy
from src.inputs.data_reader import DataReader
import unittest


class LeaspyFitTest(unittest.TestCase):
    ## Test MCMC-SAEM


    def test_fit_univariatesigmoid_mcmcsaem(self):
        path_to_model_parameters = os.path.join(test_data_dir, '_fit_univariatesigmoid_mcmcsaem',
                                                'model_parameters.json')

        path_to_algo_parameters = os.path.join(test_data_dir,
                                               '_fit_univariatesigmoid_mcmcsaem', "algorithm_settings.json")

        path_output = '../output_leaspy/univariatesigmoid_mcmcsaem/'
        if not os.path.exists(path_output):
            if not os.path.exists('../output_leaspy'):
                os.mkdir('../output_leaspy')
            os.mkdir(path_output)

        leaspy = Leaspy.from_parameters(path_to_model_parameters)

        # Create the data
        data_path = os.path.join(test_data_dir, 'univariate_data.csv')
        reader = DataReader()
        data = reader.read(data_path)

        leaspy.fit(data, path_to_algo_parameters, path_output, seed=0)

        self.assertAlmostEqual(leaspy.model.model_parameters['noise_var'], 0.003, delta=0.02)
        self.assertAlmostEqual(leaspy.model.model_parameters['tau_mean'], 1.862, delta=2)
        self.assertAlmostEqual(leaspy.model.model_parameters['tau_var'], 1.231, delta=2)
        self.assertAlmostEqual(leaspy.model.model_parameters['xi_mean'], -0.920, delta=0.2)
        self.assertAlmostEqual(leaspy.model.model_parameters['xi_var'], 0.00277, delta=0.01)
        self.assertAlmostEqual(leaspy.model.model_parameters['p0'], 0.622, delta=0.08)


    #### Test on univariate data

    def test_fit_gaussiandisstribution_mcmcsaem(self):
        path_to_model_parameters = os.path.join(test_data_dir, '_fit_gaussiandistribution_mcmcsaem',
                                                'model_parameters.json')

        path_to_fitalgo_parameters = os.path.join(test_data_dir,
                                                  '_fit_gaussiandistribution_mcmcsaem', "algorithm_settings.json")

        path_output = '../output_leaspy/gaussiandistribution_mcmcsaem/'
        if not os.path.exists(path_output):
            if not os.path.exists('../output_leaspy'):
                os.mkdir('../output_leaspy')
            os.mkdir(path_output)

        leaspy = Leaspy.from_parameters(path_to_model_parameters)

        # Create the data
        data_path = os.path.join(test_data_dir, 'univariate_data.csv')
        reader = DataReader()
        data = reader.read(data_path)

        leaspy.fit(data, path_to_fitalgo_parameters, path_output, seed=0)

        self.assertAlmostEqual(leaspy.model.model_parameters['noise_var'], 0.0040, delta=0.01)
        self.assertAlmostEqual(leaspy.model.model_parameters['intercept_mean'], 0.15880, delta=0.02)
        self.assertAlmostEqual(leaspy.model.model_parameters['intercept_var'], 0.0117, delta=0.0015)

    ## Test Gradient Descent Algorithm

    

    def  test_fit_gaussiandistribution_gradientdescent(self):
        path_to_model_parameters = os.path.join(test_data_dir, '_fit_gaussiandistribution_gradientdescent',
                                                'model_parameters.json')

        path_to_fitalgo_parameters = os.path.join(test_data_dir,
                                                  '_fit_gaussiandistribution_gradientdescent',
                                                  "algorithm_settings.json")

        path_output = '../output_leaspy/gaussiandistribution_gradientdescent/'
        if not os.path.exists(path_output):
            if not os.path.exists('../output_leaspy'):
                os.mkdir('../output_leaspy')
            os.mkdir(path_output)

        leaspy = Leaspy.from_parameters(path_to_model_parameters)

        self.assertEqual(leaspy.type, 'gaussian_distribution')
        # Create the data
        data_path = os.path.join(test_data_dir, 'univariate_data.csv')
        reader = DataReader()
        data = reader.read(data_path)

        leaspy.fit(data, path_to_fitalgo_parameters, path_output, seed=0)
        self.assertAlmostEqual(leaspy.model.model_parameters['noise_var'], 0.00335, delta=0.01)
        self.assertAlmostEqual(leaspy.model.model_parameters['intercept_mean'], 0.1596, delta=0.01)
        self.assertAlmostEqual(leaspy.model.model_parameters['intercept_var'], 0.013, delta=0.01)





    def test_fit_univariatesigmoid_gradientdescent(self):
        path_to_model_parameters = os.path.join(test_data_dir, '_fit_univariatesigmoid_gradientdescent',
                                                'model_parameters.json')

        path_to_fitalgo_parameters = os.path.join(test_data_dir,
                                                  '_fit_univariatesigmoid_gradientdescent',
                                                  "algorithm_settings.json")

        path_output = '../output_leaspy/univariatesigmoid_gradientdescent/'
        if not os.path.exists(path_output):
            if not os.path.exists('../output_leaspy'):
                os.mkdir('../output_leaspy')
            os.mkdir(path_output)

        leaspy = Leaspy.from_parameters(path_to_model_parameters)


        # Create the data
        data_path = os.path.join(test_data_dir, 'univariate_data.csv')
        reader = DataReader()
        data = reader.read(data_path)

        leaspy.fit(data, path_to_fitalgo_parameters, path_output, seed=0)

        self.assertAlmostEqual(leaspy.model.model_parameters['noise_var'], 0.00358, delta=0.008)
        self.assertAlmostEqual(leaspy.model.model_parameters['tau_mean'], 1.004027, delta=8)
        self.assertAlmostEqual(leaspy.model.model_parameters['tau_var'], 1.143, delta=10)
        self.assertAlmostEqual(leaspy.model.model_parameters['xi_mean'], -1.273, delta=0.2)
        self.assertAlmostEqual(leaspy.model.model_parameters['xi_var'], 0.00239, delta=0.08)
        self.assertAlmostEqual(leaspy.model.model_parameters['p0'], 0.2830, delta=0.08)
