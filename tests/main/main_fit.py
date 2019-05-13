import os
from tests import test_data_dir
from src.main import Leaspy
from src.inputs.data_reader import DataReader
import unittest


class LeaspyFitTest(unittest.TestCase):
    ## Test MCMC-SAEM

    """
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

        self.assertAlmostEqual(leaspy.model.model_parameters['noise_var'], 0.0205, delta=0.002)
        self.assertAlmostEqual(leaspy.model.model_parameters['tau_mean'], 77.7, delta=2)
        self.assertAlmostEqual(leaspy.model.model_parameters['tau_var'], 41.8, delta=2)
        self.assertAlmostEqual(leaspy.model.model_parameters['xi_mean'], -4.48, delta=0.2)
        self.assertAlmostEqual(leaspy.model.model_parameters['xi_var'], 0.013, delta=0.01)
        self.assertAlmostEqual(leaspy.model.model_parameters['p0'], 0.12, delta=0.08)


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

        self.assertAlmostEqual(leaspy.model.model_parameters['noise_var'], 0.0617, delta=0.01)
        self.assertAlmostEqual(leaspy.model.model_parameters['intercept_mean'], 0.17, delta=0.02)
        self.assertAlmostEqual(leaspy.model.model_parameters['intercept_var'], 0.00519, delta=0.0015)

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
        self.assertAlmostEqual(leaspy.model.model_parameters['noise_var'], 0.0448, delta=0.01)
        self.assertAlmostEqual(leaspy.model.model_parameters['intercept_mean'], 0.1489, delta=0.01)
        self.assertAlmostEqual(leaspy.model.model_parameters['intercept_var'], 0.005, delta=0.01)



    

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
        self.assertAlmostEqual(leaspy.model.model_parameters['tau_mean'], 91.297, delta=8)
        self.assertAlmostEqual(leaspy.model.model_parameters['tau_var'], 501, delta=10)
        self.assertAlmostEqual(leaspy.model.model_parameters['xi_mean'], -5.0749, delta=0.2)
        self.assertAlmostEqual(leaspy.model.model_parameters['xi_var'], 0.002, delta=0.08)"""