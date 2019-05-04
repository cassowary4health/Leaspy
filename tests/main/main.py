import os
import unittest

from tests import test_data_dir
from src.main import Leaspy
from src.models.univariate_model import UnivariateModel
from src.inputs.model_parameters_reader import ModelParametersReader
from src.inputs.data_reader import DataReader
from src.inputs.data import Data
import numpy as np

from src.algo.algo_factory import AlgoFactory
from src.inputs.algo_reader import AlgoReader

from src.algo.gradient_descent import GradientDescent

import torch


class LeaspyTest(unittest.TestCase):

    def test_constructor(self):
        leaspy = Leaspy('univariate')
        self.assertEqual(leaspy.type, 'univariate')
        self.assertEqual(type(leaspy.model), UnivariateModel)
        self.assertEqual(leaspy.model.model_parameters['p0'], 0.5)
        self.assertEqual(leaspy.model.model_parameters['tau_mean'], 70)
        self.assertEqual(leaspy.model.model_parameters['tau_std'], 5)
        self.assertEqual(leaspy.model.model_parameters['xi_mean'], -2)
        self.assertEqual(leaspy.model.model_parameters['xi_std'], 0.1)

    def test_constructor_from_parameters(self):
        path_to_model_parameters = os.path.join(test_data_dir, 'model_parameters.json')
        leaspy = Leaspy.from_parameters(path_to_model_parameters)
        self.assertEqual(leaspy.type, "univariate")
        self.assertEqual(leaspy.model.model_parameters['p0'], 0.3)
        self.assertEqual(leaspy.model.model_parameters['tau_mean'], 50)
        self.assertEqual(leaspy.model.model_parameters['tau_std'], 2)
        self.assertEqual(leaspy.model.model_parameters['xi_mean'], -10)
        self.assertEqual(leaspy.model.model_parameters['xi_std'], 0.8)


    #### Test on univariate data

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

        self.assertAlmostEqual(leaspy.model.model_parameters['noise_var'], 0.0205, delta=0.002)
        self.assertAlmostEqual(leaspy.model.model_parameters['tau_mean'], 85.3, delta=2)
        self.assertAlmostEqual(leaspy.model.model_parameters['tau_var'], 48.74, delta=2)
        self.assertAlmostEqual(leaspy.model.model_parameters['xi_mean'], -3.4109, delta=0.2)
        self.assertAlmostEqual(leaspy.model.model_parameters['xi_var'], 0.0025, delta=0.005)

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

        self.assertAlmostEqual(leaspy.model.model_parameters['noise_var'], 0.0672, delta=0.01)
        self.assertAlmostEqual(leaspy.model.model_parameters['mu'], 0.2012, delta=0.02)
        self.assertAlmostEqual(leaspy.model.model_parameters['intercept_var'], 0.00719, delta=0.001)

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
        self.assertAlmostEqual(leaspy.model.model_parameters['noise_var'], 0.0158, delta=0.01)
        self.assertAlmostEqual(leaspy.model.model_parameters['mu'], 0.16181408, delta=0.01)
        self.assertAlmostEqual(leaspy.model.model_parameters['intercept_var'], 0.015426399, delta=0.01)





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

        self.assertAlmostEqual(leaspy.model.model_parameters['noise_var'], 0.09405, delta=0.01)
        self.assertAlmostEqual(leaspy.model.model_parameters['tau_mean'], 75.99, delta=2)
        self.assertAlmostEqual(leaspy.model.model_parameters['tau_var'], 7.86, delta=0.5)
        self.assertAlmostEqual(leaspy.model.model_parameters['xi_mean'], -3.93, delta=0.2)
        self.assertAlmostEqual(leaspy.model.model_parameters['xi_var'], 0.96, delta=0.08)


"""



    def test_predict_gaussian_distribution_model(self):
        path_to_model_parameters = os.path.join(test_data_dir, '_fit_gaussiandistribution_gradientdescent','model_parameters.json')
        path_to_fitalgo_parameters = os.path.join(test_data_dir,
                                      '_fit_gaussiandistribution_gradientdescent', "algorithm_settings.json")
        path_to_predictalgo_parameters = os.path.join(test_data_dir,
                                      '_predict_gaussiandistribution_gradientdescent', "predict_algorithm_settings.json")

        leaspy = Leaspy.from_parameters(path_to_model_parameters)

        self.assertEqual(leaspy.type, 'gaussian_distribution')
        # Create the data
        data_path = os.path.join(test_data_dir, 'univariate_data.csv')
        reader = DataReader()
        data = reader.read(data_path)

        train_ids = data.indices[0:5]
        test_ids = data.indices[5:7]
        data_train, data_test = data.split(train_ids, test_ids)

        # Run or load parameters of already trained model ???
        leaspy.fit(data_train, path_to_fitalgo_parameters, seed=0)

        # Predict
        reals_ind = leaspy.predict(data_test, path_to_predictalgo_parameters, seed=0)

        # Assert
        for patient_id in reals_ind['intercept'].keys():
            self.assertAlmostEqual(reals_ind['intercept'][patient_id], np.mean(data_test[patient_id].raw_observations), delta=0.04)



    def test_simulate_gaussian_distribution_model(self):
        path_to_model_parameters = os.path.join(test_data_dir, '_fit_gaussiandistribution_gradientdescent','model_parameters.json')
        path_to_fitalgo_parameters = os.path.join(test_data_dir,
                                      '_fit_gaussiandistribution_gradientdescent', "algorithm_settings.json")
        path_to_simulation_parameters = os.path.join(test_data_dir,
                                      '_simulate_gaussiandistribution_gradientdescent', "simulation_parameters.json")

        leaspy = Leaspy.from_parameters(path_to_model_parameters)

        self.assertEqual(leaspy.type, 'gaussian_distribution')
        # Create the data
        data_path = os.path.join(test_data_dir, 'univariate_data.csv')
        reader = DataReader()
        data = reader.read(data_path)


        # Run or load parameters of already trained model ???
        leaspy.fit(data, path_to_fitalgo_parameters, seed=0)

        # Predict
        reals_ind = leaspy.simulate(path_to_simulation_parameters, seed=0)

        self.assertAlmostEqual(np.mean([value for value in reals_ind['intercept'].values()]), 0.16181408, delta=0.09)
        self.assertAlmostEqual(np.var([value for value in reals_ind['intercept'].values()]), 0.011426399, delta=0.02)



    def test_univariate_model(self):
        path_to_model_parameters = os.path.join(test_data_dir, '_univariate_gradientdescent', 'model_parameters.json')
        leaspy = Leaspy.from_parameters(path_to_model_parameters)

        self.assertEqual(leaspy.type, 'univariate')
        # Create the data
        data_path = os.path.join(test_data_dir, 'univariate_data.csv')
        reader = DataReader()
        data = reader.read(data_path)

        leaspy.fit(data, os.path.join(test_data_dir, '_univariate_gradientdescent', "fit_algorithm_settings.json"),
                   seed=0)        
        """






