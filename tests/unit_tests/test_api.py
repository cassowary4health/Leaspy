import json
import os
import unittest

import numpy as np
import torch
from torch import tensor

from leaspy.api import Leaspy
from leaspy.models.model_factory import ModelFactory
from tests import test_data_dir
from tests.unit_tests.models.test_model_factory import ModelFactoryTest


def ordered(obj):
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj


class LeaspyTest(unittest.TestCase):

    def test_constructor(self):
        """
        Test attribute's initialization of leaspy univariate model
        :return: exit code
        """
        for name in ['univariate', 'linear', 'logistic', 'logistic_parallel', 'mixed_linear-logistic']:
            leaspy = Leaspy(name)
            self.assertEqual(leaspy.type, name)
            self.assertEqual(type(leaspy.model), type(ModelFactory.model(name)))
            ModelFactoryTest().test_model_factory_constructor(leaspy.model)

    def test_load_logistic(self):
        """
        Test the initialization of a logistic model from a json file
        """
        model_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic.json')
        leaspy = Leaspy.load(model_path)

        # Test the name
        self.assertEqual(leaspy.type, 'logistic')
        self.assertEqual(type(leaspy.model), type(ModelFactory.model('logistic')))

        # Test the hyperparameters
        self.assertEqual(leaspy.model.dimension, 4)
        self.assertEqual(leaspy.model.features, ['feature_0', 'feature_1', 'feature_2', 'feature_3'])
        self.assertEqual(leaspy.model.source_dimension, 2)
        self.assertEqual(leaspy.model.loss, 'MSE')

        # Test the parameters
        parameters = {
            "g": [0.5, 1.5, 1.0, 2.0],
            "v0": [-2.0, -3.5, -3.0, -2.5],
            "betas": [[0.01, 0.06], [-0.01, 0.04], [0.03, 0.08]],
            "tau_mean": 75.2,
            "tau_std": 7.1,
            "xi_mean": 0.0,
            "xi_std": 0.2,
            "sources_mean": 0.0,
            "sources_std": 1.0,
            "noise_std": 0.2
        }
        for k, v in parameters.items():
            equality = torch.all(torch.eq(leaspy.model.parameters[k], tensor(v)))
            self.assertEqual(equality, True)

        # Test the initialization
        self.assertEqual(leaspy.model.is_initialized, True)

    def test_load_logistic_parallel(self):
        """
        Test the initialization of a logistic parallel model from a json file
        """
        model_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic_parallel.json')
        leaspy = Leaspy.load(model_path)

        # Test the name
        self.assertEqual(leaspy.type, 'logistic_parallel')
        self.assertEqual(type(leaspy.model), type(ModelFactory.model('logistic_parallel')))

        # Test the hyperparameters
        self.assertEqual(leaspy.model.dimension, 4)
        self.assertEqual(leaspy.model.features, ['feature_0', 'feature_1', 'feature_2', 'feature_3'])
        self.assertEqual(leaspy.model.source_dimension, 2)
        self.assertEqual(leaspy.model.loss, 'MSE')

        # Test the parameters
        parameters = {
            "g": 1.0,
            "tau_mean": 70.4,
            "tau_std": 7.0,
            "xi_mean": -1.7,
            "xi_std": 1.0,
            "sources_mean": 0.0,
            "sources_std": 1.0,
            "noise_std": 0.1,
            "deltas": [-3, -2.5, -1.0],
            "betas": [[0.1, -0.1], [0.5, -0.3], [0.3, 0.4]],
        }
        for k, v in parameters.items():
            equality = torch.all(torch.eq(leaspy.model.parameters[k], tensor(v)))
            self.assertEqual(equality, True)

        # Test the initialization
        self.assertEqual(leaspy.model.is_initialized, True)

    def test_load_linear(self):
        """
        Test the initialization of a linear model from a json file
        """
        model_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'linear.json')
        leaspy = Leaspy.load(model_path)

        # Test the name
        self.assertEqual(leaspy.type, 'linear')
        self.assertEqual(type(leaspy.model), type(ModelFactory.model('linear')))

        # Test the hyperparameters
        self.assertEqual(leaspy.model.dimension, 4)
        self.assertEqual(leaspy.model.features, ['feature_0', 'feature_1', 'feature_2', 'feature_3'])
        self.assertEqual(leaspy.model.source_dimension, 2)
        self.assertEqual(leaspy.model.loss, 'MSE')

        # Test the parameters
        parameters = {
            "g": [0.5, 0.06, 0.1, 0.3],
            "v0": [-0.5, -0.5, -0.5, -0.5],
            "betas": [[0.01, -0.05], [-0.01, 0.01], [-0.001, -0.01]],
            "tau_mean": 75.2,
            "tau_std": 0.9,
            "xi_mean": 0.0,
            "xi_std": 0.3,
            "sources_mean": 0.0,
            "sources_std": 1.0,
            "noise_std": 3.3,

        }
        for k, v in parameters.items():
            equality = torch.all(torch.eq(leaspy.model.parameters[k], tensor(v)))
            self.assertEqual(equality, True)

        # Test the initialization
        self.assertEqual(leaspy.model.is_initialized, True)

    def test_load_univariate(self):
        """
        Test the initialization of a linear model from a json file
        """
        model_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'univariate.json')
        leaspy = Leaspy.load(model_path)

        # Test the name
        self.assertEqual(leaspy.type, 'univariate')
        self.assertEqual(type(leaspy.model), type(ModelFactory.model('univariate')))

        # Test the hyperparameters
        self.assertEqual(leaspy.model.features, ['feature'])
        self.assertEqual(leaspy.model.loss, 'MSE')

        # Test the parameters
        parameters = {
            "g": 1.0,
            "tau_mean": 70.0,
            "tau_std": 2.5,
            "xi_mean": -1.0,
            "xi_std": 0.01,
            "noise_std": 0.2
        }

        for k, v in parameters.items():
            equality = torch.all(torch.eq(leaspy.model.parameters[k], tensor(v)))
            self.assertEqual(equality, True)

        # Test the initialization
        self.assertEqual(leaspy.model.is_initialized, True)

    def test_save_logistic(self):
        """
        Test saving the logistic model
        """
        data_path = os.path.join(test_data_dir, 'model_parameters', 'example')
        model_path = os.path.join(data_path, 'logistic.json')
        leaspy = Leaspy.load(model_path)

        # Save the file
        save_path = os.path.join(data_path, 'logistic-copy.json')
        leaspy.save(save_path)

        # Check that the files are the same
        with open(model_path, 'r') as f1:
            model_parameters = json.load(f1)
        with open(save_path, 'r') as f2:
            model_parameters_new = json.load(f2)

        self.assertEqual(model_parameters.keys(), model_parameters_new.keys())
        self.assertEqual(model_parameters['parameters'].keys(), model_parameters_new['parameters'].keys())

        for k, v in model_parameters['parameters'].items():
            diff = np.array(v) - np.array(model_parameters_new['parameters'][k])
            self.assertAlmostEqual(np.sum(diff**2).item(), 0, delta=10e-7)

        # Remove the created file
        os.remove(save_path)

    def test_save_logistic_parallel(self):
        """
        Test saving the logistic parallel model
        """
        data_path = os.path.join(test_data_dir, 'model_parameters', 'example')
        model_path = os.path.join(data_path, 'logistic_parallel.json')
        leaspy = Leaspy.load(model_path)

        # Save the file
        save_path = os.path.join(data_path, 'logistic_parallel-copy.json')
        leaspy.save(save_path)

        # Check that the files are the same
        with open(model_path, 'r') as f1:
            model_parameters = json.load(f1)
        with open(save_path, 'r') as f2:
            model_parameters_new = json.load(f2)

        self.assertEqual(model_parameters.keys(), model_parameters_new.keys())
        self.assertEqual(model_parameters['parameters'].keys(), model_parameters_new['parameters'].keys())

        for k, v in model_parameters['parameters'].items():
            diff = np.array(v) - np.array(model_parameters_new['parameters'][k])
            self.assertAlmostEqual(np.sum(diff**2).item(), 0, delta=10e-7)

        # Remove the created file
        os.remove(save_path)

    def test_save_linear(self):
        """
        Test saving the logistic model
        """
        data_path = os.path.join(test_data_dir, 'model_parameters', 'example')
        model_path = os.path.join(data_path, 'linear.json')
        leaspy = Leaspy.load(model_path)

        # Save the file
        save_path = os.path.join(data_path, 'linear-copy.json')
        leaspy.save(save_path)

        # Check that the files are the same
        with open(model_path, 'r') as f1:
            model_parameters = json.load(f1)
        with open(save_path, 'r') as f2:
            model_parameters_new = json.load(f2)

        self.assertEqual(model_parameters.keys(), model_parameters_new.keys())
        self.assertEqual(model_parameters['parameters'].keys(), model_parameters_new['parameters'].keys())

        for k, v in model_parameters['parameters'].items():
            diff = np.array(v) - np.array(model_parameters_new['parameters'][k])
            self.assertAlmostEqual(np.sum(diff**2).item(), 0, delta=10e-8)

        # Remove the created file
        os.remove(save_path)

    def test_save_univariate(self):
        """
        Test saving the univariate model
        """
        data_path = os.path.join(test_data_dir, 'model_parameters', 'example')
        model_path = os.path.join(data_path, 'univariate.json')
        leaspy = Leaspy.load(model_path)

        # Save the file
        save_path = os.path.join(data_path, 'univariate-copy.json')
        leaspy.save(save_path)

        # Check that the files are the same
        with open(model_path, 'r') as f1:
            model_parameters = json.load(f1)
        with open(save_path, 'r') as f2:
            model_parameters_new = json.load(f2)

        self.assertEqual(model_parameters.keys(), model_parameters_new.keys())
        self.assertEqual(model_parameters['parameters'].keys(), model_parameters_new['parameters'].keys())

        for k, v in model_parameters['parameters'].items():
            diff = np.array(v) - np.array(model_parameters_new['parameters'][k])
            self.assertAlmostEqual(np.sum(diff**2).item(), 0, delta=10e-8)

        # Remove the created file
        os.remove(save_path)
