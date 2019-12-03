import filecmp
import os
import unittest
import json
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
            self.assertAlmostEqual(np.sum(diff**2), 0, delta=10e-7)

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
            self.assertAlmostEqual(np.sum(diff**2), 0, delta=10e-7)

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
            self.assertAlmostEqual(np.sum(diff**2), 0, delta=10e-8)

        # Remove the created file
        os.remove(save_path)


    def test_load_individual_parameters(self, path=None):
        """
        Test load individual parameters
        :param path: string - optional - data path
        :return: exit code
        """
        if path is None:
            data_path_torch = os.path.join(test_data_dir,
                                           'individual_parameters/individual_parameters-unit_tests-torch.pt')
            data_path_json = os.path.join(test_data_dir,
                                          'individual_parameters/individual_parameters-unit_tests-json.json')
            self.test_load_individual_parameters(data_path_torch)
            self.test_load_individual_parameters(data_path_json)
        else:
            individual_parameters = Leaspy.load_individual_parameters(path)
            self.assertTrue((individual_parameters['xi'] == tensor([[1], [2], [3]])).min().item() == 1)
            self.assertTrue((individual_parameters['tau'] ==  tensor([[2], [3], [4]])).min().item() == 1)
            self.assertTrue((individual_parameters['sources'] ==
                             tensor([[1, 2], [2, 3], [3, 4]])).min().item() == 1)

    def test_save_individual_parameters(self):
        individual_parameters = {'xi': tensor([[1], [2], [3]]),
                                 'tau': tensor([[2], [3], [4]]),
                                 'sources': tensor([[1, 2], [2, 3], [3, 4]])}

        data_path_torch = os.path.join(test_data_dir,
                                       'individual_parameters/individual_parameters-unit_tests-torch.pt')
        data_path_json = os.path.join(test_data_dir,
                                      'individual_parameters/individual_parameters-unit_tests-json.json')

        data_path_torch_copy = data_path_torch[:-3] + '-Copy.pt'
        data_path_json_copy = data_path_json[:-5] + '-Copy.json'

        # Test torch file saving
        Leaspy.save_individual_parameters(data_path_torch_copy,
                                          individual_parameters,
                                          human_readable=False)
        try:
            self.test_load_individual_parameters(data_path_torch_copy)
            # filecmp does not work on torch file object - two different file can encode the same object
            os.remove(data_path_torch_copy)
        except AssertionError:
            os.remove(data_path_torch_copy)
            raise AssertionError("Leaspy.save_individual_parameters did not produce the expected torch file")

        # Test json file saving
        Leaspy.save_individual_parameters(data_path_json_copy,
                                          individual_parameters,
                                          human_readable=True)
        try:
            self.assertTrue(filecmp.cmp(data_path_json, data_path_json_copy))
            os.remove(data_path_json_copy)
        except AssertionError:
            os.remove(data_path_json_copy)
            raise AssertionError("Leaspy.save_individual_parameters did not produce the expected json file")
