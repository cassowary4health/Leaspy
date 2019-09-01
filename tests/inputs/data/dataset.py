import os
import unittest
import numpy as np
import torch

from tests import test_data_dir
from leaspy.inputs.data.data import Data
from leaspy.inputs.data.dataset import Dataset


class DatasetTest(unittest.TestCase):

    def test_constructor_univariate(self):
        path_to_data = os.path.join(test_data_dir, 'inputs', 'univariate_data_for_dataset.csv')
        data = Data.from_csv_file(path_to_data)
        dataset = Dataset(data)

        self.assertEqual(dataset.n_individuals, 3)
        self.assertEqual(dataset.max_observations, 4)
        self.assertEqual(data.dimension, 1)

        values = np.array([[[1.], [5.], [2.], [0.]],
                           [[1.], [5.], [0.], [0.]],
                           [[1.], [8.], [1.], [3.]]])

        mask = np.array([[[1.], [1.], [1.], [0.]],
                        [[1.], [1.], [0.], [0.]],
                        [[1.], [1.], [1.], [1.]]])

        self.assertEqual(np.array_equal(dataset.values, values), True)
        self.assertEqual(np.array_equal(dataset.mask, mask), True)

    def test_constructor_multivariate(self):
        path_to_data = os.path.join(test_data_dir, 'inputs', 'multivariate_data_for_dataset.csv')
        data = Data.from_csv_file(path_to_data)
        dataset = Dataset(data)

        self.assertEqual(dataset.n_individuals, 3)
        self.assertEqual(dataset.max_observations, 4)
        self.assertEqual(data.dimension, 2)

        values = np.array([[[1., 1.], [5., 2.], [2., 3.], [0., 0.]],
                           [[1., 1.], [5., 8.], [0., 0.], [0., 0.]],
                           [[1., 4.], [8., 1.], [1., 1.], [3., 2.]]])

        mask = np.array([[[1.], [1.], [1.], [0.]],
                        [[1.], [1.], [0.], [0.]],
                        [[1.], [1.], [1.], [1.]]])

        timepoints = torch.Tensor([
            [1., 2., 3., 0.],
            [1., 2., 0., 0.],
            [1., 2., 4., 5.]
        ])

        self.assertEqual(np.array_equal(dataset.values, values), True)
        #self.assertEqual(np.array_equal(dataset.mask, mask), True) #TODO check this
        print(dataset.timepoints)
        self.assertAlmostEqual((dataset.timepoints - timepoints).sum(), 0, delta=10e-5)
