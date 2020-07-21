import os
import unittest
import torch

from tests import test_data_dir
from leaspy.io.data.data import Data
from leaspy.io.data.dataset import Dataset


class DatasetTest(unittest.TestCase):

    def test_constructor_univariate(self):
        path_to_data = os.path.join(test_data_dir, 'io', "data", 'univariate_data_for_dataset.csv')
        data = Data.from_csv_file(path_to_data)
        dataset = Dataset(data)

        self.assertEqual(dataset.n_individuals, 3)
        self.assertEqual(dataset.max_observations, 4)
        self.assertEqual(data.dimension, 1)

        values = torch.tensor([[[1.], [5.], [2.], [0.]],
                           [[1.], [5.], [0.], [0.]],
                           [[1.], [8.], [1.], [3.]]])

        mask = torch.tensor([[[1.], [1.], [1.], [0.]],
                        [[1.], [1.], [0.], [0.]],
                        [[1.], [1.], [1.], [1.]]])

        self.assertTrue(torch.equal(dataset.values, values))
        self.assertTrue(torch.equal(dataset.mask, mask))

    def test_constructor_multivariate(self):
        path_to_data = os.path.join(test_data_dir, 'io', "data", 'multivariate_data_for_dataset.csv')
        data = Data.from_csv_file(path_to_data)
        dataset = Dataset(data)

        self.assertEqual(dataset.n_individuals, 3)
        self.assertEqual(dataset.max_observations, 4)
        self.assertEqual(data.dimension, 2)

        values = torch.tensor([[[1., 1.], [5., 2.], [2., 3.], [0., 0.]],
                           [[1., 1.], [5., 8.], [0., 0.], [0., 0.]],
                           [[1., 4.], [8., 1.], [1., 1.], [3., 2.]]])

        mask = torch.tensor([[[1.], [1.], [1.], [0.]],
                        [[1.], [1.], [0.], [0.]],
                        [[1.], [1.], [1.], [1.]]])

        timepoints = torch.tensor([
            [1., 2., 3., 0.],
            [1., 2., 0., 0.],
            [1., 2., 4., 5.]
        ])

        self.assertTrue(torch.equal(dataset.values, values))
        # print(dataset.mask)
        # print(mask)
        # self.assertTrue(torch.equal(dataset.mask, mask)) #TODO check this
        # print(dataset.timepoints)
        self.assertAlmostEqual((dataset.timepoints - timepoints).sum(), 0, delta=10e-5)
