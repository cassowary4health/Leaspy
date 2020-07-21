import os
import unittest

from tests import test_data_dir
from leaspy.io.data.data import Data


class DataTest(unittest.TestCase):

    def test_constructor_univariate(self):
        path_to_data = os.path.join(test_data_dir, 'io', "data", 'univariate_data.csv')
        data = Data.from_csv_file(path_to_data)
        individual = data[2]

        self.assertEqual(data.iter_to_idx[0], '100_S_0006')
        self.assertEqual(data.iter_to_idx[len(data.iter_to_idx)-1], '130_S_0232')
        self.assertEqual(data.headers, ['MMSE'])
        self.assertEqual(data.dimension, 1)
        self.assertEqual(data.n_individuals, 7)
        self.assertEqual(data.n_visits, 33)
        self.assertEqual(data.cofactors, [])

        self.assertEqual(individual.idx, '027_S_0179')
        self.assertEqual(individual.timepoints, [80.9, 81.9, 82.4, 82.8])
        self.assertEqual(individual.observations, [[0.2], [0.2], [0.3], [0.5]])

    def test_constructor_multivariate(self):
        path_to_data = os.path.join(test_data_dir, 'io', "data", 'multivariate_data.csv')
        data = Data.from_csv_file(path_to_data)
        individual = data[3]

        self.assertEqual(data.iter_to_idx[0], '007_S_0041')
        self.assertEqual(data.iter_to_idx[len(data.iter_to_idx)-1], '128_S_0138')
        self.assertEqual(data.headers, ['ADAS11', 'ADAS13', 'MMSE'])
        self.assertEqual(data.dimension, 3)
        self.assertEqual(data.n_individuals, 5)
        self.assertEqual(data.n_visits, 18)
        self.assertEqual(data.cofactors, [])

        self.assertEqual(individual.idx, '130_S_0102')
        self.assertEqual(individual.timepoints, [71.3, 71.8])
        self.assertEqual(individual.observations, [[0.3, 0.4, 0.5], [0.4, 0.5, 0.6]])
