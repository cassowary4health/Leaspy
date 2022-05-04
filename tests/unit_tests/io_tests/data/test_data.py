import enum
from leaspy.io.data.data import Data

from tests import LeaspyTestCase


class DataTest(LeaspyTestCase):

    def load_multivariate_data(self):
        path_to_data = self.get_test_data_path('data_mock', 'multivariate_data.csv')
        return Data.from_csv_file(path_to_data)

    def test_constructor_univariate(self):
        path_to_data = self.get_test_data_path('data_mock', 'univariate_data.csv')
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
        data = self.load_multivariate_data()
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

    def test_data_slicing(self):
        data = self.load_multivariate_data()
        start, stop = 1, 5
        sub_data = data[start:stop]

        individual_key = 3
        individual = data[individual_key]
        sub_individual = sub_data[individual_key - start]

        self.assertEqual(data.headers, sub_data.headers)
        self.assertEqual(data.dimension, sub_data.dimension)
        self.assertEqual(data.cofactors, sub_data.cofactors)
        self.assertEqual(sub_data.n_individuals, min(data.n_individuals, stop) - start)

        self.assertEqual(individual.idx, sub_individual.idx)
        self.assertEqual(individual.timepoints, sub_individual.timepoints)
        self.assertEqual(individual.observations, sub_individual.observations)

    def test_data_iteration(self):
        data = self.load_multivariate_data()
        for iter, individual in enumerate(data):
            expected_individual = data[iter]
            self.assertEqual(individual.idx, expected_individual.idx)
            self.assertEqual(individual.timepoints, expected_individual.timepoints)
            self.assertEqual(individual.observations, expected_individual.observations)
            if iter > 4:
                break
