import torch

from leaspy.io.data.data import Data
from leaspy.io.data.dataset import Dataset

from tests import LeaspyTestCase


class DatasetTest(LeaspyTestCase):

    def test_constructor_univariate(self):
        # no nans
        path_to_data = self.get_test_data_path('data_mock', 'univariate_data_for_dataset.csv')
        data = Data.from_csv_file(path_to_data)
        dataset = Dataset(data)

        self.assertEqual(dataset.n_individuals, 3)
        self.assertEqual(dataset.n_visits_max, 4)
        self.assertEqual(dataset.dimension, 1)
        self.assertEqual(dataset.n_visits, 9)
        self.assertEqual(dataset.n_observations, 9)  # since univariate

        values = torch.tensor([[[1.], [5.], [2.], [0.]],
                           [[1.], [5.], [0.], [0.]],
                           [[1.], [8.], [1.], [3.]]])

        mask = torch.tensor([[[1.], [1.], [1.], [0.]],
                        [[1.], [1.], [0.], [0.]],
                        [[1.], [1.], [1.], [1.]]])

        self.assertTrue(torch.equal(dataset.values, values))
        self.assertTrue(torch.equal(dataset.mask, mask))

    def test_constructor_multivariate(self):
        # no nans
        path_to_data = self.get_test_data_path('data_mock', 'multivariate_data_for_dataset.csv')
        data = Data.from_csv_file(path_to_data)
        dataset = Dataset(data)

        self.assertEqual(dataset.n_individuals, 3)
        self.assertEqual(dataset.n_visits_max, 4)
        self.assertEqual(dataset.dimension, 2)
        self.assertEqual(dataset.n_visits, 9)
        self.assertEqual(dataset.n_observations, 18)  # since bivariate without nans

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

    def test_n_observations_missing_values(self):

        path_to_data = self.get_test_data_path('data_mock', 'multivariate_data_for_dataset_with_nans.csv')
        data = Data.from_csv_file(path_to_data)
        dataset = Dataset(data)

        self.assertEqual(dataset.n_individuals, 3)
        self.assertEqual(dataset.n_visits_max, 4)
        self.assertEqual(dataset.dimension, 2)
        self.assertEqual(dataset.n_visits, 8)  # 1 row full of nans should have been dropped
        self.assertEqual(dataset.n_observations, 2*8-3)  # 3 nans
