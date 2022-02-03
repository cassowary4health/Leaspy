import pandas as pd

from leaspy.io.data.csv_data_reader import CSVDataReader

from tests import LeaspyTestCase


class CSVDataReaderTest(LeaspyTestCase):

    def test_constructor_univariate(self):
        path = self.get_test_data_path('data_mock', 'univariate_data.csv')
        reader = CSVDataReader(path)

        iter_to_idx = {
            0: '100_S_0006', 1: '018_S_0103', 2: '027_S_0179', 3: '035_S_0204',
            4: '068_S_0210', 5: '005_S_0223', 6: '130_S_0232'
        }

        self.assertEqual(reader.iter_to_idx, iter_to_idx)
        self.assertEqual(reader.headers, ['MMSE'])
        self.assertEqual(reader.dimension, 1)
        self.assertEqual(reader.n_individuals, 7)
        self.assertEqual(reader.n_visits, 33)


    def test_constructor_multivariate(self):
        path = self.get_test_data_path('data_mock', 'multivariate_data.csv')
        reader = CSVDataReader(path)

        iter_to_idx = {
            0: '007_S_0041', 1: '100_S_0069', 2: '007_S_0101', 3: '130_S_0102', 4: '128_S_0138'
        }

        self.assertEqual(reader.iter_to_idx, iter_to_idx)
        self.assertEqual(reader.headers, ['ADAS11', 'ADAS13', 'MMSE'])
        self.assertEqual(reader.dimension, 3)
        self.assertEqual(reader.n_individuals, 5)
        self.assertEqual(reader.n_visits, 18)

    def test_load_data_with_missing_values(self):
        # only test that it works (was not the case previously...)!
        path = self.get_test_data_path('data_mock', 'missing_data', 'sparse_data.csv')
        reader = CSVDataReader(path, drop_full_nan=False)

        self.assertEqual(reader.dimension, 4)
        self.assertEqual(reader.n_individuals, 2)
        self.assertEqual(reader.individuals.keys(), {'S1', 'S2'})
        self.assertEqual(reader.n_visits, 14)

        nans_count_S1 = pd.DataFrame(reader.individuals['S1'].observations, columns=reader.headers).isna().sum(axis=0)
        pd.testing.assert_series_equal(nans_count_S1, pd.Series({'Y0': 5, 'Y1': 5, 'Y2': 5, 'Y3': 5}))

        nans_count_S2 = pd.DataFrame(reader.individuals['S2'].observations, columns=reader.headers).isna().sum(axis=0)
        pd.testing.assert_series_equal(nans_count_S2, pd.Series({'Y0': 6, 'Y1': 6, 'Y2': 6, 'Y3': 6}))

        reader = CSVDataReader(path)  # drop_full_nan=True by default
        self.assertEqual(reader.dimension, 4)
        self.assertEqual(reader.n_individuals, 2)
        self.assertEqual(reader.individuals.keys(), {'S1', 'S2'})
        self.assertEqual(reader.n_visits, 9)
