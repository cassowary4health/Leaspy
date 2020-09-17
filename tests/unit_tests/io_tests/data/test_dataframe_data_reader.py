import os
import unittest
import pandas as pd

from tests import test_data_dir
from leaspy.io.data.dataframe_data_reader import DataframeDataReader


class DataframeDataReaderTest(unittest.TestCase):

    def test_constructor_univariate(self):
        path = os.path.join(test_data_dir, 'io', "data", 'univariate_data.csv')
        df = pd.read_csv(path)

        reader = DataframeDataReader(df)

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
        path = os.path.join(test_data_dir, 'io', "data", 'multivariate_data.csv')
        df = pd.read_csv(path)

        reader = DataframeDataReader(df)

        iter_to_idx = {
            0: '007_S_0041', 1: '100_S_0069', 2: '007_S_0101', 3: '130_S_0102', 4: '128_S_0138'
        }

        self.assertEqual(reader.iter_to_idx, iter_to_idx)
        self.assertEqual(reader.headers, ['ADAS11', 'ADAS13', 'MMSE'])
        self.assertEqual(reader.dimension, 3)
        self.assertEqual(reader.n_individuals, 5)
        self.assertEqual(reader.n_visits, 18)
