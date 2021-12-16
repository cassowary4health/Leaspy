import os

import pandas as pd

from leaspy.io.data.dataframe_data_reader import DataframeDataReader

from tests import LeaspyTestCase


class DataframeDataReaderTest(LeaspyTestCase):

    def test_constructor_univariate(self):
        path = os.path.join(self.test_data_dir, 'data_mock', 'univariate_data.csv')
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
        path = os.path.join(self.test_data_dir, 'data_mock', 'multivariate_data.csv')
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

    def test_load_data_with_missing_values(self):
        # only test that it works!
        path = os.path.join(self.test_data_dir, 'data_mock', 'missing_data', 'sparse_data.csv')
        df = pd.read_csv(path)
        reader = DataframeDataReader(df, drop_full_nan=False)

        self.assertEqual(reader.dimension, 4)
        self.assertEqual(reader.n_individuals, 2)
        self.assertEqual(reader.individuals.keys(), {'S1', 'S2'})
        self.assertEqual(reader.n_visits, 14)

        nans_count_S1 = pd.DataFrame(reader.individuals['S1'].observations, columns=reader.headers).isna().sum(axis=0)
        pd.testing.assert_series_equal(nans_count_S1, pd.Series({'Y0': 5, 'Y1': 5, 'Y2': 5, 'Y3': 5}))

        nans_count_S2 = pd.DataFrame(reader.individuals['S2'].observations, columns=reader.headers).isna().sum(axis=0)
        pd.testing.assert_series_equal(nans_count_S2, pd.Series({'Y0': 6, 'Y1': 6, 'Y2': 6, 'Y3': 6}))

        # drop_full_nan=True by default
        reader = DataframeDataReader(df)
        self.assertEqual(reader.dimension, 4)
        self.assertEqual(reader.n_individuals, 2)
        self.assertEqual(reader.individuals.keys(), {'S1', 'S2'})
        self.assertEqual(reader.n_visits, 9)

    def test_bad_input(self):

        df = pd.DataFrame({
            'ID':   ["S1", "S1", "S3", "S3", "S2", "S2"],
            'TIME': [75, 75.001, 76, 75, 65, 87],
            'Y0':   [.5, float('nan'), .5, .5, .5, .5],
            'Y1':   [.4]*6,
            'Y2':   [.5]*6,
        })

        # bad type (series)
        with self.assertRaises(ValueError):
            DataframeDataReader(df.set_index(['ID', 'TIME'])['Y0'])
        with self.assertRaises(ValueError):
            DataframeDataReader({'ID': [], 'TIME': [], 'Y0': []})

        # missing indexes
        with self.assertRaises(ValueError):
            DataframeDataReader(df.drop(columns='ID'))
        with self.assertRaises(ValueError):
            DataframeDataReader(df.drop(columns='TIME'))
        with self.assertRaises(ValueError):
            DataframeDataReader(df.drop(columns=['ID','TIME']))

        # bad ID
        with self.assertRaises(ValueError):
            # bad type
            DataframeDataReader(df.assign(ID=3.14))
        with self.assertRaises(ValueError):
            # empty string for 1 individual...
            DataframeDataReader(df.assign(ID=['']+['S1']*5))
        with self.assertRaises(ValueError):
            # nan string
            DataframeDataReader(df.assign(ID=[pd.NA]+['S1']*5))
        with self.assertRaises(ValueError):
            # < 0 index for 1 individual
            DataframeDataReader(df.assign(ID=[-1]+[0]*5))

        # bad TIME
        with self.assertRaises(ValueError):
            # bad type
            DataframeDataReader(df.assign(TIME=['75.12']*6))
        with self.assertRaises(ValueError):
            # no nan
            DataframeDataReader(df.assign(TIME=[float('nan')] + [75.12]*5))
        with self.assertRaises(ValueError):
            # no inf
            DataframeDataReader(df.assign(TIME=[float('-inf')] + [75.12]*5))

        # no duplicates on index
        with self.assertRaises(ValueError):
            # duplicates after rounding
            DataframeDataReader(pd.DataFrame({
                'ID':   ["S1", "S1"],
                'TIME': [75, 75-1e-10],
            }))

        # at least one feature & one row
        with self.assertRaises(ValueError):
            DataframeDataReader(df[['ID', 'TIME']])
        with self.assertRaises(ValueError):
            DataframeDataReader(df.iloc[[], :])

        # bad type for a column
        with self.assertRaises(ValueError):
            DataframeDataReader(df.assign(Y_bug=['0.33']*6))
        with self.assertRaises(ValueError):
            DataframeDataReader(df.assign(Y_bug=[float('-inf')]+[.4]*5))

        with self.assertWarnsRegex(UserWarning, 'full of nan'):
            DataframeDataReader(df.assign(Y3=[float('nan')]*6))

        # check that otherwise it passes
        reader = DataframeDataReader(df, sort_index=True)
        self.assertEqual(reader.n_individuals, 3)
        self.assertEqual(reader.n_visits, 6)
        self.assertEqual(list(reader.individuals.keys()), ['S1', 'S2', 'S3'])  # re-ordered
        self.assertEqual(reader.individuals['S3'].timepoints, [75., 76.])
