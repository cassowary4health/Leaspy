import unittest
import pandas as pd
from tests import example_data_path
from leaspy import Data, Leaspy, AlgorithmSettings
import numpy as np


class LMEModelAPITest(unittest.TestCase):

    def test_run(self):
        # Data
        # read csv
        raw_data_df = pd.read_csv(example_data_path)
        # Data must have only one feature:
        data_df = raw_data_df[["ID", "TIME", "Y0"]]
        # from dataframe
        data = Data.from_dataframe(data_df)

        # Settings
        settings = AlgorithmSettings('lme_fit')

        # Leaspy
        model = Leaspy('lme')
        model.fit(data, settings)

        # fit that should not work
        with self.assertRaises(ValueError) as context:
            model.fit(Data.from_dataframe(raw_data_df), settings)

        # Personalize
        settings = AlgorithmSettings('lme_personalize')
        ip = model.personalize(data, settings)

        # Personalize that shouldnt work
        with self.assertRaises(ValueError) as context:
            model.personalize(Data.from_dataframe(raw_data_df[["ID", "TIME", "Y1"]]), settings)

        # # Estimate
        timepoints = {'709': [80]}
        results = model.estimate(timepoints, ip)
        self.assertEqual(results.keys(), timepoints.keys())
        self.assertEqual(list(results['709'].shape)[0], 1)
        self.assertAlmostEqual(results['709'].tolist()[0], 0.57, delta=10e-2)

        # easy fake data
        # try to see when fitting on df and personalizing on unseen_df
        df = pd.DataFrame.from_records((np.arange(3, 3 + 10, 1),
                                        np.arange(15, 15 + 10, 1),
                                        np.arange(6, 6 + 10, 1)),
                                       index=['pat1', 'pat2', 'pat3'],
                                       ).T.stack()
        df = pd.DataFrame(df)
        df.index.names = ['TIME', 'ID']
        df = df.rename(columns={0: 'feat1'})
        df = df.swaplevel()
        df.loc[('pat1', 0)] = np.nan

        unseen_df = pd.DataFrame.from_records((np.arange(2, 2 + 10, 1), np.arange(18, 18 + 10, 1)),
                                              index=['pat4', 'pat5'],
                                              ).T.stack()
        unseen_df = pd.DataFrame(unseen_df)
        unseen_df.index.names = ['TIME', 'ID']
        unseen_df = unseen_df.rename(columns={0: 'feat1'})
        unseen_df = unseen_df.swaplevel()

        # from dataframe
        easy_data = Data.from_dataframe(df)

        # Settings
        easy_settings = AlgorithmSettings('lme_fit')

        # Leaspy
        easy_model = Leaspy('lme')
        easy_model.fit(easy_data, easy_settings)

        # Personalize
        easy_perso_settings = AlgorithmSettings('lme_personalize')
        unseen_easy_data = Data.from_dataframe(unseen_df)
        ip = easy_model.personalize(unseen_easy_data, easy_perso_settings)

        # # Estimate
        easy_timepoints = {'pat4': [15, 16]}
        easy_results = easy_model.estimate(easy_timepoints, ip)
        self.assertEqual(easy_results.keys(), easy_timepoints.keys())
        self.assertEqual(list(easy_results['pat4'].shape)[0], 2)
        self.assertAlmostEqual(easy_results['pat4'].tolist()[0], 17, delta=10e-1)
        self.assertAlmostEqual(easy_results['pat4'].tolist()[0], 18, delta=10e-1)
