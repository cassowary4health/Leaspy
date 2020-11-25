import unittest
import numpy as np
from leaspy.io.settings.algorithm_settings import AlgorithmSettings
from leaspy.models.lme_model import LMEModel
from leaspy.algo.others.lme_fit import LMEFitAlgorithm
from leaspy.io.data.data import Data
from leaspy.io.data.dataset import Dataset
import pandas as pd


class LMEFitAlgorithmTest(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame.from_records((np.arange(3, 3 + 10, 1),
                                        np.arange(15, 15 + 10, 1),
                                        np.arange(6, 6 + 10, 1)),
                                       index=['pat1', 'pat2', 'pat3'],
                                       ).T.stack()
        df = pd.DataFrame(df)
        df.index.names = ['TIME', 'ID']
        df = df.rename(columns={0: 'feat1'})
        df = df.swaplevel()
        # add a nan
        df.loc[('pat1', 0)] = np.nan
        self.dataframe = df
        data = Data.from_dataframe(df)
        self.dataset = Dataset(data)
        # models
        self.model = LMEModel('lme')
        self.settings = AlgorithmSettings('lme_fit')
        self.algo = LMEFitAlgorithm(self.settings)

    def test_get_reformated(self):
        ages = self.algo._get_reformated(self.dataset, 'timepoints')
        expected_ages = np.array(self.dataframe.sort_index(axis=0).index.get_level_values('TIME'))[1:]
        values = self.algo._get_reformated(self.dataset, 'values')
        expected_values = np.array(self.dataframe.sort_index(axis=0)['feat1'].values)[1:]
        self.assertTrue((ages == expected_ages).all())
        self.assertTrue((values == expected_values).all())

    def test_get_reformated_subjects(self):
        subjects = self.algo._get_reformated_subjects(self.dataset)
        expected_subjects = ['pat1'] * 9 + ['pat2'] * 10 + ['pat3'] * 10
        self.assertTrue((subjects == expected_subjects).all())

    def test_run(self):
        self.algo.run(self.model, self.dataset)
        self.assertAlmostEqual(self.model.parameters["fe_params"][0], 8, 0)
        self.assertAlmostEqual(self.model.parameters["fe_params"][1], 1., 0)
        self.assertAlmostEqual(self.model.parameters["cov_re"][0][0], 2.8888, 3)
        self.assertAlmostEqual(self.model.parameters["cov_re_unscaled"][0][0], 14e9, -9)
        self.assertAlmostEqual(self.model.parameters["bse_fe"][0], 1, 0)
        self.assertAlmostEqual(self.model.parameters["bse_fe"][1], 0, 0)
        self.assertAlmostEqual(self.model.parameters["bse_re"][0], 57140, -3)
