import unittest
import os

import numpy as np
import pandas as pd

from tests import example_data_path
from leaspy import Data, Leaspy, AlgorithmSettings

#import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMParams


class LMEModelAPITest(unittest.TestCase):

    def setUp(self) -> None:
        # Data
        # read csv
        self.raw_data_df = pd.read_csv(example_data_path, dtype={'ID':str})
        self.raw_data_df['TIME'] = round(self.raw_data_df['TIME'], 3)

        self.raw_data_df.iloc[30,2] = np.nan

        ages = self.raw_data_df.dropna(subset=['Y0'])['TIME']
        self.ages_mean, self.ages_std = ages.mean(), ages.std(ddof=0)
        self.raw_data_df['TIME_norm'] = (self.raw_data_df['TIME'] - self.ages_mean) / self.ages_std

        # Data must have only one feature:
        data_df = self.raw_data_df[["ID", "TIME", "Y0"]]
        # from dataframe
        self.data = Data.from_dataframe(data_df)

        data_df_others_ix = data_df.copy()
        data_df_others_ix['ID'] += '_new' # but same data to test easily...

        self.data_new_ix = Data.from_dataframe(data_df_others_ix)

    def test_run(self):

        # Settings
        settings = AlgorithmSettings('lme_fit')

        self.assertDictEqual(settings.parameters, {
            'with_random_slope_age': False,
            'force_independent_random_effects': False,
            'method': ['lbfgs', 'bfgs', 'powell']
        })

        # Leaspy
        lsp = Leaspy('lme')
        lsp.fit(self.data, settings)

        self.assertListEqual(lsp.model.features, ['Y0'])
        self.assertEqual(lsp.model.dimension, 1)

        self.assertEqual(lsp.model.with_random_slope_age, False)
        #self.assertGreater(lsp.model.parameters['cov_re'][0,1].abs(), 0) # not forced independent

        self.assertAlmostEqual(self.ages_mean, lsp.model.parameters['ages_mean'], places=3)
        self.assertAlmostEqual(self.ages_std, lsp.model.parameters['ages_std'], places=3)

        # fit that should not work (not multivariate!)
        with self.assertRaises(ValueError):
            lsp.fit(Data.from_dataframe(self.raw_data_df), settings)

        # Personalize
        settings = AlgorithmSettings('lme_personalize')
        ip = lsp.personalize(self.data_new_ix, settings)

        # check statsmodels consistency
        self.check_consistency_sm(lsp.model.parameters, ip, re_formula='~1')

        # Personalize that shouldnt work (different feature)
        with self.assertRaises(ValueError):
            lsp.personalize(Data.from_dataframe(self.raw_data_df[["ID", "TIME", "Y1"]]), settings)

        # # Estimate
        timepoints = {'709_new': [80]}
        results = lsp.estimate(timepoints, ip)
        self.assertEqual(results.keys(), timepoints.keys())
        self.assertEqual(results['709_new'].shape, (1,1))
        self.assertAlmostEqual(results['709_new'][0,0], 0.57, places=2)

    def test_fake_data(self):
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
        self.assertEqual(easy_results['pat4'].shape, (2,1))
        self.assertAlmostEqual(easy_results['pat4'][0,0], 17, delta=10e-1)
        self.assertAlmostEqual(easy_results['pat4'][1,0], 18, delta=10e-1)

    def check_consistency_sm(self, model_params, ip, re_formula, **fit_kws):

        # compare
        lmm_test = smf.mixedlm(
            formula='Y0 ~ 1 + TIME_norm',
            data=self.raw_data_df.dropna(subset=['Y0']),
            re_formula=re_formula,
            groups='ID'
        ).fit(method='lbfgs', **fit_kws)

        # pop effects
        self.assertTrue( np.allclose(lmm_test.fe_params, model_params['fe_params']) )
        self.assertTrue( np.allclose(lmm_test.cov_re, model_params['cov_re']) )
        self.assertTrue( np.allclose(lmm_test.scale**.5, model_params['noise_std']) )

        # ind effects
        sm_ranef = lmm_test.random_effects
        for pat_id, ind_ip in ip.items():
            exp_ranef = sm_ranef[pat_id[:-len('_new')]]
            self.assertAlmostEqual( ind_ip['random_intercept'], exp_ranef[0], places=5 )
            if 'TIME' in re_formula:
                self.assertAlmostEqual( ind_ip['random_slope_age'], exp_ranef[1], places=5 )

        return lmm_test

    def test_with_random_slope_age(self):

        # Settings
        settings = AlgorithmSettings('lme_fit', with_random_slope_age=True)

        self.assertDictEqual(settings.parameters, {
            'with_random_slope_age': True,
            'force_independent_random_effects': False,
            'method': ['lbfgs', 'bfgs', 'powell']
        })

        # Leaspy
        lsp = Leaspy('lme')
        lsp.fit(self.data, settings)

        self.assertListEqual(lsp.model.features, ['Y0'])
        self.assertEqual(lsp.model.dimension, 1)

        self.assertEqual(lsp.model.with_random_slope_age, True)
        self.assertGreater(np.abs(lsp.model.parameters['cov_re'][0,1]), 0) # not forced independent

        print(repr(lsp.model.parameters))

        # + test save/load
        lsp.save('./tmp_lme_model_1.lock.json')
        del lsp

        lsp = Leaspy.load('./tmp_lme_model_1.lock.json')
        print(lsp)
        os.unlink('./tmp_lme_model_1.lock.json')

        # Personalize
        settings = AlgorithmSettings('lme_personalize')
        ip = lsp.personalize(self.data_new_ix, settings)

        # check statsmodels consistency
        self.check_consistency_sm(lsp.model.parameters, ip, re_formula='~1+TIME_norm')

    def test_with_random_slope_age_indep(self):

        # Settings
        settings = AlgorithmSettings('lme_fit', with_random_slope_age=True,
                                     force_independent_random_effects=True)

        self.assertDictEqual(settings.parameters, {
            'with_random_slope_age': True,
            'force_independent_random_effects': True,
            'method': ['lbfgs', 'bfgs']
        })

        # Leaspy
        lsp = Leaspy('lme')
        lsp.fit(self.data, settings)

        self.assertEqual(lsp.model.with_random_slope_age, True)
        self.assertAlmostEqual(lsp.model.parameters['cov_re'][0,1], 0, places=5) # forced independent

        # Personalize
        settings = AlgorithmSettings('lme_personalize')
        ip = lsp.personalize(self.data_new_ix, settings)

        # check statsmodels consistency
        free = MixedLMParams.from_components(
            fe_params=np.ones(2),
            cov_re=np.eye(2)
        )
        self.check_consistency_sm(lsp.model.parameters, ip, re_formula='~1+TIME_norm', free=free)

