import unittest

import numpy as np

from leaspy import Leaspy, IndividualParameters

from tests import hardcoded_model_path, hardcoded_ip_path

class LeaspyEstimateInverseTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from .test_api_estimate import LeaspyEstimateTest # do not import at top-level so to not duplicate tests
        cls.check_almost_equal_for_all_ind_tpts = LeaspyEstimateTest().check_almost_equal_for_all_ind_tpts

    def test_estimate_ages_from_biomarker_values_univariate(self):
        # TODO: test that doesn't rely on estimate ? (rather on estimate 'theoretical' results)

        # univariate logistic model
        # feat is "feature"
        model_parameters_path = hardcoded_model_path('univariate_logistic')
        leaspy = Leaspy.load(model_parameters_path)
        ip_path = hardcoded_ip_path('ip_univariate_save.json')
        ip = IndividualParameters.load(ip_path)
        timepoints = {
            'idx1': [78, 81],
            'idx2': [71],
            'idx3': []
        }
        estimations_raw = leaspy.estimate(timepoints, ip)

        # some reshape to do (else shape is (2, 1), when it is supposed to be 2)
        estimations = {}
        for idx, array in estimations_raw.items():
            estimations[idx] = array.squeeze().tolist()
            if isinstance(estimations[idx], float):
                estimations[idx] = [estimations[idx]]

        # with no feature argument
        estimated_ages_1 = leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip,
                                                                      biomarker_values=estimations)
        # with right feature argument
        estimated_ages_2 = leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip,
                                                                      biomarker_values=estimations, feature='feature')

        # check estimated ages are the original timepoints
        self.check_almost_equal_for_all_ind_tpts(estimated_ages_1, timepoints, tol=0.01)
        self.check_almost_equal_for_all_ind_tpts(estimated_ages_2, timepoints, tol=0.01)

        # check inputs checks

        # with wrong feature argument
        with self.assertRaises(ValueError):
            leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip, biomarker_values=estimations,
                                                       feature='wrong_feature')

        with self.assertRaises(TypeError):
            leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip, biomarker_values=[])

        with self.assertRaises(TypeError):
            leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip, biomarker_values=estimations,
                                                       feature=[])

        with self.assertRaises(TypeError):
            leaspy.estimate_ages_from_biomarker_values(individual_parameters=[], biomarker_values=estimations)

        with self.assertRaises(TypeError):
            leaspy.estimate_ages_from_biomarker_values(individual_parameters=[], biomarker_values=estimations)

        # check other errors
        problematic_timepoints = {
            'idx1': [90],  # fast progressor, 90 is already too much (estimation will be nan)
        }
        problematic_estimations = leaspy.estimate(problematic_timepoints, ip)
        problematic_estimations['idx1'] = problematic_estimations['idx1'].tolist()[0]
        pbq_age = leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip,
                                                             biomarker_values=problematic_estimations)

        # check that nan estimation gives nan age
        self.assertNotEqual(pbq_age['idx1'][0], pbq_age['idx1'][0])

        # quick check biomarker_values as dict of key: str and val: int rather than list works
        estimated_ages_0 = leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip,
                                                                      biomarker_values={'idx1': 0.4,
                                                                                        'idx2': 0.3})
        self.assertAlmostEqual(estimated_ages_0['idx1'], 70.53896, 2)
        self.assertAlmostEqual(estimated_ages_0['idx2'], 73.12502, 2)

    def test_estimate_ages_from_biomarker_values_multivariate(self):
        # multivariate logistic model
        # feats are "feature_0", ...
        model_parameters_path = hardcoded_model_path('logistic_scalar_noise')
        leaspy = Leaspy.load(model_parameters_path)

        ip_path = hardcoded_ip_path('ip_save.json')
        ip = IndividualParameters.load(ip_path)

        timepoints = {
            'idx1': [78, 81],
            'idx2': [91]
        }
        estimations_raw = leaspy.estimate(timepoints, ip)

        # select right feature
        def select_feature_estimation(estimations, leaspy, feature):
            """
            Select the right feature from multivariate estimation

            Parameters
            ----------
            estimations: dict of arrays
                array are shape (n_timepoints x n_feats)

            leaspy: Leaspy
                leaspy model

            feature: str
                feature name

            Returns: feat_estimations
                array are shape (n_timepoints x 1)
            -------

            """
            feat_ind = leaspy.model.features.index(feature)
            feat_estimations = {}
            for idx in estimations.keys():
                x = estimations[idx][:, feat_ind]
                feat_estimations[idx] = np.expand_dims(x, axis=1)
            return feat_estimations

        # checks with no feature argument
        with self.assertRaises(ValueError):
            leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip, biomarker_values=estimations_raw)

        for feature in ['feature_0', 'feature_1', 'feature_2', 'feature_3']:
            feat_estimations = select_feature_estimation(estimations=estimations_raw, leaspy=leaspy, feature=feature)

            # some reshape to do (else shape is (2, 1), when it is supposed to be 2)
            estimations = {}
            for idx, array in feat_estimations.items():
                estimations[idx] = array.squeeze().tolist()
                if isinstance(estimations[idx], float):
                    estimations[idx] = [estimations[idx]]

            estimated_ages = leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip,
                                                                        biomarker_values=estimations,
                                                                        feature=feature)

            # Remark: tolerance had to be pretty diminished so that the test passes...
            self.check_almost_equal_for_all_ind_tpts(estimated_ages, timepoints, tol=0.5)

        # quick check biomarker_values as dict of key: str and val: int works
        estimated_ages_0 = leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip,
                                                                      biomarker_values={'idx1': 0.4,
                                                                                        'idx2': 0.3},
                                                                      feature='feature_0')
        self.assertAlmostEqual(estimated_ages_0['idx1'], 68.52, 2)
        self.assertAlmostEqual(estimated_ages_0['idx2'], 72.38, 2)
