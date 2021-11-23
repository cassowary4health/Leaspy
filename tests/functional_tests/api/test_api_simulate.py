import unittest
import warnings

import numpy as np
import pandas as pd

from leaspy.io.outputs.result import Result

from tests import LeaspyTestCase


class LeaspySimulateTest_Mixin(LeaspyTestCase):
    """Mixin holding generic simulation methods that may be safely reused in other tests (no actual test here)."""

    @classmethod
    def generic_simulate(cls, hardcoded_model_name: str, hardcoded_ip_name: str, *,
                         algo_name='simulation', **algo_params):
        """Helper for a generic simulation in following tests."""

        # load saved model (hardcoded values)
        leaspy = cls.get_hardcoded_model(hardcoded_model_name)

        # load the right data
        data = cls.get_suited_test_data_for_model(hardcoded_model_name)

        # load saved individual parameters
        individual_parameters = cls.get_hardcoded_individual_params(hardcoded_ip_name)

        # create the simulate algo settings
        simulation_settings = cls.get_algo_settings(name=algo_name, **algo_params)

        # simulate new subjects and their data
        simulation_results = leaspy.simulate(individual_parameters, data, simulation_settings)

        # return result objects
        return simulation_results


    def check_consistency_of_simulation_results(self, simulation_settings, simulation_results, data, *, expected_results_file):
        # TODO: refact, so dirty!

        self.assertIsInstance(simulation_results, Result)
        self.assertEqual(simulation_results.data.headers, data.headers)
        n = simulation_settings.parameters['number_of_subjects']
        self.assertEqual(simulation_results.data.n_individuals, n)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            self.assertEqual(len(simulation_results.get_parameter_distribution('xi')), n)
            self.assertEqual(len(simulation_results.get_parameter_distribution('tau')), n)
            self.assertEqual(len(simulation_results.get_parameter_distribution('sources_0')), n)

        path_expected_sim_res = self.test_data_path("simulation", expected_results_file)

        ## uncomment to re-generate simulation results
        #simulation_results.data.to_dataframe().to_csv(path_expected_sim_res, index=False, float_format='{:.6g}'.format)

        # Test the reproducibility of simulate
        # round is necessary, writing and reading induces numerical errors of magnitude ~ 1e-13
        # BUT ON DIFFERENT MACHINE I CAN SEE ERROR OF MAGNITUDE 1e-5 !!!
        # TODO: Can we improve this??
        simulation_df = pd.read_csv(path_expected_sim_res)

        id_simulation_is_reproducible = simulation_df['ID'].equals(simulation_results.data.to_dataframe()['ID'])
        # Check ID before - str doesn't seem to work with numpy.allclose
        self.assertTrue(id_simulation_is_reproducible)

        round_decimal = 5
        simulation_is_reproducible = np.allclose(simulation_df.loc[:, simulation_df.columns != 'ID'].values,
                                        simulation_results.data.to_dataframe().
                                        loc[:, simulation_results.data.to_dataframe().columns != 'ID'].values,
                                        atol=10 ** (-round_decimal), rtol=10 ** (-round_decimal))
        # Use of numpy.allclose instead of pandas.testing.assert_frame_equal because of buggy behaviour reported
        # in https://github.com/pandas-dev/pandas/issues/22052

        # If reproducibility error > 1e-5 => display it + visit with the biggest reproducibility error
        error_message = ''
        if not simulation_is_reproducible:
            # simulation_df = pd.read_csv(path_expected_sim_res)
            max_diff = 0.
            value_v1 = 0.
            value_v2 = 0.
            count = 0
            tol = 10 ** (-round_decimal)
            actual_simu_df = simulation_results.data.to_dataframe()
            for v1, v2 in zip(simulation_df.loc[:, simulation_df.columns != 'ID'].values.tolist(),
                              actual_simu_df.loc[:, actual_simu_df.columns != 'ID'].values.tolist()):
                diff = [abs(val1 - val2) for val1, val2 in zip(v1, v2)]
                if max(diff) > tol:
                    count += 1
                if max(diff) > max_diff:
                    value_v1 = v1
                    value_v2 = v2
                    max_diff = max(diff)
            error_message += '\nTolerance error = %.1e' % tol
            error_message += '\nMaximum error = %.3e' % max_diff
            error_message += '\n' + str([round(v, round_decimal+1) for v in value_v1])
            error_message += '\n' + str([round(v, round_decimal+1) for v in value_v2])
            error_message += '\nNumber of simulated visits above tolerance error = %d / %d \n' \
                             % (count, simulation_df.shape[0])
        # For loop before the last self.assert - otherwise no display is made
        self.assertTrue(simulation_is_reproducible, error_message)

class LeaspySimulateTest(LeaspySimulateTest_Mixin):

    @unittest.skip('TODO')
    def test_simulate_for_some_models(self):

        # TODO: hardcode a file with individuals parameters for each individual from data tiny!

        for model_codename, hardcoded_ip_file, simulation_params in [
            ('logistic_scalar_noise', ..., dict(number_of_subjects=100)),
            ('logistic_diag_noise', ..., dict(number_of_subjects=100)),
            ('logistic_binary', ..., dict(number_of_subjects=100)),
        ]:
            with self.subTest(model_codename=model_codename, **simulation_params):
                simulation_results = self.generic_simulate(model_codename, hardcoded_ip_file, **simulation_params)
