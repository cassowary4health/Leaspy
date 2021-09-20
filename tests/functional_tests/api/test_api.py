import json
import os
import unittest
import warnings

import pandas as pd
from numpy import allclose

from leaspy import Leaspy, Data, AlgorithmSettings
from leaspy.io.outputs.result import Result
from tests import example_data_path, test_data_dir, from_fit_model_path


# def ordered(obj):
#     """
#     Order a list or a dictionary in order to compare it.
#
#     Parameters
#     ----------
#     obj: dict or list
#         Object to be ordered.
#
#     Returns
#     -------
#     obj: dict or list
#         Ordered object.
#     """
#     if isinstance(obj, dict):
#         return sorted((k, ordered(v)) for k, v in obj.items())
#     if isinstance(obj, list):
#         return sorted(ordered(x) for x in obj)
#     else:
#         return obj


def dict_compare_and_display(d, e):
    """
    Compare two dictionaries up to the standard tolerance of ``numpy.allclose`` and display their differences.

    Parameters
    ----------
    d : dict
    e : dict

    Returns
    -------
    bool
        Answer to ``d`` == ``e`` up to the standard tolerance of ``numpy.allclose``.
    """
    if d == e:
        return True

    if d.keys() != e.keys():
        print("The following keys are different!")
        print(d.keys() ^ e.keys())
        return False

    # loop on keys
    for k in d.keys():
        if isinstance(d[k], dict):
            if not dict_compare_and_display(d[k], e[k]):
                return False # return now only if False ;)
        else:
            try:
                atol = 2e-1 if 'tau' in k else 5e-2
                if not allclose(d[k], e[k], atol=atol):
                    print("The following values are different for `numpy.allclose`!")
                    print(f"{k}: {d[k]} != {e[k]}")
                    return False
            except TypeError:
                if d[k] != e[k]:
                    print("The following values are different!")
                    print(f"{k}: {d[k]} != {e[k]}")
                    return False

    # return at last if no return (False) before
    return True

class LeaspyTest(unittest.TestCase):

    def model_values_test(self, model, tol=1e-2):
        """
        Avoid copy past for the functional test.

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
        """

        self.assertEqual(model.name, "logistic")
        self.assertEqual(model.features, ['Y0', 'Y1', 'Y2', 'Y3'])
        """
        # already done in `test_fit_logistic`
        self.assertAlmostEqual(model.parameters['noise_std'], 0.1567, delta=tol)
        self.assertAlmostEqual(model.parameters['tau_mean'], 78.6136, delta=tol)
        self.assertAlmostEqual(model.parameters['tau_std'], 4.8767, delta=tol)
        self.assertAlmostEqual(model.parameters['xi_mean'], 0.0, delta=tol)
        self.assertAlmostEqual(model.parameters['xi_std'], 0.8518, delta=tol)
        diff_g = model.parameters['g'] - torch.tensor([0.0823, 2.9011, 2.4907, 1.1496])
        diff_v = model.parameters['v0'] - torch.tensor([-4.6248, -4.5740, -4.7679, -4.5572])
        self.assertAlmostEqual(torch.sum(diff_g ** 2).item(), 0.0, delta=tol**2)
        self.assertAlmostEqual(torch.sum(diff_v ** 2).item(), 0.0, delta=tol**2)
        """

    def test_usecase(self):
        """
        Functional test of a basic analysis using leaspy package

        1 - Data loading
        2 - Fit logistic model with MCMC algorithm
        3 - Save paramaters & reload (remove created files to keep the repo clean)
        4 - Personalize model with 'mode_real' algorithm
        5 - Plot results
        6 - Simulate new patients
        """
        data = Data.from_csv_file(example_data_path)

        # Fit
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0)
        leaspy = Leaspy("logistic")
        leaspy.model.load_hyperparameters({'source_dimension': 2})
        leaspy.fit(data, algorithm_settings=algo_settings)
        #print(leaspy.model.parameters)

        path_to_backup_model = from_fit_model_path('test_api')
        ## uncomment to re-generate expected file
        #leaspy.save(path_to_backup_model, indent=2)

        self.model_values_test(leaspy.model)

        # Save parameters and check its consistency
        path_to_saved_model = from_fit_model_path('test_api-copy')
        leaspy.save(path_to_saved_model, indent=2)

        with open(path_to_backup_model, 'r') as f1:
            model_parameters = json.load(f1)
            # don't compare leaspy exact version...
            model_parameters['leaspy_version'] = None
        with open(path_to_saved_model) as f2:
            model_parameters_new = json.load(f2)
            # don't compare leaspy exact version...
            model_parameters_new['leaspy_version'] = None
        # self.assertTrue(ordered(model_parameters) == ordered(model_parameters_new))
        self.assertTrue(dict_compare_and_display(model_parameters, model_parameters_new))

        # Load data and check its consistency

        #leaspy = Leaspy.load(path_to_saved_model)
        os.remove(path_to_saved_model)

        # unlink 1st functional fit test from next steps...
        leaspy = Leaspy.load(path_to_backup_model)

        self.assertTrue(leaspy.model.is_initialized)
        self.model_values_test(leaspy.model)

        # Personalize
        algo_personalize_settings = AlgorithmSettings('mode_real', seed=0)
        individual_parameters = leaspy.personalize(data, settings=algo_personalize_settings)
        # TODO REFORMAT: compute the noise std afterwards
        #self.assertAlmostEqual(result.noise_std, 0.21146, delta=0.01)

        ## Plot TODO
        #path_output = os.path.join(os.path.dirname(__file__), '../../_data', "_outputs")
        #plotter = Plotter(path_output)
        # plotter.plot_mean_trajectory(leaspy.model, save_as="mean_trajectory_plot")
        #plt.close()

        # Simulate
        simulation_settings = AlgorithmSettings('simulation', seed=0)
        simulation_results = leaspy.simulate(individual_parameters, data, simulation_settings)
        self.assertIsInstance(simulation_results, Result)
        self.assertEqual(simulation_results.data.headers, data.headers)
        n = simulation_settings.parameters['number_of_subjects']
        self.assertEqual(simulation_results.data.n_individuals, n)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            self.assertEqual(len(simulation_results.get_parameter_distribution('xi')), n)
            self.assertEqual(len(simulation_results.get_parameter_distribution('tau')), n)
            self.assertEqual(len(simulation_results.get_parameter_distribution('sources')['sources0']), n)

        path_expected_sim_res = os.path.join(test_data_dir,
            "_outputs/simulation/test_api_simulation_df-post_merge-result_fix.csv")

        ## uncomment to re-generate simulation results
        #simulation_results.data.to_dataframe().to_csv(path_expected_sim_res, index=False)

        # Test the reproducibility of simulate
        # round is necessary, writing and reading induces numerical errors of magnitude ~ 1e-13
        # BUT ON DIFFERENT MACHINE I CAN SEE ERROR OF MAGNITUDE 1e-5 !!!
        # TODO: Can we improve this??
        simulation_df = pd.read_csv(path_expected_sim_res)

        id_simulation_is_reproducible = simulation_df['ID'].equals(simulation_results.data.to_dataframe()['ID'])
        # Check ID before - str doesn't seem to work with numpy.allclose
        self.assertTrue(id_simulation_is_reproducible)

        round_decimal = 5
        simulation_is_reproducible = allclose(simulation_df.loc[:, simulation_df.columns != 'ID'].values,
                                              simulation_results.data.to_dataframe().
                                              loc[:, simulation_results.data.to_dataframe().columns != 'ID'].values,
                                              atol=10 ** (-round_decimal), rtol=10 ** (-round_decimal))
        # Use of numpy.allclose instead of pandas.testing.assert_frame_equal because of buggy behaviour reported
        # in https://github.com/pandas-dev/pandas/issues/22052

        # If reproducibility error > 1e-5 => display it + visit with the biggest reproducibility error
        error_message = ''
        if not simulation_is_reproducible:
            # simulation_df = pd.read_csv(
            #     os.path.join(test_data_dir, "_outputs/simulation/test_api_simulation_df-post_merge-result_fix.csv"))
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
