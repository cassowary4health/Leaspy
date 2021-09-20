import os
import unittest

import numpy as np
import pandas as pd
import torch

from leaspy import AlgorithmSettings, Data, Leaspy
from leaspy.algo.simulate.simulate import SimulationAlgorithm
from leaspy.io.outputs.result import Result
from tests import example_data_path, test_data_dir, hardcoded_model_path


class SimulationAlgorithmTest(unittest.TestCase):

    def setUp(self):
        self.settings = AlgorithmSettings('simulation')
        self.algo = SimulationAlgorithm(self.settings)

        # reused data, model, individual parameters
        self.data = Data.from_csv_file(example_data_path)
        cofactors = pd.read_csv(os.path.join(test_data_dir, "io/data/data_tiny_covariate.csv"))
        cofactors.columns = ("ID", "Treatments")
        cofactors['ID'] = cofactors['ID'].astype(str)
        cofactors = cofactors.set_index("ID")
        self.data.load_cofactors(cofactors, ["Treatments"])

        self.model = Leaspy.load(hardcoded_model_path('logistic'))
        perso_settings = AlgorithmSettings('mode_real')
        self.individual_parameters = self.model.personalize(self.data, perso_settings)

    def test_construtor(self):
        """
        Test the initialization.
        """
        self.assertEqual(self.settings.parameters['bandwidth_method'], self.algo.bandwidth_method)
        self.assertEqual(self.settings.parameters['noise'], self.algo.noise)
        self.assertEqual(self.settings.parameters['number_of_subjects'], self.algo.number_of_subjects)
        self.assertEqual(self.settings.parameters['mean_number_of_visits'], self.algo.mean_number_of_visits)
        self.assertEqual(self.settings.parameters['std_number_of_visits'], self.algo.std_number_of_visits)
        self.assertEqual(self.settings.parameters['cofactor'], self.algo.cofactor)
        self.assertEqual(self.settings.parameters['cofactor_state'], self.algo.cofactor_state)

        settings = AlgorithmSettings('simulation', sources_method="dummy")
        self.assertRaises(ValueError, SimulationAlgorithm, settings)

    def test_get_number_of_visits(self):
        n_visit = self.algo._get_number_of_visits()
        self.assertTrue(type(n_visit) == int)
        self.assertTrue(n_visit >= 1)

    def test_get_mean_and_covariance_matrix(self):
        """
        Test the result given by the calculus with torch vs the dedicated function of numpy.
        """
        values = np.random.rand(100, 5)
        t_mean = torch.tensor(values).mean(dim=0)
        self.assertTrue(np.allclose(values.mean(axis=0),
                                    t_mean.numpy()))
        t_cov = torch.tensor(values) - t_mean[None, :]
        t_cov = 1. / (t_cov.size(0) - 1) * t_cov.t() @ t_cov
        self.assertTrue(np.allclose(np.cov(values.T),
                                    t_cov.numpy()))

    def test_check_cofactors(self):
        """
        Test Leaspy.simulate return a ``ValueError`` if the ``cofactor`` and ``cofactor_state`` parameters given
        in the ``AlgorithmSettings`` are invalid.
        """
        model, individual_parameters, data = self.model, self.individual_parameters, self.data

        # cofactor not None but cofactor_state None...
        settings = AlgorithmSettings('simulation', cofactor=["Treatments"])
        self.assertRaises(ValueError, model.simulate, individual_parameters, data, settings)

        # bad type for cofactor / cofactor state
        settings = AlgorithmSettings('simulation', cofactor="Treatments", cofactor_state=["Treatment_A"])
        self.assertRaises(AssertionError, model.simulate, individual_parameters, data, settings)

        settings = AlgorithmSettings('simulation', cofactor=["Treatments"], cofactor_state="Treatment_A")
        self.assertRaises(AssertionError, model.simulate, individual_parameters, data, settings)

        # bad length for cofactor_state
        settings = AlgorithmSettings('simulation', cofactor=["Treatments"], cofactor_state=["Treatment_A", "Treatment_B"])
        self.assertRaises(AssertionError, model.simulate, individual_parameters, data, settings)

        # invalid cofactor name
        settings = AlgorithmSettings('simulation', cofactor=["dummy"], cofactor_state=["dummy"])
        self.assertRaises(ValueError, model.simulate, individual_parameters, data, settings)

        # invalid cofactor state
        settings = AlgorithmSettings('simulation', cofactor=["Treatments"], cofactor_state=["dummy"])
        self.assertRaises(ValueError, model.simulate, individual_parameters, data, settings)


    def test_simulation_run(self):
        """
        Test if the simulation run properly with different settings.
        """
        leaspy_session, individual_parameters, data = self.model, self.individual_parameters, self.data

        settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=1000, mean_number_of_visits=3,
                                     std_number_of_visits=0, sources_method="full_kde", bandwidth_method=.2)
        new_results = leaspy_session.simulate(individual_parameters, data, settings)  # just test if run without error

        settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=1000, mean_number_of_visits=3,
                                     std_number_of_visits=0, sources_method="normal_sources", bandwidth_method=.2)
        new_results = leaspy_session.simulate(individual_parameters, data, settings)  # just test if run without error

        settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=1000, mean_number_of_visits=3,
                                     std_number_of_visits=0, sources_method="full_kde", bandwidth_method=.2,
                                     features_bounds=True)  # idem + test scores bounds
        # self.test_bounds_behaviour(leaspy_session, results, settings)

        bounds = {'Y0': (0., .5), 'Y1': (0., .1), 'Y2': (0., .1), 'Y3': (0., .1)}
        settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=1000, mean_number_of_visits=3,
                                     std_number_of_visits=0, sources_method="full_kde", bandwidth_method=.2,
                                     features_bounds=bounds)  # idem + test scores bounds
        self._bounds_behaviour(leaspy_session, individual_parameters, data, settings)

        settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=200, mean_number_of_visits=3,
                                     std_number_of_visits=0, sources_method="full_kde", bandwidth_method=.2,
                                     reparametrized_age_bounds=(65, 75))
        new_results = leaspy_session.simulate(individual_parameters, data, settings)  # just test if run without error
        # Test if the reparametrized ages are within (65, 75) up to a tolerance of 2.
        repam_age = new_results.data.to_dataframe().groupby('ID').first()['TIME'].values
        repam_age -= new_results.individual_parameters['tau'].squeeze().numpy()
        repam_age *= np.exp(new_results.individual_parameters['xi'].squeeze().numpy())
        repam_age += leaspy_session.model.parameters['tau_mean'].item()
        self.assertTrue(all(repam_age > 63) & all(repam_age < 77))

    def test_simulation_cofactors_run(self):
        """
        Test if the simulation run properly with different settings (no result check, only unit test).
        """
        leaspy_session, individual_parameters, data = self.model, self.individual_parameters, self.data

        settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=1000, mean_number_of_visits=3,
                                     std_number_of_visits=0, sources_method="full_kde", bandwidth_method=.2,
                                     cofactor=['Treatments'], cofactor_state=['Treatment_A'])
        leaspy_session.simulate(individual_parameters, data, settings)  # just test if run without error


    def _bounds_behaviour(self, leaspy_session, individual_parameters, data, settings):
        """
        Test the good behaviour of the ``features_bounds`` parameter.

        Parameters
        ----------
        leaspy_session : :class:`.Leaspy`
        results : :class:`~.io.outputs.result.Result`
        settings : :class:`.AlgorithmSettings`
            Contains the ``features_bounds`` parameter.
        """

        results = Result(data, individual_parameters.to_pytorch()[1])

        new_results = leaspy_session.simulate(individual_parameters, data, settings)
        new_results_max_bounds: np.ndarray = new_results.data.to_dataframe().groupby('ID').first().max().values[1:]
        new_results_min_bounds: np.ndarray = new_results.data.to_dataframe().groupby('ID').first().min().values[1:]

        if type(settings.parameters['features_bounds']) == dict:
            results_max_bounds = np.array([val[1] for val in settings.parameters["features_bounds"].values()])
            results_min_bounds = np.array([val[0] for val in settings.parameters["features_bounds"].values()])
        elif settings.parameters['features_bounds']:
            results_max_bounds: np.ndarray = results.data.to_dataframe().groupby('ID').first().max().values[1:]
            results_min_bounds: np.ndarray = results.data.to_dataframe().groupby('ID').first().min().values[1:]
        else:
            raise ValueError('features_bounds is not defined')

        self.assertTrue(all(new_results_max_bounds <= results_max_bounds),
                        "Generated scores contain scores outside the bounds")
        self.assertTrue(all(new_results_min_bounds >= results_min_bounds),
                        "Generated scores contain scores outside the bounds")

    def test_simulate_univariate(self):
        from leaspy import AlgorithmSettings, Data
        from leaspy.datasets import Loader

        putamen_df = Loader.load_dataset('parkinson-putamen-train_and_test')
        data = Data.from_dataframe(putamen_df.xs('train', level='SPLIT'))
        leaspy_logistic = Loader.load_leaspy_instance('parkinson-putamen-train')
        individual_parameters = Loader.load_individual_parameters('parkinson-putamen-train')

        simulation_settings = AlgorithmSettings('simulation', seed=0)
        simulated_data = leaspy_logistic.simulate(individual_parameters, data, simulation_settings)

        simu_df = simulated_data.data.to_dataframe()
        self.assertEqual(['ID', 'TIME', 'PUTAMEN'], list(simu_df.columns))
        simu_df.set_index('ID', inplace=True)
        self.assertTrue(list(simu_df.dtypes) == ['float64']*2)
        self.assertTrue(all(simu_df['PUTAMEN'].values <= 1))
        self.assertTrue(all(simu_df['PUTAMEN'].values >= 0))
        self.assertTrue(all(simu_df['TIME'].values <= 150))
        self.assertTrue(all(simu_df['TIME'].values >= 10))

    # global behaviour of SimulationAlgorithm class is tested in the functional test test_api.py
