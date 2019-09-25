import unittest

from leaspy.inputs.settings.algorithm_settings import AlgorithmSettings
from leaspy.algo.simulate.simulate import SimulationAlgorithm


class SimulationAlgorithmTest(unittest.TestCase):

    def setUp(self):
        self.settings = AlgorithmSettings('simulation')
        self.algo = SimulationAlgorithm(self.settings)

    def test_construtor(self):
        self.assertEqual(self.settings.parameters['bandwidth_method'], self.algo.bandwidth_method)
        self.assertEqual(self.settings.parameters['noise'], self.algo.noise)
        self.assertEqual(self.settings.parameters['number_of_subjects'], self.algo.number_of_subjects)
        self.assertEqual(self.settings.parameters['mean_number_of_visits'], self.algo.mean_number_of_visits)
        self.assertEqual(self.settings.parameters['std_number_of_visits'], self.algo.std_number_of_visits)

    def test_get_number_of_visits(self):
        n_visit = self.algo._get_number_of_visits()
        self.assertTrue(type(n_visit) == int)
        self.assertTrue(n_visit >= 1)
