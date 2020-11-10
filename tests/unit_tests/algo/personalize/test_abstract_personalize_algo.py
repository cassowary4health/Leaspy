import unittest

from leaspy.algo.personalize.abstract_personalize_algo import AbstractPersonalizeAlgo
from leaspy.io.settings.algorithm_settings import AlgorithmSettings


class AbstractPersonalizeAlgoTest(unittest.TestCase):

    def test_constructor(self):
        settings = AlgorithmSettings('scipy_minimize')

        algo = AbstractPersonalizeAlgo(settings)
        self.assertEqual(algo.algo_parameters, {'n_iter': 100, 'n_jobs': 1, "progress_bar": False})
        self.assertEqual(algo.name, 'scipy_minimize')
        self.assertEqual(algo.seed, None)

