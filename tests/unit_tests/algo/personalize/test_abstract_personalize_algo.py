import unittest

from leaspy.algo.personalize.abstract_personalize_algo import AbstractPersonalizeAlgo
from leaspy.inputs.settings.algorithm_settings import AlgorithmSettings


class AbstractPersonalizeAlgoTest(unittest.TestCase):

    def test_constructor(self):
        settings = AlgorithmSettings('scipy_minimize')

        algo = AbstractPersonalizeAlgo(settings)
        self.assertEqual(algo.algo_parameters, {'n_iter': 100, 'n_jobs': -1, 'parallel': False,
                                                'regularity_method': 'prior',
                                                'regularity_weight': 1, 'attachment_weight': 1})
        self.assertEqual(algo.name, 'scipy_minimize')
        self.assertEqual(algo.seed, None)

