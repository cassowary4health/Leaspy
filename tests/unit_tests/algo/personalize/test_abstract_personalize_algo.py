from tests import allow_abstract_class_init
import unittest

from leaspy.algo.personalize.abstract_personalize_algo import AbstractPersonalizeAlgo
from leaspy.io.settings.algorithm_settings import AlgorithmSettings

from tests import allow_abstract_class_init

class AbstractPersonalizeAlgoTest(unittest.TestCase):

    @allow_abstract_class_init(AbstractPersonalizeAlgo)
    def test_constructor(self):
        settings = AlgorithmSettings('scipy_minimize')

        algo = AbstractPersonalizeAlgo(settings)
        self.assertEqual(algo.algo_parameters, {'n_iter': 100, 'n_jobs': 1, "use_jacobian":False, "progress_bar": False})
        self.assertEqual(algo.name, 'scipy_minimize')
        self.assertEqual(algo.seed, None)

