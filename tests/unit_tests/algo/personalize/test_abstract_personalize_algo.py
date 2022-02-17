from leaspy.algo.personalize.abstract_personalize_algo import AbstractPersonalizeAlgo
from leaspy.io.settings.algorithm_settings import AlgorithmSettings

from tests import LeaspyTestCase


class AbstractPersonalizeAlgoTest(LeaspyTestCase):

    @LeaspyTestCase.allow_abstract_class_init(AbstractPersonalizeAlgo)
    def test_constructor(self):
        settings = AlgorithmSettings('scipy_minimize')

        AbstractPersonalizeAlgo.name = 'scipy_minimize'  # new logic this is now a class attribute...

        algo = AbstractPersonalizeAlgo(settings)
        self.assertEqual(algo.name, 'scipy_minimize')
        self.assertEqual(algo.seed, None)
        self.assertEqual(algo.family, 'personalize')

