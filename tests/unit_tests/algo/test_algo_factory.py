from leaspy.algo.algo_factory import AlgoFactory
from leaspy.io.settings.algorithm_settings import AlgorithmSettings
from leaspy.algo.fit.tensor_mcmcsaem import TensorMCMCSAEM

from tests import LeaspyTestCase


class TestAlgoFactory(LeaspyTestCase):

    def test_algo(self):
        """Test attributes static method"""
        # Test for one name
        settings = AlgorithmSettings('mcmc_saem')
        algo = AlgoFactory.algo('fit', settings)
        self.assertIsInstance(algo, TensorMCMCSAEM)

        # Test if raise ValueError if wrong string arg for name
        wrong_arg_exemples = ['mcmc', 'blabla']
        for wrong_arg in wrong_arg_exemples:
            settings.name = wrong_arg
            self.assertRaises(ValueError, AlgoFactory.algo, 'fit', settings)

    def test_get_class(self):
        algo_class = AlgoFactory.get_class('mcmc_saem')
        self.assertIs(algo_class, TensorMCMCSAEM)

    def test_loading_default_for_all_algos(self):
        # bit of a functional test
        for family, algos in AlgoFactory._algos.items():
            for algo_name, algo_class in algos.items():

                # test creation of algorithm with defaults
                algo_inst = AlgoFactory.algo(family, AlgorithmSettings(algo_name))
                self.assertIsInstance(algo_inst, algo_class, algo_name)

                # test get_class
                self.assertIs(AlgoFactory.get_class(algo_name), algo_class, algo_name)
