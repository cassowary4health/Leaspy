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

        # Unknown algo family
        with self.assertRaisesRegex(ValueError, 'family'):
            AlgoFactory.algo('unknown-family', settings)

    def test_get_class(self):
        algo_class = AlgoFactory.get_class('mcmc_saem')
        self.assertIs(algo_class, TensorMCMCSAEM)

        with self.assertRaisesRegex(ValueError, 'known algorithm'):
            AlgoFactory.get_class('unknown-algo')

    def test_loading_default_for_all_algos(self):
        # bit of a functional test
        for family, algos in AlgoFactory._algos.items():
            for algo_name, algo_class in algos.items():

                # test creation of algorithm with defaults
                algo_inst = AlgoFactory.algo(family, AlgorithmSettings(algo_name))
                self.assertIsInstance(algo_inst, algo_class, algo_name)

                # test get_class
                self.assertIs(AlgoFactory.get_class(algo_name), algo_class, algo_name)

    def test_auto_burn_in(self):

        for algo_name, algo_family in {'mcmc_saem': 'fit', 'mode_real': 'personalize', 'mean_real': 'personalize'}.items():
            with self.subTest(algo_name=algo_name):
                default_settings = AlgorithmSettings(algo_name)

                # get & check coherence of default parameters for those algos
                self.assertIsNone(default_settings.parameters['n_burn_in_iter'])
                default_n_iter = default_settings.parameters['n_iter']
                self.assertIsNotNone(default_n_iter)
                self.assertIsInstance(default_n_iter, int)
                self.assertTrue(default_n_iter > 0)
                self.assertEqual(default_n_iter % 100, 0)
                default_burn_in_frac = default_settings.parameters['n_burn_in_iter_frac']
                self.assertIsNotNone(default_burn_in_frac)
                self.assertTrue(0 < default_burn_in_frac < 1, default_burn_in_frac)
                self.assertAlmostEqual((default_burn_in_frac * 100) % 1, 0, places=8)

                # check behavior
                algo = AlgoFactory.algo(algo_family, default_settings)
                self.assertEqual(algo.algo_parameters['n_burn_in_iter'], int(default_burn_in_frac*default_n_iter))

                settings = AlgorithmSettings(algo_name, n_iter=2100)
                algo = AlgoFactory.algo(algo_family, settings)
                self.assertEqual(algo.algo_parameters['n_burn_in_iter'], int(default_burn_in_frac*2100))

                settings = AlgorithmSettings(algo_name, n_burn_in_iter_frac=.80001)
                algo = AlgoFactory.algo(algo_family, settings)
                self.assertEqual(algo.algo_parameters['n_burn_in_iter'], int(.8*default_n_iter))

                settings = AlgorithmSettings(algo_name, n_iter=1001, n_burn_in_iter_frac=.8)
                algo = AlgoFactory.algo(algo_family, settings)
                self.assertEqual(algo.algo_parameters['n_burn_in_iter'], 800)

                # priority case, with warning
                settings = AlgorithmSettings(algo_name, n_burn_in_iter=42)
                with self.assertWarns(FutureWarning):  # warn because n_burn_in_iter_frac is not None
                    algo = AlgoFactory.algo(algo_family, settings)
                self.assertEqual(algo.algo_parameters['n_burn_in_iter'], 42)

                # explicit `n_burn_in_iter_frac=None` (no warning)
                settings = AlgorithmSettings(algo_name, n_burn_in_iter=314, n_burn_in_iter_frac=None)
                algo = AlgoFactory.algo(algo_family, settings)
                self.assertEqual(algo.algo_parameters['n_burn_in_iter'], 314)

                # error case (both n_burn_in_iter_frac & n_burn_in_iter are None)
                settings = AlgorithmSettings(algo_name, n_burn_in_iter_frac=None)
                with self.assertRaises(ValueError):
                    AlgoFactory.algo(algo_family, settings)


    def test_auto_annealing(self):

        default_settings = AlgorithmSettings('mcmc_saem')
        default_n_iter = default_settings.parameters['n_iter']

        # get & check coherence of default parameters for those algos
        self.assertFalse(default_settings.parameters['annealing']['do_annealing'])
        self.assertIsNone(default_settings.parameters['annealing']['n_iter'])
        default_annealing_iter_frac = default_settings.parameters['annealing']['n_iter_frac']
        self.assertIsNotNone(default_annealing_iter_frac)
        self.assertTrue(0 < default_annealing_iter_frac < 1, default_annealing_iter_frac)
        self.assertAlmostEqual((default_annealing_iter_frac * 100) % 1, 0, places=8)

        settings = AlgorithmSettings('mcmc_saem', n_iter=1000)
        algo = AlgoFactory.algo('fit', settings)
        self.assertEqual(algo.algo_parameters['annealing']['n_iter'], None)

        # also test new partial dictionary behavior for annealing
        settings = AlgorithmSettings('mcmc_saem', n_iter=1001, annealing=dict(do_annealing=True))
        algo = AlgoFactory.algo('fit', settings)
        self.assertEqual(algo.algo_parameters['annealing']['n_iter'], int(default_annealing_iter_frac*1001))

        settings = AlgorithmSettings('mcmc_saem', annealing=dict(do_annealing=True, n_iter_frac=.40001))
        algo = AlgoFactory.algo('fit', settings)
        self.assertEqual(algo.algo_parameters['annealing']['n_iter'], int(default_n_iter*.4))

        settings = AlgorithmSettings('mcmc_saem', n_iter=1000, annealing=dict(do_annealing=True, n_iter_frac=.3))
        algo = AlgoFactory.algo('fit', settings)
        self.assertEqual(algo.algo_parameters['annealing']['n_iter'], 300)

        # priority case, with warning
        settings = AlgorithmSettings('mcmc_saem', annealing=dict(do_annealing=True, n_iter=42))
        with self.assertWarns(FutureWarning):  # warn because annealing.n_iter_frac is not None
            algo = AlgoFactory.algo('fit', settings)
        self.assertEqual(algo.algo_parameters['annealing']['n_iter'], 42)

        # explicit `n_burn_in_iter_frac=None` (no warning)
        settings = AlgorithmSettings('mcmc_saem', annealing=dict(do_annealing=True, n_iter=314, n_iter_frac=None))
        algo = AlgoFactory.algo('fit', settings)
        self.assertEqual(algo.algo_parameters['annealing']['n_iter'], 314)

        # error case (both n_burn_in_iter_frac & n_burn_in_iter are None)
        settings = AlgorithmSettings('mcmc_saem', annealing=dict(do_annealing=True, n_iter_frac=None))
        with self.assertRaises(ValueError):
            AlgoFactory.algo('fit', settings)
