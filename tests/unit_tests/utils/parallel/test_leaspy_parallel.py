import unittest

import numpy as np

from leaspy import AlgorithmSettings, Data, Leaspy
from leaspy.io.outputs.individual_parameters import IndividualParameters
from leaspy.utils.parallel import leaspy_parallel_calibrate, leaspy_parallel_personalize


class TestLeaspyParallel(unittest.TestCase):

    def setUp(self):
        super().setUp()

        self.n_runs = 9

        seed = 42
        np.random.seed(seed)

        p = 5  # pats number
        v = 3  # vis number
        f = 2  # feat number
        t = np.sort(np.random.rand(self.n_runs, p, v), axis=2)
        y = np.sort(np.random.rand(self.n_runs, p, v, f), axis=2)

        self.datas = [Data.from_individuals(np.arange(p), t[r, :], y[r, :],
                                            [f'feature_{i}' for i in range(f)]) for r in range(self.n_runs)]
        self.model_type = 'logistic'
        self.settings_algos_fit = [AlgorithmSettings('mcmc_saem', n_iter=10, seed=seed)] * self.n_runs
        self.settings_algos_perso = [AlgorithmSettings('mode_real', seed=seed)] * self.n_runs
        # self.src_dim = 1

    def test_fit_and_perso(self):

        def leaspy_factory(_):
            leaspy = Leaspy(self.model_type)
            # leaspy.model.load_hyperparameters({'source_dimension': self.src_dim}) # Optional
            return leaspy

        def leaspy_cb(leaspy, k):
            return leaspy, k

        def leaspy_res_cb(ips, k):
            return ips, k

        outs_fit_cb = leaspy_parallel_calibrate(self.datas, self.settings_algos_fit,
                                                leaspy_factory, leaspy_cb, verbose=50)

        self.assertEqual(len(outs_fit_cb), self.n_runs)
        for i, (leaspy, j) in enumerate(outs_fit_cb):
            with self.subTest(i=i):
                self.assertEqual(i, j)  # order outputs
                self.assertIsInstance(leaspy, Leaspy)  # leaspy object
                leaspy.check_if_initialized()  # raises an error otherwise

        leaspys = [out[0] for out in outs_fit_cb]  # iterable of leaspy fitted models
        outs_perso_cb = leaspy_parallel_personalize(leaspys, self.datas, self.settings_algos_perso,
                                                    leaspy_res_cb, verbose=50)

        self.assertEqual(len(outs_perso_cb), self.n_runs)
        for i, (ips, j) in enumerate(outs_perso_cb):
            with self.subTest(i=i):
                self.assertEqual(i, j)  # order outputs
                self.assertIsInstance(ips, IndividualParameters)  # leaspy IndividualParameters object
