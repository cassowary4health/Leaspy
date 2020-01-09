import unittest

from leaspy import Leaspy, Data, AlgorithmSettings
from leaspy.inputs.data.result import Result

from leaspy.utils.parallel import leaspy_parallel_calibrate, leaspy_parallel_personalize

import numpy as np

class TestLeaspyParallel(unittest.TestCase):

    def setUp(self):
        super().setUp()

        self.n_runs = 9

        self.model_type = 'logistic'
        #self.src_dim = 1

        seed = 42
        np.random.seed(seed)

        self.fit_algos_settings = [AlgorithmSettings('mcmc_saem', n_iter=10, seed=seed) for r in range(self.n_runs)]
        self.perso_algos_settings = [AlgorithmSettings('mode_real', seed=seed) for r in range(self.n_runs)]

        p = 5 # pats number
        v = 3 # vis number
        f = 2 # feat number
        t = np.sort(np.random.rand(self.n_runs, p, v), axis=2)
        y = np.sort(np.random.rand(self.n_runs, p, v, f), axis=2)

        self.datas = [Data.from_individuals(np.arange(p),t[r,:],y[r,:],[f'feature_{i}' for i in range(f)]) for r in range(self.n_runs)]

    def test_fit_and_perso(self):

        def leaspy_factory(i):
            leaspy = Leaspy(self.model_type)
            #leaspy.model.load_hyperparameters({'source_dimension': self.src_dim}) # Optional
            return leaspy

        def leaspy_cb(leaspy, i):
            return (leaspy, i)

        def leaspy_res_cb(res, i):
            return (res, i)

        outs_fit_cb = leaspy_parallel_calibrate(self.datas, self.fit_algos_settings, leaspy_factory, leaspy_cb, verbose=50)

        self.assertEqual(len(outs_fit_cb), self.n_runs)
        for i, (leaspy, j) in enumerate(outs_fit_cb):
            with self.subTest(i=i):
                self.assertEqual(i,j) # order outputs
                self.assertIsInstance(leaspy, Leaspy) # leaspy object
                leaspy.check_if_initialized() # raises an error otherwise

        leaspys = [out[0] for out in outs_fit_cb] # iterable of leaspy fitted models
        outs_perso_cb = leaspy_parallel_personalize(leaspys, self.datas, self.perso_algos_settings, leaspy_res_cb, verbose=50)

        self.assertEqual(len(outs_perso_cb), self.n_runs)
        for i, (res, j) in enumerate(outs_perso_cb):
            with self.subTest(i=i):
                self.assertEqual(i,j) # order outputs
                self.assertIsInstance(res, Result) # leaspy object

        return 0
