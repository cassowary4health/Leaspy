# from os import path
# from subprocess import Popen
import unittest

import torch.random

from leaspy.algo.abstract_algo import AbstractAlgo
# from leaspy.leaspy_io.settings.algorithm_settings import AlgorithmSettings
# from leaspy.utils.logs.fit_output_manager import FitOutputManager

# from tests import test_data_dir
from tests import allow_abstract_class_init


class Test(unittest.TestCase):

    @allow_abstract_class_init(AbstractAlgo)
    def test_constructor(self):
        algo = AbstractAlgo()
        self.assertEqual(algo.algo_parameters, None)
        self.assertEqual(algo.name, None)
        self.assertEqual(algo.output_manager, None)
        self.assertEqual(algo.seed, None)

    @allow_abstract_class_init(AbstractAlgo)
    def test_initialize_seed(self):
        algo = AbstractAlgo()
        seed = torch.randint(10000, (1,)).item()
        algo._initialize_seed(seed)
        self.assertEqual(seed, torch.random.initial_seed())

    @allow_abstract_class_init(AbstractAlgo)
    def test_load_parameters(self):
        algo = AbstractAlgo()
        algo.algo_parameters = {'param1': 1, 'param2': 2}
        parameters = {'param1': 10, 'param3': 3}
        algo.load_parameters(parameters)
        self.assertEqual(list(algo.algo_parameters.keys()), ['param1', 'param2', 'param3'])
        self.assertEqual(algo.algo_parameters['param1'], 10)
        self.assertEqual(algo.algo_parameters['param2'], 2)
        self.assertEqual(algo.algo_parameters['param3'], 3)

    @allow_abstract_class_init(AbstractAlgo)
    def test_set_output_manager(self):
        algo = AbstractAlgo()
        algo.set_output_manager(None)
        self.assertEqual(algo.output_manager, None)

        # TODO: capture question & answer yes automatically (use subprocess.Popen.communicate ?) or remove interactive mode
        # settings = AlgorithmSettings('mcmc_saem')
        # output_path = path.join(test_data_dir, "_outputs", 'output_manager')
        # settings.set_logs(output_path)
        # algo.set_output_manager(settings.logs)
        # self.assertTrue(type(algo.output_manager) == FitOutputManager)
