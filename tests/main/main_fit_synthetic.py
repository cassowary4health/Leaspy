import os
from tests import test_data_dir
from src.main import Leaspy
from src.inputs.data_reader import DataReader
import unittest
from src.utils.data_generator import generate_data_from_model

from src.inputs.algo_settings import AlgoSettings


class LeaspyFitTest(unittest.TestCase):


    # With Smart Initialization
    def test_fit_univariatesigmoid_mcmcsaem_smartinitialization(self):

        # Path output
        path_output = '../output_leaspy/synthetic_data_validation/'
        if not os.path.exists(path_output):
            if not os.path.exists('../output_leaspy'):
                os.mkdir('../output_leaspy')
            os.mkdir(path_output)
        #path_output = None


        # Algorithm settings
        path_to_algo_parameters = os.path.join(test_data_dir, '_generate_data',
                                               "algorithm_settings.json")
        algo_settings = AlgoSettings(path_to_algo_parameters)
        algo_settings.output_path = path_output

        # Create the data
        path_to_model_parameters = os.path.join(test_data_dir, '_generate_data', 'model_parameters.json')
        leaspy_dummy = Leaspy.from_model_settings(path_to_model_parameters)
        data = generate_data_from_model(leaspy_dummy.model, n_patients=100)

        leaspy = Leaspy('univariate')
        leaspy.fit(data, algo_settings, seed=0)

        # Plot the convergence ???
        # TODO
