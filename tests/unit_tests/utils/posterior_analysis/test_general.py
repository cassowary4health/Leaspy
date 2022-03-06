import os
import unittest
import numpy as np
import pandas as pd
#import torch

from leaspy import Leaspy, IndividualParameters#, Data, AlgorithmSettings
from leaspy.utils.posterior_analysis.general import append_spaceshifts_to_individual_parameters_dataframe, get_reparametrized_ages, compute_trajectory_of_population
from tests import test_data_dir, hardcoded_model_path

class TestUtilsGeneral(unittest.TestCase):

    def test_append_spaceshifts_to_individual_parameters_dataframe(self):
        df = pd.DataFrame(data=[[0.1, 70, 0.1, -0.3], [0.2, 73, -0.4, 0.1], [0.3, 58, -0.6, 0.2]],
                          index=["idx1", "idx2", "idx3"],
                          columns=["xi", "tau", "sources_0", "sources_1"])

        leaspy = Leaspy.load(hardcoded_model_path('logistic'))

        df_w = append_spaceshifts_to_individual_parameters_dataframe(df, leaspy)
        #TODO : the above test just check  that it runs, not the results!

    def test_get_reparametrized_ages(self):
        leaspy = Leaspy.load(hardcoded_model_path('logistic'))
        ip = IndividualParameters.load(os.path.join(test_data_dir, 'leaspy_io', 'outputs', 'ip_save.json'))

        ages = {'idx1': [70, 80], 'idx3': [100]}
        reparametrized_ages = get_reparametrized_ages(ages, ip, leaspy)

        self.assertEqual(reparametrized_ages.keys(), ages.keys())
        self.assertTrue(np.allclose(reparametrized_ages['idx1'], [75.2, 86.2517]))
        self.assertTrue(np.allclose(reparametrized_ages['idx3'], [131.8941]))

    def test_compute_trajectory_of_population(self):
        leaspy = Leaspy.load(hardcoded_model_path('logistic'))
        ip = IndividualParameters.load(os.path.join(test_data_dir, 'leaspy_io', 'outputs', 'ip_save.json'))

        timepoints = [70, 71, 72, 73, 74, 75, 76]

        trajectory = compute_trajectory_of_population(timepoints, ip, leaspy)
        #self.assertTrue(torch.is_tensor(trajectory))
        # TODO : choose a convention for output type : Numpy or Torch ? For now it seems numpy in api.estimate
        self.assertEqual(trajectory.shape[0], 7)
        self.assertEqual(trajectory.shape[1], 4)
        # TODO : the above test just check that it runs, not the results!