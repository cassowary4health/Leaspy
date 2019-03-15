import unittest
import numpy as np
import torch

from src.inputs.individual_data import IndividualData


class IndividualDataTest(unittest.TestCase):

    def test_constructor(self):
        data_int = IndividualData(1)
        self.assertEqual(data_int.idx, 1)
        self.assertEqual(data_int.individual_parameters, None)
        self.assertEqual(data_int.timepoints, None)
        self.assertEqual(data_int.raw_observations, None)
        self.assertEqual(data_int.tensor_observations, None)

        data_float = IndividualData(1.2)
        self.assertEqual(data_float.idx, 1.2)

        data_string = IndividualData('test')
        self.assertEqual(data_string.idx, 'test')

    def test_add_observation(self):
        ### Add first observation
        data = IndividualData('test')
        data.add_observation(70, 30)

        self.assertEqual(data.idx, 'test')
        self.assertEqual(data.individual_parameters, None)

        self.assertEqual(data.timepoints, [70])
        self.assertEqual(data.raw_observations, [30])
        self.assertEqual(data.tensor_observations, torch.from_numpy(np.array([30])).float())

        ### Add second observation
        data.add_observation(80, 40)
        self.assertEqual(data.timepoints, [70, 80])
        self.assertEqual(data.raw_observations, [30, 40])

        ### Add third observation
        data.add_observation(75, 35)
        self.assertEqual(data.timepoints, [70, 75, 80])
        self.assertEqual(data.raw_observations, [30, 35, 40])