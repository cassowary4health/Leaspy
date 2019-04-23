import unittest

from src.inputs.individual_data import IndividualData
from src.inputs.data import Data


class DataTest(unittest.TestCase):

    def test_constructor(self):
        data = Data()
        self.assertEqual(data.indices, [])
        self.assertEqual(data.individuals, {})

    def test_add_individual(self):
        individual = IndividualData('idx')
        individual.add_observation(70, [1])

        data = Data()
        data.add_individual(individual)
        self.assertEqual(data.indices, ['idx'])
        self.assertEqual(data.individuals, {'idx' : individual})
        self.assertEqual(data.individuals['idx'].timepoints, [70])

        individual.add_observation(80, [2])
        self.assertEqual(data.individuals['idx'].timepoints, [70, 80])
        self.assertEqual(data.individuals['idx'].raw_observations, [[1], [2]])
