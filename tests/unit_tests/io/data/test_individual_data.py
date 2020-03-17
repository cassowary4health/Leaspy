import unittest

from leaspy.io.data.individual_data import IndividualData


class IndividualDataTest(unittest.TestCase):

    def test_constructor(self):
        data_int = IndividualData(1)
        self.assertEqual(data_int.idx, 1)
        self.assertEqual(data_int.timepoints, None)
        self.assertEqual(data_int.observations, None)
        self.assertEqual(data_int.individual_parameters, {})
        self.assertEqual(data_int.cofactors, {})

        data_float = IndividualData(1.2)
        self.assertEqual(data_float.idx, 1.2)

        data_string = IndividualData('test')
        self.assertEqual(data_string.idx, 'test')

    def test_add_observation(self):
        # Add first observation
        data = IndividualData('test')
        data.add_observation(70, [30])

        self.assertEqual(data.idx, 'test')
        self.assertEqual(data.individual_parameters, {})

        self.assertEqual(data.timepoints, [70])
        self.assertEqual(data.observations, [[30]])

        # Add second observation
        data.add_observation(80, [40])
        self.assertEqual(data.timepoints, [70, 80])
        self.assertEqual(data.observations, [[30], [40]])

        # Add third observation
        data.add_observation(75, [35])
        self.assertEqual(data.timepoints, [70, 75, 80])
        self.assertEqual(data.observations, [[30], [35], [40]])

        # Add individual parameter
        data.add_individual_parameters("xi", 0.02)
        self.assertEqual(data.individual_parameters, {'xi': 0.02})

        # Add cofactors
        data.add_cofactor('gender', 'male')
        self.assertEqual(data.cofactors, {'gender': 'male'})
