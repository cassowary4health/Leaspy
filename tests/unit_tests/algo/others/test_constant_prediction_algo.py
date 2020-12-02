import unittest
import pandas as pd

from leaspy.io.data.data import Data
from leaspy.io.data.dataset import Dataset
from leaspy.io.settings.algorithm_settings import AlgorithmSettings
from leaspy import Leaspy
from leaspy.algo.others.constant_prediction_algo import ConstantPredictionAlgorithm


class ConstantPredictionAlgorithmTest(unittest.TestCase):

    def setUp(self):
        arr = [
            ['1', 1., 2., 1.],
            ['1', 2., 4., 3.],
            ['1', 3., 3., float('nan')]
        ]

        df = pd.DataFrame(data=arr, columns=['ID', 'TIME', 'A', 'B']).set_index(['ID', 'TIME'])
        data = Data.from_dataframe(df)
        self.dataset = Dataset(data)

    def test_constructor(self):
        settings = AlgorithmSettings('constant_prediction')
        algo = ConstantPredictionAlgorithm(settings)
        self.assertEqual(algo.name, 'constant_prediction')
        self.assertEqual(algo.prediction_type, 'last')

        for prediction_type in ['last', 'last_known', 'max', 'mean']:
            settings = AlgorithmSettings('constant_prediction', prediction_type=prediction_type)
            algo = ConstantPredictionAlgorithm(settings)
            self.assertEqual(algo.prediction_type, prediction_type)

    def test_get_individual_last_values(self):
        times = [31, 32, 34, 33]
        values = [
            [1., 0.5],
            [2., 0.5],
            [float('nan'), 2.],
            [3., float('nan')]
        ]

        results = [
            ('last', {'A': float('nan'), 'B': 2.}),
            ('last_known', {'A': 3., 'B': 2.}),
            ('max', {'A': 3., 'B': 2.}),
            ('mean', {'A': 2., 'B': 1.})
        ]

        for (prediction_type, res) in results:
            settings = AlgorithmSettings('constant_prediction', prediction_type=prediction_type)
            algo = ConstantPredictionAlgorithm(settings)
            algo.features = ['A', 'B']
            ind_ip = algo._get_individual_last_values(times, values)

            if ind_ip['A'] == ind_ip['A'] and ind_ip['B'] == ind_ip['B']:
                self.assertDictEqual(ind_ip, res)

            elif ind_ip['A'] != ind_ip['A']:
                self.assertTrue(res['A'] != res['A'])
                self.assertTrue(ind_ip['B'] == res['B'])

    def test_run_last(self):
        results = [
            ('last', {'1': {'A': 3., 'B': float('nan')}}),
            ('last_known', {'1': {'A': 3, 'B': 3.}}),
            ('max', {'1': {'A': 4., 'B': 3.}}),
            ('mean', {'1': {'A': 3., 'B': 2.}}),
        ]

        for (pred_type, res) in results:

            settings = AlgorithmSettings('constant_prediction', prediction_type=pred_type)
            algo = ConstantPredictionAlgorithm(settings)
            model = Leaspy('constant')

            ip, noise = algo.run(model, self.dataset)
            self.assertEqual(noise, 0.)
            self.assertListEqual(ip._indices, ['1'])
            self.assertDictEqual(ip._parameters_shape, {'A': (), 'B': ()})
            self.assertEqual(ip._default_saving_type, 'csv')

            dict_ip = ip._individual_parameters

            if pred_type == 'last':
                self.assertEqual(dict_ip.keys(), {'1': 0}.keys())
                self.assertEqual(dict_ip['1']['A'], 3.)
                self.assertTrue(dict_ip['1']['B'] != dict_ip['1']['B'])
            else:
                self.assertDictEqual(ip._individual_parameters, res)
