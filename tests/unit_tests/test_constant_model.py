import unittest
import pandas as pd
from tests import example_data_path
from leaspy import Data, Leaspy, AlgorithmSettings

from leaspy.models.constant_model import ConstantModel


class ConstantModelAPITest(unittest.TestCase):

    def test_run(self):
        # Data
        data = Data.from_csv_file(example_data_path)

        # Settings
        # The `prediction_type` could be `last`, `last_known`, `max` or `mean`
        settings = AlgorithmSettings('constant_prediction', prediction_type='last')

        self.assertEqual(settings.parameters['prediction_type'], 'last')

        # Leaspy
        lsp = Leaspy('constant')
        model = lsp.model
        self.assertIsInstance(model, ConstantModel)

        # Personalize
        ip = lsp.personalize(data, settings)

        self.assertListEqual(model.features, ['Y0','Y1','Y2','Y3'])
        self.assertEqual(model.dimension, 4)

        # Estimate
        timepoints = {'178': [30, 31]}
        results = lsp.estimate(timepoints, ip)
        self.assertEqual(results.keys(), {'178':0}.keys())
        self.assertEqual(results['178'].shape, (2, 4))
        for i, v in enumerate([0.73333, 0.0, 0.2, 0.4]):
            self.assertAlmostEqual(results['178'][0].tolist()[i], v, delta=10e-5)
            self.assertAlmostEqual(results['178'][1].tolist()[i], v, delta=10e-5)