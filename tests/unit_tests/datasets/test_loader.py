import unittest

from torch import tensor

from leaspy.datasets.loader import Loader


class DataTest(unittest.TestCase):

    def test_load_dataset(self):
        """
        Check ID and dtype of ID, TIME and values.
        """
        self.assertEqual(list(Loader().data_paths.keys()),
                         ['alzheimer-multivariate', 'parkinson-multivariate',
                          'parkinson-putamen', 'parkinson-putamen-train_and_test'])
        for name in Loader().data_paths.keys():
            df = Loader.load_dataset(name)
            if 'train_and_test' in name:
                self.assertEqual(df.index.names, ['ID', 'TIME', 'SPLIT'])
            else:
                self.assertEqual(df.index.names, ['ID', 'TIME'])
            self.assertTrue(all(df.dtypes.values == 'float64'))
            self.assertEqual(df.index.get_level_values('ID').unique().tolist(),
                             ['GS-' + '0'*(3 - len(str(i))) + str(i) for i in range(1, 201)])
            self.assertTrue(df.index.get_level_values('TIME').dtype in ('float64', 'float32'))

    def test_load_leaspy_instance(self):
        """
        Check ID and dtype of ID, TIME and values.
        """
        self.assertEqual(list(Loader().model_paths.keys()), ['parkinson-putamen-train'])

        leaspy_instance = Loader.load_leaspy_instance('parkinson-putamen-train')
        self.assertEqual(leaspy_instance.type, 'univariate_logistic')
        self.assertEqual(leaspy_instance.model.features, ['PUTAMEN'])
        self.assertEqual(leaspy_instance.model.loss, 'MSE')

        parameters = {"g": tensor([-0.7901085019111633]),
                      "tau_mean": tensor(64.18125915527344),
                      "tau_std": tensor(10.199116706848145),
                      "xi_mean": tensor(-2.346343994140625),
                      "xi_std": tensor(0.5663877129554749),
                      "noise_std": tensor(0.021229960024356842)}
        self.assertEqual(leaspy_instance.model.parameters, parameters)

    # TODO: add test for other methods
