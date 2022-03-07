from torch import tensor, allclose

from leaspy.datasets.loader import Loader

from tests import LeaspyTestCase

# TODO: regenerate example models + individual parameters

class LoaderTest(LeaspyTestCase):

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
            self.assertIn(df.index.get_level_values('TIME').dtype, ('float64', 'float32'))

    def test_load_leaspy_instance(self):
        """
        Check that all models are loadable, and check parameter values for one model.
        """
        self.assertEqual(list(Loader().model_paths.keys()), ['alzheimer-multivariate', 'parkinson-multivariate', 'parkinson-putamen-train'])

        for name in Loader().model_paths.keys():
            leaspy_instance = Loader.load_leaspy_instance(name)
            if 'multivariate' in name:
                self.assertEqual(leaspy_instance.type, 'logistic')
            else:
                self.assertEqual(leaspy_instance.type, 'univariate_logistic')

        leaspy_instance = Loader.load_leaspy_instance('parkinson-putamen-train')
        self.assertEqual(leaspy_instance.model.features, ['PUTAMEN'])
        self.assertEqual(leaspy_instance.model.noise_model, 'gaussian_scalar')

        parameters = {"g": tensor([-0.7901085019111633]),
                      "tau_mean": tensor(64.18125915527344),
                      "tau_std": tensor(10.199116706848145),
                      "xi_mean": tensor(-2.346343994140625),
                      "xi_std": tensor(0.5663877129554749),
                      "noise_std": tensor(0.021229960024356842)}
        self.assertEqual(leaspy_instance.model.parameters, parameters)

    def test_load_individual_parameters(self):
        """
        Check that all ips are loadable, and check values for one individual_parameters
        instance.
        """
        self.assertEqual(list(Loader().ip_paths.keys()), ['alzheimer-multivariate', 'parkinson-multivariate', 'parkinson-putamen-train'])

        for name in Loader().ip_paths.keys():
            ip = Loader.load_individual_parameters(name)

        ip = Loader.load_individual_parameters('alzheimer-multivariate')

        self.assertAlmostEqual(ip.get_mean('tau'), 76.9612791442871)
        self.assertAlmostEqual(ip.get_mean('xi'), 0.0629326763143763)
        self.assertAllClose(ip.get_mean('sources'),
                            [0.003150840562302619, -0.02109330625506118], what='sources.mean')
