import torch

from tests import LeaspyTestCase


class TestMultivariateModelOrdinal(LeaspyTestCase):

    def test_reload_model(self):

        model = self.get_hardcoded_model('logistic_ordinal').model

        self.assertEqual(model.noise_model, 'ordinal')

        ordinal_mask = model.ordinal_infos.pop('mask')  # test after
        self.assertEqual(model.ordinal_infos, {
                'batch_deltas': False,
                'features': [{'max_level': 3, 'name': 'Y0'},
                             {'max_level': 4, 'name': 'Y1'},
                             {'max_level': 6, 'name': 'Y2'},
                             {'max_level': 10, 'name': 'Y3'}],
                'max_level': 10,
        })

        expected_mask = torch.tensor([[[
            [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
        ]]])
        self.assertTrue(torch.eq(ordinal_mask, expected_mask).all())  # not approximate

    def test_reload_model_batched(self):

        model = self.get_hardcoded_model('logistic_ordinal_b').model

        self.assertEqual(model.noise_model, 'ordinal')

        ordinal_mask = model.ordinal_infos.pop('mask')  # test after
        self.assertEqual(model.ordinal_infos, {
                'batch_deltas': True,
                'features': [{'max_level': 3, 'name': 'Y0'},
                             {'max_level': 4, 'name': 'Y1'},
                             {'max_level': 6, 'name': 'Y2'},
                             {'max_level': 10, 'name': 'Y3'}],
                'max_level': 10,
        })

        expected_mask = torch.tensor([[[
            [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
        ]]])
        self.assertTrue(torch.eq(ordinal_mask, expected_mask).all())  # not approximate
