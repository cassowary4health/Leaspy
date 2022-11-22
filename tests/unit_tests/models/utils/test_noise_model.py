from typing import List, Tuple

import torch

from leaspy.models.utils.noise_struct import NoiseStruct, NOISE_STRUCTS, MultinomialDistribution
from leaspy.models.utils.noise_model import NoiseModel
from leaspy.io.data.data import Data
from leaspy.io.data.dataset import Dataset

from tests import LeaspyTestCase


def torch_broadcast_shapes(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    """Placeholder for torch.broadcast_shapes which is only available in Pytorch >= 1.8"""
    if hasattr(torch, 'broadcast_shapes'):
        return torch.broadcast_shapes(*shapes)
    else:
        return torch.broadcast_tensors(*map(torch.empty, shapes))[0].shape


class FakeModel:

    MOCK_NOISE_STD_SCALAR = torch.tensor([.05])
    MOCK_NOISE_STD_DIAG = torch.tensor([.03, .07, .02, .081293])

    def __init__(self, features: List[str], noise_model: str, noise_std = None):
        self.noise_model = noise_model
        self.features = features
        self.dimension = len(features)
        self.parameters = {}
        if noise_std is not None:
            if not isinstance(noise_std, torch.Tensor):
                noise_std = torch.tensor(noise_std)
            self.parameters = {
                'noise_std': noise_std.view(-1)
            }

    def _kws(self, kw_scale: str, mode: str = 'model'):
        # helper to ease tests
        kws = {}

        if 'gaussian' in self.noise_model:
            if mode == 'model':
                kws[kw_scale] = self.parameters['noise_std']
            else:
                # mock values
                if self.noise_model == 'gaussian_scalar':
                    kws[kw_scale] = self.MOCK_NOISE_STD_SCALAR
                else:
                    kws[kw_scale] = self.MOCK_NOISE_STD_DIAG

        return kws

    # useful for to test `noise_std_from_data`
    def compute_sum_squared_tensorized(self, dataset: Dataset, individual_params):
        # shape (n_individuals,)
        s = dataset.n_observations * (self.MOCK_NOISE_STD_SCALAR ** 2).view(-1) # 1D
        assert s.shape == (1,)
        return s

    def compute_sum_squared_per_ft_tensorized(self, dataset: Dataset, individual_params):
        # shape (n_individuals, dimension)
        s = (dataset.n_observations_per_ft * (self.MOCK_NOISE_STD_DIAG ** 2)).view(1, -1) # 2D
        assert s.shape == (1, self.dimension)
        return s

class TestNoiseModelAndNoiseStruct(LeaspyTestCase):

    def assertEqual(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertAllClose(a, b)
        else:
            super().assertEqual(a, b)

    def assertDictWithTensorsMatch(self, a, b):

        self.assertEqual(type(a), type(b))

        if isinstance(a, dict):
            keys = a.keys()
            self.assertEqual(keys, b.keys())
            for k, a_v in a.items():
                self.assertDictWithTensorsMatch(a_v, b[k])
        else:
            self.assertEqual(a, b)

    @classmethod
    def setUpClass(cls):
        # for tmp handling
        super().setUpClass()

        cls.data = Data.from_csv_file(cls.example_data_path)
        cls.dataset = Dataset(cls.data)
        cls.models = {
            'bernoulli': FakeModel(cls.dataset.headers, 'bernoulli'),
            'gaussian_scalar': FakeModel(cls.dataset.headers, 'gaussian_scalar', noise_std=0.0314),
            'gaussian_diagonal': FakeModel(cls.dataset.headers, 'gaussian_diagonal',
                                           noise_std=torch.tensor([.04, .05, .06, .0314])),
            'ordinal': FakeModel(cls.dataset.headers, 'ordinal'),
            'ordinal_ranking': FakeModel(cls.dataset.headers, 'ordinal_ranking'),
        }

    def setUp(self) -> None:
        torch.random.manual_seed(42)

    def test_predefined_noise_structs(self):
        for noise_struct_name, noise_struct in NOISE_STRUCTS.items():
            self.assertIsInstance(noise_struct, NoiseStruct, msg=noise_struct_name)

    def test_no_noise(self):
        nm = NoiseModel(None)
        # self.assertIsNone(nm.name)
        self.assertEqual(nm.struct, NoiseStruct())
        self.assertIsNone(nm.struct.distribution_factory)
        self.assertEqual(nm.distributions_kws, {})
        self.assertIsNone(nm.scale)

        for shape in [(), (1,), (2, 3), (5, 2)]:
            t = torch.randn(shape)
            # input is output (no noise & no copy)
            sampler = nm.sampler_around(t)
            self.assertIs(sampler(), t)
            self.assertIs(nm.sample_around(t), t)
            # no random variable
            with self.assertRaises(Exception):
                nm.rv_around(t)  # RV is undefined for no-noise!

    def test_bad_noise_model(self):

        with self.assertRaises(Exception):
            NoiseModel('bad_model_type')
        with self.assertRaises(Exception):
            NoiseModel('gaussian_block')
        with self.assertRaises(Exception):
            NoiseModel('model')  # only in NoiseModel.from_model
        with self.assertRaises(Exception):
            NoiseModel(42.5)  # bad type

    def test_bernoulli_model(self):

        nm = NoiseModel('bernoulli')
        # self.assertEqual(nm.name, 'bernoulli')
        self.assertIsInstance(nm.struct, NoiseStruct)
        self.assertEqual(nm.struct.distribution_factory, torch.distributions.Bernoulli)
        self.assertEqual(nm.distributions_kws, {})
        self.assertIsNone(nm.scale)

        # check compat with canonically associated model
        nm.check_compat_with_model(self.models['bernoulli'])

        for shape in [(), (1,), (2, 3), (5, 2)]:
            probs = torch.randn(shape).clamp(min=1e-3, max=1-1e-3)
            # input is output (no noise & no copy)
            rv = nm.rv_around(probs)
            self.assertIsInstance(rv, torch.distributions.Bernoulli)
            self.assertEqual(rv.probs, probs)
            sampler = nm.sampler_around(probs)
            self.assertEqual(nm.sample_around(probs).shape, probs.shape)
            self.assertEqual(sampler().shape, probs.shape)
            self.assertEqual(rv.sample().shape, probs.shape)

        self.assertEqual(nm.sample_around(torch.tensor([0., 0.])), torch.tensor([0., 0.]))
        self.assertEqual(nm.sample_around(torch.tensor([[1.]])), torch.tensor([[1.]]))
        self.assertEqual(nm.sample_around(torch.tensor([[0.], [1.]])), torch.tensor([[0.], [1.]]))

        # errors in input
        with self.assertRaises(Exception):
            NoiseModel('bernoulli', scale=37.5)
        with self.assertRaises(Exception):
            NoiseModel('Bernoulli')

        # errors in probs
        with self.assertRaises(Exception):
            nm.sample_around(torch.tensor(-1.))  # bad range for probs
        with self.assertRaises(Exception):
            nm.sample_around(torch.tensor([[.5, 1.05]]))  # bad range for one of probs
        #with self.assertRaises(Exception):
        #    nm.sample_around(.5)  # bad type --> OK accepted
        with self.assertRaises(Exception):
            nm.sample_around('0.5')  # bad type

    def test_gaussian_model(self):

        for noise_struct_name, scale in {
                'gaussian_scalar': FakeModel.MOCK_NOISE_STD_SCALAR,
                'gaussian_diagonal': FakeModel.MOCK_NOISE_STD_DIAG
            }.items():

            nm = NoiseModel(noise_struct_name, scale=scale)
            # self.assertEqual(nm.name, noise_struct)
            self.assertEqual(nm.struct.distribution_factory, torch.distributions.Normal)
            self.assertDictWithTensorsMatch(nm.distributions_kws, {'scale': scale})
            self.assertEqual(nm.scale, scale)

            # check compat with canonically associated model
            nm.check_compat_with_model(self.models[noise_struct_name])

            for shape in [(), (1,), (4, scale.numel()), (3, 5, scale.numel())]:
                locs = torch.randn(shape)
                # input is output (no noise & no copy)
                rv = nm.rv_around(locs)
                self.assertIsInstance(rv, torch.distributions.Normal)
                self.assertEqual(rv.scale, scale)
                self.assertEqual(rv.loc, locs)

                # shape is broadcasted between noise shape and locs shape <!>
                expected_shape = torch_broadcast_shapes(locs.shape, scale.shape)

                self.assertEqual(nm.sample_around(locs).shape, expected_shape)
                self.assertEqual(nm.sampler_around(locs)().shape, expected_shape)
                self.assertEqual(rv.sample().shape, expected_shape)

            # errors in input
            with self.assertRaises(Exception):
                NoiseModel(noise_struct_name)  # no scale...
            with self.assertRaises(Exception):
                NoiseModel(noise_struct_name, Scale=5.)  # bad param name
            with self.assertRaises(Exception):
                NoiseModel(noise_struct_name, bad_param=5.)  # bad param name
            with self.assertRaises(Exception):
                NoiseModel(noise_struct_name, scale=scale, bad_param=5.)  # bad extra param

            # errors in scale
            with self.assertRaises(Exception):
                # bad 0-std-dev
                NoiseModel(noise_struct_name, scale=0.).sample_around(torch.tensor([1.]))
            #with self.assertRaises(Exception):
            #    nm.sample_around(.5)  # bad type --> OK accepted
            with self.assertRaises(Exception):
                nm.sample_around('0.5')  # bad type

            # in-compat locs & shape
            if 'diagonal' in noise_struct_name:
                with self.assertRaises(Exception):
                    nm.sample_around(torch.zeros((len(scale)+1,)))
                with self.assertRaises(Exception):
                    nm.sample_around(torch.zeros((len(scale),42)))

        # non-univariate scale
        with self.assertRaises(Exception):
            NoiseModel('gaussian_scalar', noise_scale=torch.tensor([.1, .5]))
        # scale with non-compat model
        nm = NoiseModel('gaussian_diagonal', scale=torch.tensor([.1, .5, .2]))
        with self.assertRaises(Exception):
            nm.check_compat_with_model(FakeModel(['A', 'B'], 'gaussian_diagonal'))

        # shape ok
        nm.check_compat_with_model(FakeModel(['A', 'B', 'C'], 'gaussian_diagonal'))
        # even univariate
        nm_univ = NoiseModel('gaussian_diagonal', scale=torch.tensor([.1111]))
        nm_univ.check_compat_with_model(FakeModel(['A'], 'gaussian_diagonal'))

    def test_noise_from_model(self):

        for mod in self.models.values():
            nm = NoiseModel.from_model(mod)
            # self.assertEqual(nm.name, mod.noise_model)
            self.assertIsNotNone(nm.struct.distribution_factory)
            self.assertDictWithTensorsMatch(nm.distributions_kws, mod._kws('scale', mode='model'))

        # errors in input
        with self.assertRaises(Exception):
            NoiseModel.from_model(model='bad_model_type')
        with self.assertRaises(Exception):
            NoiseModel.from_model(model=None)
        with self.assertRaises(Exception):
            NoiseModel.from_model(model=FakeModel(self.dataset.headers, 'bad_noise_model'))
        with self.assertRaises(Exception):
            # missing noise scale!
            NoiseModel.from_model(model=FakeModel(self.dataset.headers, 'gaussian_scalar', noise_std=None))

    def test_inherit_mode(self):

        # inherit noise_model from model but need to pass values when needed
        for kw in ['inherit_struct', 'default']:
            for mod in self.models.values():

                nm_factory = lambda: NoiseModel.from_model(mod, kw, **mod._kws('scale', mode='mock'))
                if kw == 'default':
                    with self.assertWarns(FutureWarning):
                        nm = nm_factory()
                else:
                    nm = nm_factory()

                # self.assertEqual(nm.name, mod.noise_model)
                self.assertIsNotNone(nm.struct.distribution_factory)
                self.assertDictWithTensorsMatch(nm.distributions_kws, mod._kws('scale', mode='mock'))

        # test errors (in-compat with model)
        with self.assertRaises(Exception):
            NoiseModel.from_model(self.models['gaussian_scalar'], 'inherit_struct', scale=FakeModel.MOCK_NOISE_STD_DIAG)
        with self.assertRaises(Exception):
            NoiseModel.from_model(self.models['gaussian_diagonal'], 'inherit_struct', scale=[.3, .2])  # bad nb of features
        with self.assertRaises(Exception):
            NoiseModel.from_model(self.models['bernoulli'], 'inherit_struct', not_needed_param=42)
        with self.assertRaises(Exception):
            NoiseModel.from_model(self.models['gaussian_diagonal'], 'inherit_struct')  # missing `scale`

    def test_noise_from_model_reconstruction_in_data(self):

        # forced scalar mode (no matter noise model)
        fake_model = FakeModel(['blabla', 'ft2'], 'no_matter')
        n = NoiseModel.rmse_model(fake_model, self.dataset, {}, scalar=True)
        self.assertEqual(n, FakeModel.MOCK_NOISE_STD_SCALAR)

        fake_model = FakeModel([f'ft{i}' for i in range(FakeModel.MOCK_NOISE_STD_DIAG.numel())], 'no_matter')
        n = NoiseModel.rmse_model(fake_model, self.dataset, {}, scalar=False)
        self.assertEqual(n, FakeModel.MOCK_NOISE_STD_DIAG)

        # scalar inferred from model.noise_model
        fake_model = FakeModel(['blabla', 'ft2'], 'gaussian_scalar')
        n = NoiseModel.rmse_model(fake_model, self.dataset, {})
        self.assertEqual(n, FakeModel.MOCK_NOISE_STD_SCALAR)

        fake_model = FakeModel([f'ft{i}' for i in range(FakeModel.MOCK_NOISE_STD_DIAG.numel())], 'gaussian_diagonal')
        n = NoiseModel.rmse_model(fake_model, self.dataset, {})
        self.assertEqual(n, FakeModel.MOCK_NOISE_STD_DIAG)

    def test_multinomial_loss(self):

        nm = NoiseModel('ordinal')
        self.assertIsInstance(nm.struct, NoiseStruct)
        self.assertEqual(nm.struct.distribution_factory, MultinomialDistribution.from_pdf)
        self.assertEqual(nm.distributions_kws, {})
        self.assertIsNone(nm.scale)

        # check compat with canonically associated model
        nm.check_compat_with_model(self.models['ordinal'])

        for shape in [(1,2), (2, 3), (5, 4)]:
            probs = torch.rand(shape)  # must be in [0, 1]
            probs = probs / probs.sum(dim=-1, keepdim=True)
            # input is output (no noise & no copy)
            rv = nm.rv_around(probs)
            self.assertIsInstance(rv, MultinomialDistribution)
            #self.assertEqual(rv.probas, probs) # pdf are not stored, only the cdf
            sampler = nm.sampler_around(probs)
            self.assertEqual(nm.sample_around(probs).shape, probs.shape[:-1])
            self.assertEqual(sampler().shape, probs.shape[:-1])
            self.assertEqual(rv.sample().shape, probs.shape[:-1])

        self.assertEqual(nm.sample_around(torch.tensor([0., 1.])), torch.tensor([1]))
        self.assertEqual(nm.sample_around(torch.tensor([[1., 0., 0.]])), torch.tensor([0]))
        self.assertEqual(nm.sample_around(torch.tensor([[0., 0., 1.]])), torch.tensor([[2]]))
        self.assertEqual(nm.sample_around(torch.tensor([[0., 1., 0.], [1., 0., 0.]])), torch.tensor([[1, 0]]))

        # errors in input
        with self.assertRaises(Exception):
            NoiseModel('ordinal', scale=37.5)
        with self.assertRaises(Exception):
            NoiseModel('Ordinal')

        # errors in probs
        with self.assertRaises(Exception):
            nm.sample_around(torch.tensor(-1.))  # bad range for probs
        #with self.assertRaises(Exception):
        #   nm.sample_around(torch.tensor([[.5, .5, .5]]))  # does not sum to one --> no check for now
        with self.assertRaises(Exception):
            nm.sample_around('0.5')  # bad type

    def test_ordinal_ranking_loss(self):

        nm = NoiseModel('ordinal_ranking')
        self.assertIsInstance(nm.struct, NoiseStruct)
        self.assertEqual(nm.struct.distribution_factory, MultinomialDistribution)
        self.assertEqual(nm.distributions_kws, {})
        self.assertIsNone(nm.scale)

        # check compat with canonically associated model
        nm.check_compat_with_model(self.models['ordinal_ranking'])

        for shape in [(1,2), (2, 3), (5, 4)]:
            probs = torch.rand(shape)  # must be in [0, 1]
            probs = probs / probs.sum(dim=-1, keepdim=True)
            sf = (1. - probs.cumsum(dim=-1))
            # input is output (no noise & no copy)
            rv = nm.rv_around(sf)
            self.assertIsInstance(rv, MultinomialDistribution)
            #self.assertEqual(rv.probas, probs) # pdf are not stored, only the cdf
            sampler = nm.sampler_around(sf)
            self.assertEqual(nm.sample_around(sf).shape, probs.shape[:-1])
            self.assertEqual(sampler().shape, probs.shape[:-1])
            self.assertEqual(rv.sample().shape, probs.shape[:-1])

        self.assertEqual(nm.sample_around(torch.tensor([1., 0.])), torch.tensor([1]))
        self.assertEqual(nm.sample_around(torch.tensor([[0., 0., 0.]])), torch.tensor([0]))
        self.assertEqual(nm.sample_around(torch.tensor([[1., 1., 0.]])), torch.tensor([[2]]))
        self.assertEqual(nm.sample_around(torch.tensor([[1., 1., 1.], [1., 0., 0.]])), torch.tensor([[3, 1]]))

        # errors in input
        with self.assertRaises(Exception):
            NoiseModel('ordinal_ranking', scale=37.5)
        with self.assertRaises(Exception):
            NoiseModel('OrdinalRanking')

        # errors in probs
        with self.assertRaises(Exception):
            nm.sample_around(torch.tensor(-1.))  # bad range for probs
        #with self.assertRaises(Exception):
        #   nm.sample_around(torch.tensor([[.3, .6, .5]]))  # not decreasing --> no check for now
        with self.assertRaises(Exception):
            nm.sample_around('0.5')  # bad type
