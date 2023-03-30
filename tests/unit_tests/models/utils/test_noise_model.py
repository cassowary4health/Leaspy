import torch

from leaspy.models.noise_models import (
    NOISE_MODELS,
    NO_NOISE,
    BaseNoiseModel,
    BernoulliNoiseModel,
    noise_model_factory,
    GaussianScalarNoiseModel,
    GaussianDiagonalNoiseModel,
    DistributionFamily,
    OrdinalNoiseModel,
    OrdinalRankingNoiseModel,
)
from leaspy.exceptions import LeaspyModelInputError
from leaspy.utils.distributions import MultinomialDistribution

from tests import LeaspyTestCase


class NoiseModelFactoryTest(LeaspyTestCase):
    def test_return_type_from_string_input(self):
        """
        Test that the noise model factory returns noise model
        instances when passing a string.
        """
        for name in NOISE_MODELS.keys():
            with self.subTest(model_name=name):
                model = noise_model_factory(name)
                self.assertIsInstance(model, NOISE_MODELS[name])

    def test_return_type_from_dict_input(self):
        """
        Test that the noise model factory returns noise model
        instances when passing a dictionary.
        """
        for name in NOISE_MODELS.keys():
            with self.subTest(model_name=name):
                model = noise_model_factory({"name": name})
                self.assertIsInstance(model, NOISE_MODELS[name])

    def test_return_type_from_class_input(self):
        """
        Test that the noise model factory returns noise model instances
        when passing a noise model instance directly.
        """
        for model_name, model_class in NOISE_MODELS.items():
            with self.subTest(model_name=model_name):
                model = noise_model_factory(model_class())
                self.assertIsInstance(model, model_class)

    def test_lower_case_input(self):
        """
        Test that the noise model factory is case-insensitive
        and handles "-" vs. "_" properly.
        """
        names = (
            "gaussian-scalar",
            "GauSsiaN_scAlaR",
            "gaussian-SCALAR",
            "GAUSSIAN_scalar",
            "GAUSSIAN-SCALAR",
        )
        for name in names:
            model = noise_model_factory(name)
            self.assertIsInstance(model, GaussianScalarNoiseModel)

    def test_wrong_string_argument(self):
        """
        Test that an error is raised when providing a bad
        string argument to the factory.
        """
        bad_inputs = ("gausian-scalar", "foo", "")  # noqa
        for bad_input in bad_inputs:
            self.assertRaises(
                LeaspyModelInputError, noise_model_factory, bad_input
            )

    def test_wrong_non_string_argument(self):
        """
        Test that an error is raised when providing
        a bad non string argument to the factory.
        """
        bad_inputs = (3.8, {"foo": .1})
        for bad_input in bad_inputs:
            self.assertRaises(
                LeaspyModelInputError, noise_model_factory, bad_input
            )


class TestNoNoise(LeaspyTestCase):

    model = NO_NOISE

    def test_constructor(self):
        self.assertIsInstance(self.model, DistributionFamily)
        self.assertIs(self.model.factory, None)
        self.assertIs(self.model.parameters, None)
        self.assertEqual(len(self.model.free_parameters), 0)

    def test_to_dict(self):
        self.assertEqual(self.model.to_dict(), {})

    def test_sample_around(self):
        loc = torch.ones((1,))
        self.assertEqual(self.model.sample_around(loc), loc)


class TestBernoulliNoiseModel(LeaspyTestCase):
    def test_constructor(self):
        model = BernoulliNoiseModel()
        self.assertIsInstance(model, BaseNoiseModel)
        self.assertIsInstance(model, DistributionFamily)
        self.assertEqual(model.factory, torch.distributions.Bernoulli)
        self.assertEqual(len(model.free_parameters), 0)
        self.assertIs(model.parameters, None)

    def test_to_dict(self):
        model = BernoulliNoiseModel()
        self.assertEqual(model.to_dict(), {})


class TestGaussianScalarNoiseModel(LeaspyTestCase):
    def test_constructor(self):
        """Test the initialization"""
        model = GaussianScalarNoiseModel()
        self.assertIsInstance(model, BaseNoiseModel)
        self.assertIsInstance(model, DistributionFamily)
        self.assertIs(model.scale_dimension, None)
        self.assertEqual(model.factory, torch.distributions.Normal)
        self.assertEqual(len(model.free_parameters), 1)
        self.assertIs(model.parameters, None)

    def test_to_dict(self):
        model = GaussianScalarNoiseModel()
        self.assertEqual(model.to_dict(), {})


class TestGaussianDiagonalNoiseModel(LeaspyTestCase):
    def test_constructor(self):
        """Test the initialization"""
        model = GaussianDiagonalNoiseModel()
        self.assertIsInstance(model, BaseNoiseModel)
        self.assertIsInstance(model, DistributionFamily)
        self.assertIs(model.scale_dimension, None)
        self.assertEqual(model.factory, torch.distributions.Normal)
        self.assertEqual(len(model.free_parameters), 1)
        self.assertIs(model.parameters, None)


class TestOrdinalNoiseModel(LeaspyTestCase):
    def test_constructor(self):
        """Test the initialization"""
        model = OrdinalNoiseModel()
        self.assertIsInstance(model, BaseNoiseModel)
        self.assertIsInstance(model, DistributionFamily)
        self.assertIs(model.max_levels, None)
        self.assertEqual(model.factory, MultinomialDistribution.from_pdf)
        self.assertEqual(len(model.free_parameters), 0)
        self.assertIs(model.parameters, None)


class TestOrdinalRankingNoiseModel(LeaspyTestCase):
    def test_constructor(self):
        """Test the initialization"""
        model = OrdinalRankingNoiseModel()
        self.assertIsInstance(model, BaseNoiseModel)
        self.assertIsInstance(model, DistributionFamily)
        self.assertIs(model.max_levels, None)
        self.assertEqual(model.factory, MultinomialDistribution)
        self.assertEqual(len(model.free_parameters), 0)
        self.assertIs(model.parameters, None)

