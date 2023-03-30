from leaspy.models.noise_models import (
    NOISE_MODELS,
    BernoulliNoiseModel,
    noise_model_factory,
    GaussianScalarNoiseModel,
)
from leaspy.exceptions import LeaspyModelInputError

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
        for model_class in NOISE_MODELS.values():
            with self.subTest(model_name=model_class):
                model = noise_model_factory(model_class)
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


class TestBernoulliNoiseModel(LeaspyTestCase):
    def test_constructor(self):
        """Test the initialization"""
        model = BernoulliNoiseModel()

