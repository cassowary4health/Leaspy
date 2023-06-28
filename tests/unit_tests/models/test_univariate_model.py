from unittest import skip

from leaspy.models.abstract_model import AbstractModel
from leaspy.models.univariate import UnivariateModel
from leaspy.models.obs_models import FullGaussianObs

from tests import LeaspyTestCase


class ManifoldModelTestMixin(LeaspyTestCase):

    def check_common_attrs(self, model: AbstractModel):
        self.assertIsInstance(model, AbstractModel)
        for variable in ("g", "tau_mean", "tau_std", "xi_mean", "xi_std"):
            self.assertIn(variable, model.state.dag)


class UnivariateModelTest(ManifoldModelTestMixin):

    def _generic_testing(self, model_name: str):
        model = UnivariateModel(model_name)

        self.assertIsInstance(model, UnivariateModel)
        self.assertEqual(model.name, model_name)
        self.assertEqual(model.dimension, 1)
        self.assertEqual(model.source_dimension, 0)
        self.assertIsInstance(model.obs_models[0], FullGaussianObs)

        model.initialize_state()
        self.check_common_attrs(model)

    def test_univariate_logistic_constructor(self):
        """
        Test attribute's initialization of leaspy univariate logistic model.
        """
        self._generic_testing("univariate_logistic")

    @skip("Linear models are currently broken")
    def test_univariate_linear_constructor(self):
        """
        Test attribute's initialization of leaspy univariate linear model.
        """
        self._generic_testing("univariate_linear")

    def test_wrong_name(self):
        with self.assertRaises(ValueError):
            UnivariateModel("univariate_unknown-suffix")
