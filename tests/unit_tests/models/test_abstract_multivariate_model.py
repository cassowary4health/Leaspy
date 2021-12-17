from leaspy.models.abstract_multivariate_model import AbstractMultivariateModel

# <!> NEVER import real tests classes at top-level (otherwise their tests will be duplicated...), only MIXINS!!
from tests.unit_tests.models.test_univariate_model import ManifoldModelTest_Mixin


class AbstractMultivariateModelTest(ManifoldModelTest_Mixin):

    @ManifoldModelTest_Mixin.allow_abstract_class_init(AbstractMultivariateModel)
    def test_constructor_abstract_multivariate(self):
        """
        Test attribute's initialization of leaspy abstract multivariate model

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`, optional (default None)
            An instance of a subclass of leaspy AbstractModel.
        """

        # Abstract Multivariate Model
        model = AbstractMultivariateModel('dummy')
        self.assertEqual(type(model), AbstractMultivariateModel)
        self.assertEqual(model.name, 'dummy')

        # Test common initialization with univariate
        self.check_common_attrs(model)

        # Test specific multivariate initialization
        self.assertEqual(model.dimension, None)
        self.assertEqual(model.source_dimension, None)
        self.assertEqual(model.parameters['betas'], None)
        self.assertEqual(model.parameters['sources_mean'], None)
        self.assertEqual(model.parameters['sources_std'], None)
        self.assertEqual(model.MCMC_toolbox['priors']['betas_std'], None)

