from .abstract_multivariate_model import AbstractMultivariateModelTest

from leaspy.models.multivariate_model import MultivariateModel


class MultivariateModelTest(AbstractMultivariateModelTest):

    def test_constructor_multivariate(self, model=None):
        """
        Test attribute's initialization of leaspy multivariate model
        :param model: leaspy model object
        :return: exit code
        """
        print("Unit-test constructor MultivariateModel")

        if model is None:
            model = MultivariateModel("logistic")
            self.assertEqual(type(model), MultivariateModel)

        self.assertEqual(model.parameters['v0'], None)
        self.assertEqual(model.MCMC_toolbox['priors']['v0_std'], None)
