from .abstract_multivariate_model import AbstractMultivariateModelTest

from leaspy.models.multivariate_parallel_model import MultivariateParallelModel


class MultivariateParallelModelTest(AbstractMultivariateModelTest):

    def test_constructor_multivariate_parallel(self, model=None):
        """
        Test attribute's initialization of leaspy multivariate parallel model
        :param model: leaspy model object
        :return: exit code
        """
        print("Unit-test constructor MultivariateModel")

        if model is None:
            model = MultivariateParallelModel("logistic")
            self.assertEqual(type(model), MultivariateParallelModel)

        self.assertEqual(model.parameters['deltas'], None)
        self.assertEqual(model.MCMC_toolbox['priors']['deltas_std'], None)
