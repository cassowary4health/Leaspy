from typing import Optional

from leaspy.models.multivariate_model import MultivariateModel
from leaspy.models.noise_models import GaussianScalarNoiseModel, BaseNoiseModel
from leaspy.exceptions import LeaspyModelInputError
from leaspy.utils.docs import doc_with_super

# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either model values or model values + jacobians wrt individual parameters
# TODO refact? subclass or other proper code technique to extract model's concrete formulation depending on if linear, logistic, mixed log-lin, ...


@doc_with_super()
class UnivariateModel(MultivariateModel):
    """
    Univariate (logistic or linear) model for a single variable of interest.

    Parameters
    ----------
    name : str
        Name of the model
    **kwargs
        Hyperparameters of the model

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        * If `name` is not one of allowed sub-type: 'univariate_linear' or 'univariate_logistic'
        * If hyperparameters are inconsistent
    """

    SUBTYPES_SUFFIXES = {
        'univariate_linear': '_linear',
        'univariate_logistic': '_logistic'
    }

    def __init__(self, name: str, noise_model: Optional[BaseNoiseModel] = None, **kwargs):

        # consistency checks
        if kwargs.pop('dimension', 1) not in {1, None}:
            raise LeaspyModelInputError("You should not provide `dimension` != 1 for univariate model.")

        if kwargs.pop('source_dimension', 0) not in {0, None}:
            raise LeaspyModelInputError("You should not provide `source_dimension` != 0 for univariate model.")

        # default noise model is gaussian_scalar
        noise_model = noise_model or GaussianScalarNoiseModel(name)

        super().__init__(name, dimension=1, source_dimension=0, noise_model=noise_model, **kwargs)

    def check_noise_model_compatibility(self, model: BaseNoiseModel) -> None:
        if not isinstance(model, GaussianScalarNoiseModel):
            raise ValueError(
                f"Expected a GaussianScalarNoiseModel instance, but received a {model.__class__.__name__}"
            )