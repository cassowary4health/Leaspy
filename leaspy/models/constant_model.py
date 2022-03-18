import torch

from leaspy.models.generic_model import GenericModel
from leaspy.exceptions import LeaspyModelInputError

from leaspy.utils.docs import doc_with_super


@doc_with_super()
class ConstantModel(GenericModel):
    r"""
    `ConstantModel` is a benchmark model that predicts constant values (no matter what the patient's ages are).

    These constant values depend on the algorithm setting and the patient's values provided during calibration.
    It could predict:
        * `last`: last value seen during calibration (even if NaN),
        * `last_known`: last non NaN value seen during calibration*§,
        * `max`: maximum (=worst) value seen during calibration*§,
        * `mean`: average of values seen during calibration§.

    | \\* <!> depending on features, the `last_known` / `max` value may correspond to different visits.
    | § <!> for a given feature, value will be NaN if and only if all values for this feature were NaN.

    Parameters
    ----------
    name : str
        The model's name
    **kwargs
        Hyperparameters for the model.
        None supported for now.

    Attributes
    ----------
    name : str
        The model's name
    is_initialized : bool
        Always True (no true initialization needed for constant model)
    features : list[str]
        List of the model features.
        Unlike most models features will be determined at `personalization` only (because it does not needed any `fit`)
    dimension : int
        Number of features (read-only)
    parameters : dict
        Model has no parameters: empty dictionary.
        The `prediction_type` parameter should be defined during `personalization`.
        Example:
            >>> AlgorithmSettings('constant_prediction', prediction_type='last_known')

    See Also
    --------
    :class:`~leaspy.algo.others.constant_prediction_algo.ConstantPredictionAlgorithm`
    """

    def __init__(self, name: str, **kwargs):

        super().__init__(name, **kwargs)

        # no fit algorithm is needed for constant model; every "personalization" will re-initialize model
        # however, we need to mock that model is personalization-ready by setting self.is_initialized (API requirement)
        self.is_initialized = True

    def compute_individual_trajectory(self, timepoints, individual_parameters):

        if self.features is None:
            raise LeaspyModelInputError('The model was not properly initialized.')

        values = [individual_parameters[f] for f in self.features]
        return torch.tensor([[values] * len(timepoints)], dtype=torch.float32)
