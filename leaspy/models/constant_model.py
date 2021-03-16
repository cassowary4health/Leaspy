import torch

from leaspy.models.generic_model import GenericModel

class ConstantModel(GenericModel):
    """
    `ConstantModel` is a benchmark model that predicts constant values no matter of the patient's ages.

    These constant values depend on the algorithm setting and the patient's values provided during calibration.
    It could predict:
        * `last_known`: last non NaN value seen during calibration*§,
        * `last`: last value seen during calibration (even if NaN),
        * `max`: maximum (=worst) value seen during calibration*§,
        * `mean`: average of values seen during calibration§.

    | \\* <!> depending on features, the `last_known` / `max` value may correspond to different visits.
    | § <!> for a given feature, value will be NaN if and only if all values for this feature were NaN.

    Attributes
    ----------
    name: str
        The model's name
    features: list[str]
        List of the model features
    dimension: int
        Number of features (read-only)
    parameters: dict
        Population parameters: empty dictionary.

    See also
    --------
    leaspy.algo.others.constant_prediction_algo.ConstantPredictionAlgorithm
    """

    def __init__(self, name):

        super().__init__(name)

        # no more initialization needed...
        self.is_initialized = True

    def compute_individual_trajectory(self, timepoints, ip):
        """
        Compute scores values at the given timepoints given a subject's individual parameters.
        """
        values = [ip[f] for f in self.features]
        return torch.tensor([[values] * len(timepoints)], dtype=torch.float32)

