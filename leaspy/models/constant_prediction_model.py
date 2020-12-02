import torch

class ConstantModel():
    """
    ConstantPredictionModel is a benchmark model that predicts a predict which is constant to the last seen visit.
`
    The prediction depends on the algorithm setting. It could predict the `last` value, the `last_known` (which is
    different is `last` has NaN), `max` and `mean`

    Attributes
    ----------
    name: str
        The model's name
    parameters: dict
        Contains the model parameters
    features: list[str]
        List of the model  features
    """
    def __init__(self, name):
        self.is_initialized = True
        self.name = name
        self.features = None
        self.dimension = None

    def compute_individual_trajectory(self, timepoints, ip):
        values = [ip[f] for f in self.features]
        return torch.Tensor([[values] * len(timepoints)])