import numpy as np
import statsmodels.api as sm
import torch

class LMEModel():
    """
    LMEModel is a benchmark model that fits and personalize the following linear mixt-effects model :
    y_i ~ fixed_slope * ages_i + random_intercept_i + fixed_intercept
    With:
    y_i: feature array of the i-th patient,
    ages_i : ages array of the i-th patient,
    Hence, this model must be fitted on one feature only.

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
        self.parameters = None # TODO load defaults ?

    def load_parameters(self, parameters):
        self.parameters = parameters

    def compute_individual_trajectory(self, timepoints, ip):
        X = sm.add_constant(timepoints, prepend=True, has_constant='add')
        return torch.Tensor([np.dot(self.parameters['fe_params'], X.T) + ip['random_intercept']])