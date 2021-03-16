import numpy as np
import statsmodels.api as sm
import torch

from leaspy.models.generic_model import GenericModel

class LMEModel(GenericModel): # TODO should inherit from AbstractModel?
    """
    LMEModel is a benchmark model that fits and personalize a linear mixed-effects model

    The model specification is the following:
    :math:`y_{ij} = fixed_{intercept} + random_{intercept_i} + (fixed_{slope} + random_{slope_{age_i}}) * ages_{ij}`
    with:
        * y_ij: feature array of the i-th patient (n_i visits),
        * ages_ij: ages array of the i-th patient (n_i visits)

    This model must be fitted on one feature only (univariate model).

    Attributes
    ----------
    name: str
        The model's name
    parameters: dict
        Contains the model parameters
    features: list[str]
        List of the model features

    See Also
    --------
    leaspy.algo.others.lme_fit.LMEFitAlgorithm
    leaspy.algo.others.lme_personalize.LMEPersonalizeAlgorithm
    """

    _hyperparameters = ('features', 'with_random_slope_age')

    def __init__(self, name):

        super().__init__(name)
        # <!> always univariate

        self.with_random_slope_age = False

        self.is_initialized = True

    def compute_individual_trajectory(self, timepoints, ip):
        """
        Compute scores values at the given time-point(s) given a subject's individual parameters.

        Parameters
        ----------
        timepoints: array-like of ages (not normalized)
            Timepoints to compute individual trajectory at

        ip: dict
            Individual parameters:
                * random_intercept
                * random_slope_age (if ``with_random_slope_age == True``)

        Returns
        -------
        :class:`torch.Tensor` of float of shape (n_individuals == 1, n_tpts == len(timepoints), n_features == 1)
        """

        # normalize ages (np.ndarray of float, 1D)
        ages_norm = (np.array(timepoints).reshape(-1) - self.parameters['ages_mean']) / self.parameters['ages_std']

        # design matrix (same for fixed and random effects)
        X = sm.add_constant(ages_norm, prepend=True, has_constant='add')

        #assert 'random_intercept' in ip
        if not self.with_random_slope_age:
            # no random slope on ages (fixed effect only)
            re_params = np.array([ ip['random_intercept'].item(), 0 ])
        else:
            #assert 'random_slope_age' in ip
            re_params = np.array([ ip['random_intercept'].item(), ip['random_slope_age'].item() ])

        y = X @ (self.parameters['fe_params'] + re_params)

        return torch.tensor(y, dtype=torch.float32).reshape((1, -1, 1))
