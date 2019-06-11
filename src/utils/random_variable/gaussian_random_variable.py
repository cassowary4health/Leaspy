from src.utils.random_variable.abstract_random_variable import AbstractRandomVariable
import src.utils.conformity.Profiler
import numpy as np
from numba import jit

# TODO Numba class ??? do the sampling as well ???


@jit(nopython=True, cache=True)
def compute_negativeloglikelihood_numba(x, mu, inv_var, log_ctst):
    return inv_var * (x - mu) ** 2 + log_ctst

class GaussianRandomVariable(AbstractRandomVariable):
    def __init__(self, name, mu, variance):

        self.name = name
        self._mu = mu
        self._variance = variance
        self._variance_inverse = 1/variance
        self._log_constant = np.log(np.sqrt(2*np.pi*variance))

    @property
    def mu(self):
        return self._mu

    @property
    def variance(self):
        return self._variance

    @property
    def variance_inverse(self):
        return self._variance_inverse

    @property
    def log_constant(self):
        return self._log_constant

    @mu.setter
    def mu(self, mu):
        self._mu = mu
        self._log_constant = np.log(np.sqrt(2 * np.pi * self.variance))

    @variance.setter
    def variance(self, variance):
        self._variance = variance
        self._variance_inverse = 1 / variance
        self._log_constant = np.log(np.sqrt(2 * np.pi * variance))




    #@src.utils.conformity.Profiler.do_profile()
    def compute_negativeloglikelihood(self, x):
        return self.variance_inverse * (x - self.mu) ** 2 + self._log_constant

        """
        return compute_negativeloglikelihood_numba(x.detach().numpy(),
                                                   self.mu,
                                                   self.variance_inverse,
                                                   self._log_constant)"""




