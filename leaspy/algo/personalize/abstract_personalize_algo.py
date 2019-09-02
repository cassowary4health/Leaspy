from ..abstract_algo import AbstractAlgo
import numpy as np

class AbstractPersonalizeAlgo(AbstractAlgo):
    """
    Abstract class for "personalize" algorithm.
    Estimation of individual parameters of a given "Data" file with
    a freezed model (already estimated, or loaded from known parameters)
    """

    def __init__(self, settings):
        """
        Initialize from settings object
        :param settings:
        """
        # Algorithm parameters
        self.algo_parameters = settings.parameters

    def _get_individual_parameters(self, model, times, values):
        """
        Abstract
        :param model:
        :param times:
        :param values:
        :return:
        """
        raise NotImplementedError('This algorithm does not present a personalization procedure')

    def run(self, model, data):
        """
        Main estimation function
        :param model:
        :param data:
        :return:
        """
        individual_parameters = {}
        total_error = []

        for idx in range(data.n_individuals):
            times = data.get_times_patient(idx)
            values = data.get_values_patient(idx)

            xi, tau, sources, err = self._get_individual_parameters(model, times, values)

            individual_parameters[data.indices[idx]] = {
                'xi': xi,
                'tau': tau,
                'sources': sources
            }

            total_error.append(err.squeeze(0).detach().numpy())

        noise_std = np.std(np.vstack(total_error))
        print("The standard deviation of the noise at the end of the personalization is of {:.4f}".format(noise_std))

        return individual_parameters, noise_std
