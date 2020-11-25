import numpy as np

from leaspy.io.outputs.individual_parameters import IndividualParameters
import statsmodels.api as sm


class LMEPersonalizeAlgorithm():

    def __init__(self, settings):
        self.name = 'lme_personalize'
        assert settings.name == self.name
        self.features = None  # not very useful

    def run(self, model, dataset):
        self.features = dataset.headers
        if len(dataset.headers) > 1:
            raise ValueError(
                "LME model cannot be personalized on more than 1 feature; "
                "dataset features : {}".format(dataset.headers))
        if model.features != dataset.headers:
            raise ValueError(
                "LME model was not fitted on the same features than those you want to personalize on. "
                "Model features : {}, data features: {}".format(model.features, dataset.headers))
        ip = IndividualParameters()
        for it in range(dataset.n_individuals):
            idx = dataset.indices[it]
            times = dataset.get_times_patient(it)
            values = dataset.get_values_patient(it).numpy()
            ind_ip = self._get_individual_random_effect(model, times, values)

            ip.add_individual_parameters(str(idx), ind_ip)
        return ip, 0

    @staticmethod
    def _remove_nans(values, times):
        values = values.flatten()
        mask = ~np.isnan(values)
        values = values[mask]
        times = times[mask]
        return values, times

    def _get_individual_random_effect(self, model, times, values):
        # remove nans
        values, times = self._remove_nans(values, times)
        X = sm.add_constant(times, prepend=True, has_constant='add')
        residual = values - np.dot(X, model.parameters['fe_params'])
        # only valid with random intercept ("Z"=[1,...,1] and cov_re is a scalar)
        individual_random_effect = np.mean(residual) / (
                    1 + 1 / (len(values) * model.parameters['cov_re_unscaled'].item()))
        return {'random_intercept': individual_random_effect}

    def set_output_manager(self, settings):
        return 0
