import numpy as np
from leaspy.io.outputs.individual_parameters import IndividualParameters


class ConstantPredictionAlgorithm():

    def __init__(self, settings):
        """
        ConstantPredictionAlgorithm is the algorithm that outputs a constant prediction

        This constant prediction could be 'last', 'last_known', 'max', 'mean'.
        """
        self.name = 'constant_prediction'
        assert settings.name == self.name
        self.features = None
        if settings.parameters['prediction_type'] not in ['last', 'last_known', 'max', 'mean']:
            raise ValueError('The `prediction_type` of the constant prediction should be `last`, `last_known`, `max` or `mean`')
        self.prediction_type = settings.parameters['prediction_type']

    def run(self, model, dataset):
        self.features = dataset.headers
        model.features = dataset.headers
        ip = IndividualParameters()
        for it in range(dataset.n_individuals):
            idx = dataset.indices[it]
            times = dataset.get_times_patient(it)
            values = dataset.get_values_patient(it).numpy()
            ind_ip = self._get_individual_last_values(times, values)

            ip.add_individual_parameters(str(idx), ind_ip)

        return ip, 0

    def _get_individual_last_values(self, times, values):
        # Return the maximum value
        if self.prediction_type == 'max':
            ip = np.nanmax(values, axis=0)
            ind_ip = dict(zip(self.features, ip))
            return ind_ip

        # Return the mean value
        if self.prediction_type == 'mean':
            ip = np.nanmean(values, axis=0)
            ind_ip = dict(zip(self.features, ip))
            return ind_ip

        # Return the last or last-known
        sorted_indices = sorted(range(len(times)), key=lambda k: times[k])
        last_values = values[sorted_indices[-1]]
        ind_ip = dict(zip(self.features, last_values))

        # Sometimes, last value can be a NaN. If this bahavior is intended, then return
        if self.prediction_type == 'last':
            return ind_ip

        # Here, take previous values if NaN
        for i, (feat, value) in enumerate(ind_ip.items()):
            if value == value:
                continue

            for visit_idx in sorted_indices:
                potential_val = values[visit_idx][i]
                if potential_val == potential_val:
                    ind_ip[feat] = potential_val
                    continue

        return ind_ip

    def set_output_manager(self, settings):
        return 0