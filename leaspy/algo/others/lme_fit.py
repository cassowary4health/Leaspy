import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import torch
import numpy as np


class LMEFitAlgorithm():

    def __init__(self, settings):
        self.name = 'lme_fit'
        assert settings.name == self.name
        self.features = None  # not very useful

    def run(self, model, dataset):
        # get inputs in right format
        if len(dataset.headers) > 1:
            raise ValueError(
                "LME model cannot be fitted on more than 1 feature"
                "dataset features : {}".format(dataset.headers))
        model.features = dataset.headers

        # get data
        ages = self._get_reformated(dataset, 'timepoints')
        y = self._get_reformated(dataset, 'values')
        subjects_with_repeat = self._get_reformated_subjects(dataset)

        # model
        X = sm.add_constant(ages, prepend=True, has_constant='add')
        lme = MixedLM(y, X, subjects_with_repeat, missing='raise')
        fitted_lme = lme.fit()
        parameters = {"fe_params": fitted_lme.fe_params,
                    "cov_re": fitted_lme.cov_re,
                    "cov_re_unscaled": fitted_lme.cov_re_unscaled,
                    "bse_fe": fitted_lme.bse_fe,
                    "bse_re": fitted_lme.bse_re}
        model.load_parameters(parameters)

    @staticmethod
    def _get_reformated(dataset, elem):
        # reformat ages
        dataset_elem = dataset.__getattribute__(elem)
        # flatten
        flat_elem = torch.flatten(dataset_elem).numpy()
        # remove padding & nans
        final_elem = flat_elem[torch.flatten(dataset.mask>0)]
        return final_elem

    @staticmethod
    def _get_reformated_subjects(dataset):
        subjects_with_repeat = []
        for ind, subject in enumerate(dataset.indices):
            subjects_with_repeat += [subject]*max(dataset.nb_observations_per_individuals) #[ind]
        subjects_with_repeat = np.array(subjects_with_repeat)
        # remove padding & nans
        subjects_with_repeat = subjects_with_repeat[torch.flatten(dataset.mask > 0)]
        return subjects_with_repeat

    def set_output_manager(self, settings):
        return 0