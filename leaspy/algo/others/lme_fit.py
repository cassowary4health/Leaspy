import warnings
from leaspy.algo.abstract_algo import AbstractAlgo

import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMParams
import torch
import numpy as np

class LMEFitAlgorithm(AbstractAlgo): # AbstractFitAlgo not so generic (EM)
    """
    Calibration algorithm associated to :class:`~.models.lme_model.LMEModel`

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        * with_random_slope_age : bool
            If False: only varying intercepts
            If True: random intercept & random slope w.r.t ages
        * force_independent_random_effects : bool
            Force independence of random intercept & random slope
        * other keyword arguments passed to :meth:`statsmodels.regression.mixed_linear_model.MixedLM.fit`

    See also
    --------
    :class:`statsmodels.regression.mixed_linear_model.MixedLM`
    """

    def __init__(self, settings):

        super().__init__()

        assert settings.name == 'lme_fit'
        self.name = 'lme_fit'

        params = settings.parameters.copy()

        # Algorithm parameters
        self.hyperparams = {
            hp_name: params.pop(hp_name)
            for hp_name in [
                "with_random_slope_age",
                "force_independent_random_effects" # only an algo setting
            ]
        }
        self.sm_fit_parameters = params # popped from other params
        self.seed = settings.seed

    def run(self, model, dataset):
        """
        Main method, refer to abstract definition in :meth:`~.algo.fit.abstract_fit_algo.AbstractFitAlgo.run`.

        TODO fix proper inheritance

        Parameters
        ----------
        model : :class:`~.models.lme_model.LMEModel`
            A subclass object of leaspy `LMEModel`.
        dataset : :class:`.Dataset`
            Dataset object build with leaspy class objects Data, algo & model
        """

        # Initialize Model
        self._initialize_seed(self.seed)

        # get inputs in right format
        if len(dataset.headers) != 1:
            raise ValueError(
                "LME model is univariate only, provided features: {}".format(dataset.headers))

        # Store hyperparameters in model
        model.load_hyperparameters({
            'features': dataset.headers,
            #**self.hyperparams
            'with_random_slope_age': self.hyperparams['with_random_slope_age']
        })

        # get data
        ages = self._get_reformated(dataset, 'timepoints')
        ages_mean, ages_std = np.mean(ages).item(), np.std(ages).item()
        ages_norm = (ages - ages_mean) / ages_std

        y = self._get_reformated(dataset, 'values')
        subjects_with_repeat = self._get_reformated_subjects(dataset)

        # model
        X = sm.add_constant(ages_norm, prepend=True, has_constant='add')

        if self.hyperparams['with_random_slope_age']:
            exog_re = X

            if self.hyperparams['force_independent_random_effects']:
                free = MixedLMParams.from_components(
                    fe_params=np.ones(2),
                    cov_re=np.eye(2)
                )
                self.sm_fit_parameters['free'] = free
                methods_not_compat_with_free = {'powell','nm'}.intersection(self.sm_fit_parameters['method']) # cf. statsmodels doc
                if len(methods_not_compat_with_free) > 0:
                    warnings.warn(f"<!> Methods {'powell','nm'} are not compatible with `force_independent_random_effects`")
        else:
            exog_re = None # random_intercept only

        lme = MixedLM(y, X, subjects_with_repeat, exog_re, missing='raise')
        fitted_lme = lme.fit(**self.sm_fit_parameters)

        try:
            cov_re_unscaled_inv = np.linalg.inv(fitted_lme.cov_re_unscaled)
        except np.linalg.LinAlgError:
            raise ValueError("Cannot predict random effects from "
                             "singular covariance structure.")

        parameters = {
            "ages_mean": ages_mean,
            "ages_std": ages_std,
            "fe_params": fitted_lme.fe_params,
            "cov_re": fitted_lme.cov_re,
            "cov_re_unscaled_inv": cov_re_unscaled_inv, # really useful for personalization
            "noise_std": fitted_lme.scale ** .5,
            "bse_fe": fitted_lme.bse_fe,
            "bse_re": fitted_lme.bse_re
        }

        # update model parameters
        model.load_parameters(parameters)

        # display `(fitted_lme.resid ** 2).mean() ** .5` instead?
        print_noise = '{:.4f}'.format(model.parameters['noise_std'].item())
        print("The standard deviation of the noise at the end of the calibration is: " + print_noise)

    @staticmethod
    def _get_reformated(dataset, elem):
        # reformat ages
        dataset_elem = getattr(dataset, elem)
        # flatten
        flat_elem = torch.flatten(dataset_elem).numpy()
        # remove padding & nans
        final_elem = flat_elem[torch.flatten(dataset.mask > 0)]
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
        """
        Not implemented.
        """
        return
