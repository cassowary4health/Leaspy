from typing import Dict, Hashable, Union

import numpy as np
import torch

from leaspy.exceptions import LeaspyInputError, LeaspyModelInputError


class OrdinalModelMixin:
    """Mix-in to add some useful properties & methods for models supporting the ordinal and ranking noise (univariate or multivariate)."""

    ## PUBLIC

    @property
    def is_ordinal(self) -> bool:
        """Property to check if the model is of ordinal sub-type."""
        return self.noise_model in ['ordinal', 'ordinal_ranking']

    def postprocess_model_estimation(self, estimation: np.ndarray, *, ordinal_method: str, **kws) -> Union[np.ndarray, Dict[Hashable, np.ndarray]]:
        """
        Extra layer of processing used to output nice estimated values in main API `Leaspy.estimate`.

        Parameters
        ----------
        estimation : numpy.ndarray[float]
            The raw estimated values by model (from `compute_individual_trajectory`)
        ordinal_method : str
            <!> Only used for ordinal models.
            * 'MLE' or 'maximum_likelihood' returns maximum likelihood estimator for each point (int)
            * 'E' or 'expectation' returns expectation (float)
            * 'P' or 'probabilities' returns probabilities of all-possible levels for a given feature:
              {feature_name: array[float]<0..max_level_ft>}
        **kws
            Some extra keywords arguments that may be handled in the future.

        Returns
        -------
        numpy.ndarray[float] or dict[str, numpy.ndarray[float]]
            Post-processed values.
            In case using 'probabilities' mode, the values are a dictionary with keys being:
            `(feature_name: str, feature_level: int<0..max_level_for_feature>)`
            Otherwise it is a standard numpy.ndarray corresponding to different model features (in order)
        """
        if not self.is_ordinal:
            return estimation

        if ordinal_method in {'MLE', 'maximum_likelihood'}:
            return estimation.argmax(axis=-1)
        elif ordinal_method in {'E', 'expectation'}:
            return np.flip(estimation, axis=-1).cumsum(axis=-1).sum(axis=-1) - 1.
        elif ordinal_method in {'P', 'probabilities'}:
            # we construct a dictionary with the appropriate keys
            d_ests = {}
            for ft_i, feat in enumerate(self.ordinal_infos["features"]):
                for ft_lvl in range(0, feat["max_level"] + 1):
                    d_ests[(feat["name"], ft_lvl)] = estimation[..., ft_i, ft_lvl]

            return d_ests
        else:
            raise LeaspyInputError("`ordinal_method` should be in: {'maximum_likelihood', 'MLE', 'expectation', 'E', 'probabilities', 'P'}"
                                   f" not {ordinal_method}")


    def compute_likelihood_from_ordinal_cdf(self, model_values: torch.Tensor) -> torch.Tensor:
        """
        Computes the likelihood of an ordinal model assuming that the model_values are the CDF.

        Parameters
        ----------
        model_values : `torch.Tensor`
            Cumulative distribution values : model_values[..., l] is the proba to be superior or equal to l+1
            Dimensions are:
            * 0=individual
            * 1=visit
            * 2=feature
            * 3=ordinal_level
            * [4=individual_parameter_dim_when_gradient]

        Returns
        -------
        likelihood : `torch.Tensor` (same shape as input)
            likelihood[..., l] is the proba to be equal to l
        """
        # nota: torch.diff was introduced in v1.8 but would not highly improve performance of this routine anyway
        s = list(model_values.shape)
        s[3] = 1
        mask = self.ordinal_infos["mask"]
        if len(s) == 5:  # in the case of gradient we added a dimension
            mask = mask.unsqueeze(-1)
            first_row = torch.zeros(size=tuple(s)).float()  # gradient(P>=0) = 0
        else:
            first_row = torch.ones(size=tuple(s)).float()  # (P>=0) = 1
        model = model_values * mask
        cdf_sup = torch.cat([first_row, model], dim=3)
        last_row = torch.zeros(size=tuple(s)).float()
        cdf_inf = torch.cat([model, last_row], dim=3)
        likelihood = cdf_sup - cdf_inf

        return likelihood


    ## PRIVATE

    @property
    def _attributes_factory_ordinal_kws(self) -> dict:
        # we put this here because of the
        return dict(ordinal_infos=getattr(self, 'ordinal_infos',  None))

    def _export_extra_ordinal_settings(self, model_settings) -> None:

        if self.is_ordinal:
            model_settings['batch_deltas_ordinal'] = self.ordinal_infos["batch_deltas"]

    def _handle_ordinal_hyperparameters(self, hyperparameters) -> tuple:
        # return a tuple of extra hyperparameters that are recognized

        if not self.is_ordinal:
            return tuple()  # no extra hyperparameters recognized

        if self.name not in {'logistic', 'univariate_logistic'}:
            raise LeaspyModelInputError(f"Noise model 'ordinal' is only compatible with 'logistic' and 'univariate_logistic' models, not {self.name}")

        if hasattr(self, 'ordinal_infos'):
            self.ordinal_infos["batch_deltas"] = hyperparameters.get('batch_deltas_ordinal',
                                                                     self.ordinal_infos["batch_deltas"])
        else:
            # initialize the ordinal_infos dictionary
            self.ordinal_infos = {"batch_deltas": hyperparameters.get('batch_deltas_ordinal', False),
                                  "max_level": None,
                                  "features": [],
                                  "mask": None,
                                 }

        return ('batch_deltas_ordinal',)

    def _initialize_MCMC_toolbox_ordinal_priors(self) -> None:

        if not self.is_ordinal:
            return

        if self.ordinal_infos['batch_deltas']:
            self.MCMC_toolbox['priors']['deltas_std'] = 0.1
        else:
            for feat in self.ordinal_infos["features"]:
                self.MCMC_toolbox['priors'][f'deltas_{feat["name"]}_std'] = 0.1

    def _update_MCMC_toolbox_ordinal(self, vars_to_update: tuple, realizations, values: dict) -> None:
        # update `values` dict in-place

        if not self.is_ordinal:
            return

        update_all = 'all' in vars_to_update
        if self.ordinal_infos['batch_deltas']:
            if update_all or 'deltas' in vars_to_update:
                values['deltas'] = realizations['deltas'].tensor_realizations
        else:
            for feat in self.ordinal_infos["features"]:
                if update_all or 'deltas_'+feat["name"] in vars_to_update:
                    values['deltas_'+feat["name"]] = realizations['deltas_'+feat["name"]].tensor_realizations

    def _add_ordinal_tensor_realizations(self, realizations, dict_to_update: dict) -> None:
        if not self.is_ordinal:
            return

        if self.ordinal_infos['batch_deltas']:
            dict_to_update['deltas'] = realizations['deltas'].tensor_realizations
        else:
            for feat in self.ordinal_infos["features"]:
                dict_to_update['deltas_'+feat["name"]] = realizations['deltas_'+feat["name"]].tensor_realizations

    def _add_ordinal_sufficient_statistics(self, suff_stats: dict, dict_to_update: dict) -> None:
        if not self.is_ordinal:
            return

        # The only difference with `_add_ordinal_tensor_realizations` is that suff_stats is a dict of Tensors and not
        # a CollectionRealizations object (which needs) to fetch the `.tensor_realizations` attribute...
        if self.ordinal_infos['batch_deltas']:
            dict_to_update['deltas'] = suff_stats['deltas']
        else:
            for feat in self.ordinal_infos["features"]:
                dict_to_update['deltas_'+feat["name"]] = suff_stats['deltas_'+feat["name"]]


    def _rebuild_ordinal_infos_from_model_parameters(self) -> None:

        # is this an ordinal model?
        if not self.is_ordinal:
            return

        # if yes: re-build the number of levels per feature
        deltas_p = {k: v for k, v in self.parameters.items() if k.startswith('deltas')}

        if self.ordinal_infos['batch_deltas']:
            assert deltas_p.keys() == {'deltas'}
            # Find ordinal infos from the delta values themselves
            bool_array = (self.parameters['deltas'] != 0).int()
            self.ordinal_infos["max_level"] = bool_array.shape[1] + 1
            for i, feat in enumerate(self.features):
                bool_array_ft = bool_array[i, :]
                if 0 in bool_array_ft:
                    max_lvl_ft = bool_array_ft.argmin().item() + 1
                else:
                    max_lvl_ft = self.ordinal_infos["max_level"]
                self.ordinal_infos["features"].append({"name": feat, "max_level": max_lvl_ft})
        else:
            assert deltas_p.keys() == {f'deltas_{ft}' for ft in self.features}
            for k, v in deltas_p.items():
                feat = k[7:]  #k[7:] removes the deltas_ to extract the feature's name
                self.ordinal_infos["features"].append({"name": feat, "max_level": v.shape[0] + 1})

            self.ordinal_infos["max_level"] = max([feat["max_level"] for feat in self.ordinal_infos["features"]])

        # re-build the mask to account for possible difference in levels per feature
        self.ordinal_infos["mask"] = torch.cat([
            torch.cat([
                torch.ones((1,1,1,feat['max_level'])),
                torch.zeros((1,1,1,self.ordinal_infos['max_level'] - feat['max_level'])),
            ], dim=-1) for feat in self.ordinal_infos["features"]
        ], dim=2)

    def _get_deltas(self, attribute_type: str) -> torch.Tensor:
        """
        Get the deltas attribute for ordinal models.

        Parameters
        ----------
        attribute_type: None or 'MCMC'

        Returns
        -------
        The deltas in the ordinal model
        """
        return self._call_method_from_attributes('get_deltas', attribute_type)

    def _add_ordinal_random_variables(self, variables_infos: dict) -> None:

        if not self.is_ordinal:
            return

        if self.ordinal_infos['batch_deltas']:
            # Instead of a sampler for each feature, sample deltas for all features in one sampler class
            max_level = self.ordinal_infos["max_level"]
            deltas_infos = {
                "name": "deltas",
                "shape": torch.Size([self.dimension, max_level - 1]),
                "type": "population",
                "rv_type": "multigaussian",
                "scale": .5,
                "mask": self.ordinal_infos["mask"][0,0,:,1:], # cut the zero level
            }
            variables_infos['deltas'] = deltas_infos
        else:
            # For each feature : create a sampler for deltas of size (max_level_of_the_feature - 1)
            for feat in self.ordinal_infos["features"]:
                deltas_infos = {
                    "name": "deltas_"+feat["name"],
                    "shape": torch.Size([feat["max_level"] - 1]),
                    "type": "population",
                    "rv_type": "gaussian",
                    "scale": .5,
                }
                variables_infos['deltas_'+feat["name"]] = deltas_infos

        # Finally: change the v0 scale since it has not the same meaning
        if 'v0' in variables_infos:  # not in univariate case!
            variables_infos['v0']['scale'] = 0.1
